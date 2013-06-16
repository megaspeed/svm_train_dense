#include "common.cpp"
#include <cuda_runtime_api.h>
#include <float.h>
#include "device_launch_parameters.h"
#include "kernels.cu"
# define cudaCheck\
 {\
 cudaError_t err = cudaGetLastError ();\
 if ( err != cudaSuccess ){\
 printf(" cudaError = '%s' \n in '%s' %d\n", cudaGetErrorString( err ), __FILE__ , __LINE__ );\
 exit(0);}}


void classifier(svm_model *model, svm_sample *test, float *rate)
{
	int nTV = test->nTV;
	int nSV = model->nSV;
	int nfeatures = model->nfeatures;

	float *d_TV = 0;	
	float *d_SV = 0;
	cudaMalloc((void**) &d_SV, nSV*nfeatures*sizeof(float));cudaCheck
	cudaMemcpy(d_SV, model->SV_dens, nSV*nfeatures*sizeof(float),cudaMemcpyHostToDevice);cudaCheck

	float *d_l_SV = 0;
	cudaMalloc((void**) &d_l_SV, nSV*sizeof(float));cudaCheck
	cudaMemcpy(d_l_SV, model->l_SV, nSV*sizeof(float),cudaMemcpyHostToDevice);cudaCheck

	size_t remainingMemory = 0;
	size_t totalMemory = 0;
	cudaMemGetInfo(&remainingMemory, &totalMemory);	cudaCheck
	int cache_size = remainingMemory/(nSV*sizeof(float)); // # of TVs in cache
	if (nTV <= cache_size){	cache_size = nTV; }

	cudaMalloc((void**) &d_TV, cache_size*nfeatures*sizeof(float));cudaCheck

	int nthreads = MAXTHREADS;
	int nblocks_cache = min(MAXBLOCKS, (cache_size + nthreads - 1)/nthreads);
	int nblocks_SV = min(MAXBLOCKS, (nSV + nthreads - 1)/nthreads);
	dim3 dim_block = dim3(nblocks_cache, 1, 1);
	dim3 dim_thread = dim3(MAXTHREADS, 1, 1);
	// Allocate device memory for F
	float* h_fdata= (float*) malloc(nblocks_SV*sizeof(float));
	float* d_fdata=0;
	cudaMalloc((void**) &d_fdata, nblocks_SV*sizeof(float));cudaCheck
	int offset = 0;
	int num_of_parts =  (nTV + cache_size - 1)/cache_size;
	int* h_l_estimated = (int*)malloc(test->nTV*sizeof(int));
	for (int ipart = 0; ipart < num_of_parts; ipart++)
	{
		if ((ipart == (num_of_parts - 1)) && ((nTV - offset) != 0) )
		{
			cache_size = nTV - offset;
		}
		cudaMemcpy(d_TV, &test->TV[offset*nfeatures], cache_size*nfeatures*sizeof(float),cudaMemcpyHostToDevice);cudaCheck
			for (int i = 0; i < cache_size; i++)
			{				
				reduction<<<nblocks_SV, MAXTHREADS, MAXTHREADS*sizeof(float)>>>(d_SV, &d_TV[i*nfeatures], d_l_SV, nSV, nfeatures, model->coef_gamma, model->kernel_type, d_fdata);cudaCheck
				cudaMemcpy(h_fdata, d_fdata, nblocks_SV*sizeof(float), cudaMemcpyDeviceToHost); cudaCheck

				float sum = 0;
				for (int k = 0; k < nblocks_SV; k++)
					sum += h_fdata[k];

				sum -= model->b;
				if (sum > 0)
				{
					h_l_estimated[i + offset] = 1;
				}
				else
				{
					h_l_estimated[i + offset] = -1;
				}
			}
			offset += cache_size;
	}
	cudaFree(d_fdata);cudaCheck
	cudaFree(d_l_SV);cudaCheck
	cudaFree(d_SV);cudaCheck
	cudaFree(d_TV);cudaCheck
	cudaDeviceReset();cudaCheck

	int errors=0;
	for (int i=0; i<test->nTV; i++)
	{
		if( test->l_TV[i]!=h_l_estimated[i])
		{
			errors++;
		}
	}
	*rate = (float)(test->nTV - errors)/test->nTV;
	
	free(h_l_estimated);
	free(h_fdata);
	printf("# of testing samples %i, # errors %i, Rate %f\n", nTV, errors, 100* (float)(nTV-errors)/nTV);
}
void Reduce_step(int *d_y, float *d_a, float *d_f, float *d_B, unsigned int *d_I, float *param, int ntraining, int nblocks,
				 float *h_B, unsigned int *h_I, float* h_B_global, unsigned int *h_I_global, int *active, int ntasks)
{
	int smem = MAXTHREADS*(sizeof(float) + sizeof(int));
	for (int itask = 0; itask < ntasks; itask++)
	{
		if (active[itask] == 1)
		{
			Local_Reduce_Min<<<nblocks, MAXTHREADS, smem>>>(d_y, &d_a[itask*ntraining], &d_f[itask*ntraining], &d_B[itask*2*nblocks], &d_I[itask*2*nblocks], &param[2*itask], ntraining);

			Local_Reduce_Max<<<nblocks, MAXTHREADS, smem>>>(d_y, &d_a[itask*ntraining], &d_f[itask*ntraining], &d_B[itask*2*nblocks+nblocks], &d_I[itask*2*nblocks+nblocks], &param[2*itask], ntraining);
					//cudaDeviceSynchronize();
		}
	}
	cudaMemcpy(h_B, d_B, ntasks*2*nblocks*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_I, d_I, ntasks*2*nblocks*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	for (int itask = 0; itask < ntasks; itask++)
	{
		if (active[itask] == 1)
		{
			// Global reduction
			float global_Bup = h_B[itask*2*nblocks];
			float global_Blow = h_B[itask*2*nblocks+nblocks];
			int global_Iup = h_I[itask*2*nblocks];
			int global_Ilow = h_I[itask*2*nblocks+nblocks];

			for (int i = 1; i < nblocks; i++)
			{
				if (h_B[itask*2*nblocks+i] < global_Bup)
				{
					global_Bup = h_B[itask*2*nblocks+i];
					global_Iup = h_I[itask*2*nblocks+i];
				}
				if (h_B[itask*2*nblocks+nblocks + i] > global_Blow)
				{
					global_Blow = h_B[itask*2*nblocks+nblocks + i];
					global_Ilow = h_I[itask*2*nblocks+nblocks + i];
				}
			}

			h_B_global[itask*2] = global_Bup;
			h_B_global[itask*2+1] = global_Blow;
			h_I_global[itask*2] = global_Iup;
			h_I_global[itask*2+1] = global_Ilow;
		}
	}
}

void cross_validation(svm_sample *train, svm_model *model)
{
	int ntasks = model->ntasks;
	int nTV = train->nTV;
	int nfeatures = model->nfeatures;
	//Grid configuration
	int nthreads = MAXTHREADS;
	int nblocks = min(MAXBLOCKS, (nthreads + nTV - 1)/nthreads);
	int task_threads = min(1024/MAXTHREADS, ntasks);
	int task_blocks = (ntasks + task_threads - 1)/task_threads;

	float *d_TV = 0;//training vectors
	cudaMalloc((void**) &d_TV, nTV*nfeatures*sizeof(float));
	cudaMemcpy(d_TV, train->TV, nTV*nfeatures*sizeof(float),cudaMemcpyHostToDevice);

	float *d_SV = 0;//support vectors

	float *d_params = 0;
	cudaMalloc((void**) &d_params, 2*ntasks*sizeof(float));
	cudaMemcpy(d_params, model->params, 2*ntasks*sizeof(float),cudaMemcpyHostToDevice);

	int *d_y = 0;// binary labels
	cudaMalloc((void**) &d_y, nTV*sizeof(int));
	cudaMemcpy(d_y, train->l_TV, nTV*sizeof(int),cudaMemcpyHostToDevice);

	float *d_a = 0; //alphas
	cudaMalloc((void**) &d_a, ntasks*nTV*sizeof(float));

	float *d_f = 0;//object functions
	cudaMalloc((void**) &d_f, ntasks*nTV*sizeof(float));

	//locally reduced thresholds {Bup:Blow}
	float *h_B = (float*)malloc(2*nblocks*ntasks*sizeof(float));
	float *d_B = 0;
	cudaMalloc((void**) &d_B, 2*nblocks*ntasks*sizeof(float));

	//indeces of locally reduced Lagrange multipliers {Iup:Ilow}
	unsigned int *h_I = (unsigned int*)malloc(2*nblocks*ntasks*sizeof(unsigned int));
	unsigned int *d_I = 0; 
	cudaMalloc((void**) &d_I, 2*nblocks*ntasks*sizeof(unsigned int));

	//global tresholds {Bup:Blow}
	float *h_B_global = (float*)malloc(2*ntasks*sizeof(float));

	unsigned int *h_I_global = (unsigned int*)malloc(2*ntasks*sizeof(unsigned int));
	unsigned int *d_I_global = 0; 
	cudaMalloc((void**) &d_I_global, 2*ntasks*sizeof(unsigned int));
	unsigned int *h_I_cache = (unsigned int*)malloc(2*ntasks*sizeof(unsigned int));
	unsigned int *d_I_cache = 0; 
	cudaMalloc((void**) &d_I_cache, 2*ntasks*sizeof(unsigned int));

	float *d_delta_a = 0;
	cudaMalloc((void**) &d_delta_a, 2*ntasks*sizeof(float));

	int *h_active = (int*)malloc(ntasks*sizeof(int));
	for (int i = 0; i < ntasks; i++)
		h_active[i] = 1;

	int *d_active = 0;
	cudaMalloc((void**) &d_active, ntasks*sizeof(int));
	cudaMemcpy(d_active, h_active, ntasks*sizeof(int),cudaMemcpyHostToDevice);


	initialization<<<dim3(nblocks, task_blocks), dim3(nthreads, task_threads)>>>(d_a, d_f, d_y, nTV, ntasks);
	cudaDeviceSynchronize();
	Reduce_step(d_y, d_a, d_f, d_B, d_I, d_params, nTV, nblocks, h_B, h_I, h_B_global, h_I_global, h_active, ntasks);

	unsigned int remainingMemory;
	unsigned int totalMemory;
	cudaMemGetInfo(&remainingMemory, &totalMemory);
	//printf("%u bytes of memory found on device, %u bytes currently free\n", totalMemory, remainingMemory);
	int sizeOfCache = remainingMemory/(nTV*sizeof(float));
	if (nTV < sizeOfCache)
		sizeOfCache = nTV;
	sizeOfCache = max((int)((float)sizeOfCache*KMEM/16),2*ntasks);
	printf("%u rows of kernel matrix will be cached (%u bytes per row)\n", sizeOfCache, nTV*sizeof(float));
	float *d_k = 0;// gramm matrix
	cudaMalloc((void**) &d_k, sizeOfCache*nTV*sizeof(float));

	int iter = 0;
	std::list<std::pair<unsigned int,unsigned int> > cache;
	float ctime = 0;
	float cc ;
	while (chech_condition(h_B_global, h_active, ntasks, iter))
	{
		++iter;	
		cc = cuGetTimer();
		for (int itask = 0; itask < ntasks; itask++)
		{
			if (h_active[itask] == 1)
			{
				if(check_cache(h_I_global[2*itask], &h_I_cache[2*itask], &cache, sizeOfCache))		//Iup - second
				{
					get_row<<<nblocks, nthreads>>>(d_k, d_TV, nfeatures, h_I_global[2*itask], h_I_cache[2*itask], nTV);
				}
				if(check_cache(h_I_global[2*itask+1], &h_I_cache[2*itask+1], &cache, sizeOfCache))//Ilow - fist
				{
					get_row<<<nblocks, nthreads>>>(d_k, d_TV, nfeatures, h_I_global[2*itask+1], h_I_cache[2*itask+1], nTV);
				}
			}
		}
		ctime +=cuGetTimer() - cc;
		cudaMemcpy(d_active, h_active, ntasks*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(d_I_cache, h_I_cache, ntasks*2*sizeof(unsigned int),cudaMemcpyHostToDevice);
		cudaMemcpy(d_I_global, h_I_global, ntasks*2*sizeof(unsigned int),cudaMemcpyHostToDevice);

		Update1<<<1,ntasks>>>(d_k, d_y, d_f, d_a, d_delta_a, d_I_global, d_I_cache, d_params, d_active, nTV);
		cudaDeviceSynchronize();
		Map<<<dim3(nblocks, task_blocks), dim3(nthreads, task_threads)>>>(d_f, d_k, d_y, d_delta_a, d_I_global, d_I_cache, d_params, d_active, nTV);
		cudaDeviceSynchronize();
		Reduce_step(d_y, d_a, d_f, d_B, d_I, d_params, nTV, nblocks, h_B, h_I, h_B_global, h_I_global, h_active, ntasks);
	}
	printf("All tasks convergented in %f on %d iter cctime = %f\n", cuGetTimer(), iter, ctime);

	model->l_SV = (float*)malloc(nTV*ntasks*sizeof(float));
	cudaMemcpy(model->l_SV, d_a, nTV*ntasks*sizeof(float), cudaMemcpyDeviceToHost);

	model->mass_b = (float*)malloc(ntasks*sizeof(float));
	for (int itask = 0; itask < model->ntasks; itask++)
	{
		model->mass_b[itask] = (h_B_global[2*itask+1]+h_B_global[2*itask])/2;
	}
	cudaDeviceReset();
}

void classification( svm_sample *train, svm_sample *test, svm_model *model, FILE *output)
{
	int ntraining = train->nTV;
	float *buf_l = model->l_SV;
	float rate;
	float min_C = 0;
	float max_rate= 0;
	int max_rate_ind;
	for (int itask = 0; itask < model->ntasks; itask++)
	{
		float *SV = (float*)malloc(ntraining*model->nfeatures*sizeof(float));
		float *l_sv = (float*)malloc(ntraining*sizeof(float));
		model->l_SV = &buf_l[itask*ntraining];
		int nSV = 0;
		for (int i = 0; i < ntraining; i++)
		{
			if (model->l_SV[i] != 0)
			{
				l_sv[nSV] = train->l_TV[i]*model->l_SV[i];
				for (int j = 0; j < model->nfeatures; j++)
					SV[nSV*model->nfeatures+j] = train->TV[i*model->nfeatures+j];
				++nSV;
			}
		}
		if (nSV == 0)
		{
			printf("Task %d has bad parameters C=%f and gamma=%f #SV=%d\n",itask, model->params[2*itask], model->params[2*itask+1], nSV);
			continue;
		}

		model->nSV = nSV;
		model->b = model->mass_b[itask];
		model->C = model->params[2*itask];
		model->coef_gamma = model->params[2*itask+1];
		model->l_SV=(float*)realloc(l_sv, nSV*sizeof(float));
		model->SV_dens=(float*)realloc(SV, nSV*model->nfeatures*sizeof(float));
		classifier(model, test, &rate);
		save_model(output, model);
		if ((max_rate <= rate) && (min_C < model->params[2*itask]))
		{
			max_rate = rate;
			max_rate_ind = itask;
			min_C = model->params[2*itask];
		}
		free(model->l_SV);
		free(model->SV_dens);
		printf("Task %d occuracy is %f with C=%f and gamma=%f #SV=%d\n",itask, rate, model->params[2*itask], model->params[2*itask+1], nSV);
	}
	printf("best occuracy is %f with C=%f and gamma=%f\n", max_rate, model->params[2*max_rate_ind], model->params[2*max_rate_ind+1]);

	free(model->mass_b);
	//free(model->params);
	//free(model);
	//free(test->l_TV);
	//free(test->TV);
	//free(test);
}
void set_folds(svm_sample *train, svm_sample *test, int nfolds, int nfeatures)
{
	svm_sample *folds = (svm_sample*)malloc(sizeof(svm_sample));
	int n = train->nTV;
	int m = n/nfolds;
	int total_elem = nfolds*m;
	folds->l_TV = (int*)malloc(total_elem*sizeof(int));
	folds->TV = (float*)malloc(total_elem*nfeatures*sizeof(float));
	int shift = m*nfeatures;
	for (int i = 0; i < m; i++)
	{
		for (int ifold = 0; ifold < nfolds; ifold++)
		{
			folds->l_TV[ifold*m+i] = train->l_TV[i*nfolds+ifold];
			for (int k = 0; k < nfeatures; k++)
			{
				folds->TV[ifold*shift+i*nfeatures+k] = train->TV[(i*nfolds+ifold)*nfeatures+k];
			}
		}
	}
	free(train->TV);
	free(train->l_TV);
	train->nTV = (nfolds-1)*m;
	test->nTV = m;
	train->TV = folds->TV;
	train->l_TV = folds->l_TV;
	test->TV = &folds->TV[train->nTV*nfeatures];
	test->l_TV = &folds->l_TV[train->nTV];
}
int main(int argc, char **argv)
{
	FILE *input_train, *input_test, *output;
	if (argc==1)
	{
		argc = 5;
		argv[1] = "C:\\Data\\b.txt";
		argv[2] = "C:\\Data\\b.model";
		argv[3] = "C:\\Data\\b.txt";
		argv[4] = "10";
		//argv[1] = "C:\\Data\\a9a";
		//argv[2] = "C:\\Data\\a9a.model";
		//argv[3] = "C:\\Data\\a9a.t";
		//argv[4] = "123";
		//argv[1] = "C:\\Data\\mushrooms";
		//argv[2] = "C:\\Data\\mushrooms.model";
		//argv[3] = "C:\\Data\\mushrooms.t";
		//argv[4] = "112";
		//argv[1] = "C:\\Data\\ijcnn1";
		//argv[2] = "C:\\Data\\ijcnn1.model";
		//argv[3] = "C:\\Data\\ijcnn1.t";
		//argv[4] = "22";
		//argv[1] = "C:\\Data\\w8a";
		//argv[2] = "C:\\Data\\w8a.model";
		//argv[3] = "C:\\Data\\w8a.t";
		//argv[4] = "300";
		argv[1] = "C:\\Data\\cod-rna";
		argv[2] = "C:\\Data\\cod.model";
		argv[3] = "C:\\Data\\cod-rna.t";
		argv[4] = "8";
		//argv[1] = "C:\\Data\\cov";
		//argv[2] = "C:\\Data\\cov.model";
		//argv[3] = "C:\\Data\\cov.t";
		//argv[4] = "54";
	}
	if(argc<5)
		exit_with_help();
	svm_model *model = (svm_model*)malloc(sizeof(svm_model));
	svm_sample *train = (svm_sample*)malloc(sizeof(svm_sample));
	svm_sample *test = (svm_sample*)malloc(sizeof(svm_sample));
	sscanf(argv[4],"%d",&model->nfeatures);

	if((input_train = fopen(argv[1],"r")) == NULL)
	{
		fprintf(stderr,"can't open training file %s\n",argv[1]);
		exit(1);
	}

	if((output = fopen(argv[2],"w")) == NULL)
	{
		fprintf(stderr,"can't create model file %s\n",argv[2]);
		exit(1);
	}
	if((input_test = fopen(argv[3],"r")) == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv[1]);
		exit(1);
	}
	set_model_param(model, 1, 1, 0.03125, 1);
	converg_time= (float*)malloc(model->ntasks*sizeof(float));
	for (int itask = 0; itask < model->ntasks; itask++)
		converg_time[itask] = 0;
	parse_TV(input_test, test, model);
	parse_TV(input_train, train, model);
	set_labels(test, model);
	set_labels(train, model);
	//sort_by_class(train, model->nfeatures);
	
	//int nfolds = 5;
	//set_folds(train, test, nfolds, model->nfeatures);

	cuResetTimer();
	cross_validation(train, model);
	printf("Train time %f\n", cuGetTimer());
	classification(train, test, model, output);
	printf("Cache: hits %d miss %d percent of hits %f\n", cache_hit, cache_miss, (float)cache_hit/cache_miss*100);
	printf("Total time %f\n", cuGetTimer());
	cudaDeviceReset();
	return 0;
}
