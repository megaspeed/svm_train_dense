#include "common.cpp"
#include <cuda_runtime_api.h>
#include <float.h>
#include "device_launch_parameters.h"
#include "kernels.cu"
#include <list>
#include <algorithm>


int cache_hit = 0;
void Reduce_step(int *d_y, float *d_a, float *d_f, float *d_B, unsigned int *d_I, float *param, int ntraining, int nblocks,
				 float *h_B, unsigned int *h_I, float* h_B_global, unsigned int *h_I_global, int *active, int ntasks)
{
	int smem = MAXTHREADS*(sizeof(float) + sizeof(int));
	for (int itask = 0; itask < ntasks; itask++)
	{
		if (active[itask] == 1)
		{
			Local_Reduce_Min<<<nblocks, MAXTHREADS, smem>>>(d_y, &d_a[itask*ntraining], &d_f[itask*ntraining], &d_B[itask*2*nblocks], &d_I[itask*2*nblocks], param[2*itask], ntraining);//
			Local_Reduce_Max<<<nblocks, MAXTHREADS, smem>>>(d_y, &d_a[itask*ntraining], &d_f[itask*ntraining], &d_B[itask*2*nblocks+nblocks], &d_I[itask*2*nblocks+nblocks], param[2*itask], ntraining);//
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
					global_Iup = h_I[itask*2*nblocks+nblocks+i];
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

bool check_cache(unsigned int irow, unsigned int *cached_row, std::list<std::pair<unsigned int,unsigned int>> *cache, int cache_size)
{
	unsigned int pos = 0;
	std::list<std::pair<unsigned int, unsigned int>>::iterator findIter;
	for (findIter = cache->begin(); findIter != cache->end(); ++findIter, ++pos)
	{
		if (irow == findIter->first)
		{
			*cached_row = findIter->second;
			cache->remove(*findIter);
			cache->push_front(std::make_pair(irow, *cached_row));
			cache_hit++;
			return false;
		}
	}

	if (cache->size() == cache_size)
	{
		*cached_row = (--findIter)->second;
		cache->pop_back();
	}
	else
	{
		*cached_row = pos;
	}
	cache->push_front(std::make_pair(irow, *cached_row));
	return true;	
}
bool chech_condition(float* B, int *active_task, int ntasks)
{
	bool run = false;
	for (int i = 0; i < ntasks; i++)
	{
		if (B[2*i+1] <= B[2*i] + 2*TAU)
		{
			active_task[i] = 0;
		}
		run = run||active_task[i];
	}
	return run;
}

void cross_validation(svm_sample *train, svm_model *model)
{
	cudaEvent_t start, stop;
	cudaEventCreate ( &start );
	cudaEventCreate ( &stop  );

	int ntasks = model->ntasks;
	int nTV = train->nTV;
	int nfeatures = model->nfeatures;
	//Grid configuration
	int nthreads = MAXTHREADS;
	int nblocks = min(MAXBLOCKS, (nthreads + nTV - 1)/nthreads);

	float *d_TV = 0;//training vectors
	cudaMalloc((void**) &d_TV, nTV*nfeatures*sizeof(float));
	cudaMemcpy(d_TV, train->TV, nTV*nfeatures*sizeof(float),cudaMemcpyHostToDevice);

	float *d_SV = 0;//support vectors

	float *d_params = 0;// binary labels
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

	initialization<<<dim3(nblocks, 1), dim3(nthreads, ntasks)>>>(d_a, d_f, d_y, nTV);
	Reduce_step(d_y, d_a, d_f, d_B, d_I, model->params, nTV, nblocks, h_B, h_I, h_B_global, h_I_global, h_active, ntasks);

	unsigned int remainingMemory;
	unsigned int totalMemory;
	cudaMemGetInfo(&remainingMemory, &totalMemory);

	printf("%u bytes of memory found on device, %u bytes currently free\n", totalMemory, remainingMemory);

	int sizeOfCache = remainingMemory/(nTV*sizeof(float));

	sizeOfCache = (int)((float)sizeOfCache*KMEM);
	if (nTV < sizeOfCache)
		sizeOfCache = nTV;

	printf("%u rows of kernel matrix will be cached (%u bytes per row)\n", sizeOfCache, nTV*sizeof(float));

	float *d_k = 0;// gramm matrix
	cudaMalloc((void**) &d_k, sizeOfCache*nTV*sizeof(float));

	cudaStream_t *stream = (cudaStream_t*)malloc(2*sizeof(cudaStream_t));
	for (int i = 0; i < 2; i++)
	{
		cudaStreamCreate(&stream[i]);
	}
	
	int iter = 0;
	std::list<std::pair<unsigned int, unsigned int>>cache;

	while (chech_condition(h_B_global, h_active, ntasks))
	{
		++iter;	
		for (int itask = 0; itask < ntasks; itask++)
		{
			if (h_active[itask] == 1)
			{
				if(check_cache(h_I_global[2*itask], &h_I_cache[2*itask], &cache, sizeOfCache))		//Iup - second
					get_row<<<nblocks, nthreads,0>>>(d_k, d_TV, model->params[2*itask+1], nfeatures, h_I_global[2*itask], h_I_cache[2*itask], nTV);
				if(check_cache(h_I_global[2*itask+1], &h_I_cache[2*itask+1], &cache, sizeOfCache))//Ilow - fist
					get_row<<<nblocks, nthreads,0>>>(d_k, d_TV, model->params[2*itask+1], nfeatures, h_I_global[2*itask+1], h_I_cache[2*itask+1], nTV);
			}
		}
		cudaMemcpy(d_active, h_active, ntasks*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(d_I_cache, h_I_cache, ntasks*2*sizeof(unsigned int),cudaMemcpyHostToDevice);
		cudaMemcpy(d_I_global, h_I_global, ntasks*2*sizeof(unsigned int),cudaMemcpyHostToDevice);

		Update<<<1,ntasks>>>(d_k, d_y, d_f, d_a, d_delta_a, d_I_global, d_I_cache, d_params, d_active, nTV);

		Map<<<dim3(nblocks, 1), dim3(nthreads, ntasks)>>>(d_f, d_k, d_y, d_delta_a, d_I_global, d_I_cache, d_active, nTV);
		cudaDeviceSynchronize();

		Reduce_step(d_y, d_a, d_f, d_B, d_I, model->params, nTV, nblocks, h_B, h_I, h_B_global, h_I_global, h_active, ntasks);
	}
	printf("All tasks convergented\n");

	//output only for best rate result
	float *l_sv = (float*)malloc(nTV*sizeof(float));
	float *SV = (float*)malloc(nTV*nfeatures*sizeof(float));
	cudaMemcpy(l_sv, d_a, nTV*sizeof(float), cudaMemcpyDeviceToHost);
	model->b = (float*)malloc(sizeof(float));
	*model->b = (h_B_global[1]+h_B_global[0])/2;

	int nSV = 0;
	for (int i = 0; i < nTV; i++)
	{
		if (l_sv[i] != 0)
		{
			if (i != nSV)
			{
				l_sv[nSV] = train->l_TV[i]*l_sv[i];
				for (int j = 0; j < nfeatures; j++)
					SV[nSV*nfeatures+j] = train->TV[i*nfeatures+j];
			}			
			++nSV;
		}
	}
	model->nSV = nSV;
	model->l_SV=(float*)realloc(l_sv, nSV*sizeof(float));
	model->SV_dens=(float*)realloc(SV, nSV*nfeatures*sizeof(float));


	cudaDeviceReset();
}
void set_model_param(svm_model *model, float cbegin, int c_col, float gbegin, int g_col)
{
	model->C = (float*)malloc((c_col)*sizeof(float));
	model->coef_gamma = (float*)malloc((g_col)*sizeof(float));
	model->C[0] = cbegin;
	for (int i = 1; i < c_col; i++)
	{
		model->C[i] = model->C[i-1]/2;
	}

	model->coef_gamma[0] = 1./model->nfeatures;
	if(g_col > 1)
		model->coef_gamma[1] = gbegin;
	for (int i = 2; i < g_col; i++)
	{
		model->coef_gamma[i] = model->coef_gamma[i-1]/2;
	}

	model->ntasks = c_col*g_col;
	model->params = (float*)malloc(model->ntasks*2*sizeof(float));
	for (int i = 0; i < c_col; i++)
	{
		for (int j = 0; j < g_col; j++)
		{
			model->params[2*(i*c_col+j)] = model->C[i];
			model->params[2*(i*c_col+j)+1] = model->coef_gamma[j];
		}
	}
	model->kernel_type = 0;
	model->svm_type = 0;
}

int main(int argc, char **argv)
{
	FILE *input, *output;
	if (argc==1)
	{
		argc = 4;
		argv[1] = "C:\\Data\\b.txt";
		argv[2] = "C:\\Data\\b.model";
		argv[3] = "10";
		//argv[1] = "C:\\Data\\a9a";
		//argv[2] = "C:\\Data\\a9a.model";
		//argv[3] = "123";
		//argv[1] = "C:\\Data\\mushrooms";
		//argv[2] = "C:\\Data\\mushrooms.model";
		//argv[3] = "112";
		//argv[1] = "C:\\Data\\ijcnn1";
		//argv[2] = "C:\\Data\\ijcnn1.model";
		//argv[3] = "22";

	}
	if(argc<4)
		exit_with_help();
	struct svm_model *model = (svm_model*)malloc(sizeof(svm_model));
	struct svm_sample *train = (svm_sample*)malloc(sizeof(svm_sample));
	sscanf(argv[3],"%d",&model->nfeatures);

	set_model_param(model, 1, 1, 1, 1);
	if((input = fopen(argv[1],"r")) == NULL)
	{
		fprintf(stderr,"can't open training file %s\n",argv[1]);
		exit(1);
	}

	if((output = fopen(argv[2],"w")) == NULL)
	{
		fprintf(stderr,"can't create model file %s\n",argv[2]);
		exit(1);
	}

	parse_TV(input, train, model);
	set_labels(train, model);
	cuResetTimer();
	cross_validation(train, model);
	printf("time %f cache hits %d\n", cuGetTimer(), cache_hit);
	save_model(output, model);

	return 0;
}
