#include "common.cpp"
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <float.h>
#include "device_launch_parameters.h"
#include "kernels.cu"
#include <list>
#include <algorithm>
# define cudaCheck\
 do{\
 cudaError_t err = cudaGetLastError ();\
 if ( err != cudaSuccess ){\
 printf(" cudaError = '%s' \n in '%s' %d\n", cudaGetErrorString( err ), __FILE__ , __LINE__ );\
 exit(0);}}while(0);

int cache_hit = 0;
void Reduce_step(int *d_y, float *d_a, float *d_f, float *d_B, unsigned int *d_I, float C, int ntraining, int nblocks,
				 float *h_B, unsigned int *h_I, float* h_B_global, unsigned int *h_I_global)
{
	int smem = MAXTHREADS*(sizeof(float) + sizeof(int));
	Local_Reduce_Min<<<nblocks, MAXTHREADS, smem>>>(d_y, d_a, d_f, d_B, d_I, C, ntraining);
	Local_Reduce_Max<<<nblocks, MAXTHREADS, smem>>>(d_y, d_a, d_f, &d_B[nblocks], &d_I[nblocks], C, ntraining);

	cudaMemcpy(h_B, d_B, 2*nblocks*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_I, d_I, 2*nblocks*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	// Global reduction
	float global_Bup = h_B[0];
	float global_Blow = h_B[nblocks];
	int global_Iup = h_I[0];
	int global_Ilow = h_I[nblocks];

	for (int i = 1; i < nblocks; i++)
	{
		if (h_B[i] < global_Bup)
		{
			global_Bup = h_B[i];
			global_Iup = h_I[i];
		}
		if (h_B[nblocks + i] > global_Blow)
		{
			global_Blow = h_B[nblocks + i];
			global_Ilow = h_I[nblocks + i];
		}
	}

	h_B_global[0] = global_Bup;
	h_B_global[1] = global_Blow;
	h_I_global[0] = global_Iup;
	h_I_global[1] = global_Ilow;
}
float scal(float *x, float *y, int n)
{
	float val = 0;
	for (int i = 0; i < n; i++)
	{
		val += x[i]*y[i];
	}
	return val;
}
float getK(float *tv, int ncol, float gamma, int i, int j)
{
	float val = scal(&tv[i*ncol], &tv[i*ncol], ncol)+
				scal(&tv[j*ncol], &tv[j*ncol], ncol)-
				2*scal(&tv[i*ncol], &tv[j*ncol], ncol);
	return exp(-gamma*val);
}
void init(float *a, int *y, float *f, int ntv)
{
	for (int i = 0; i < ntv; i++)
	{
		a[i] = 0.;
		f[i] = -1.*y[i]; 
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

void reduction(int *y, float *a, float *f, float C, int nTV, float *blow, float *bup, unsigned int *iup, unsigned int *ilow)
{
	*bup = FLT_MAX;
	*blow = -1.*FLT_MAX;
	for (int i = 0; i < nTV; i++)
	{
		if (( (y[i]==1 && a[i]>0 && a[i]<C) || (y[i]==-1 && a[i]>0 && a[i]<C)) ||
			(y[i]==1 && a[i]==0)||(y[i]==-1 && a[i]==C))
		{
			if (*bup > f[i])
			{
				*bup = f[i];
				*iup = i;
			}
		}//bup
		if (( (y[i]==1 && a[i]>0 && a[i]<C) ||	(y[i]==-1 && a[i]>0 && a[i]<C)) ||
			(y[i]==1 && a[i]==C)||(y[i]==-1 && a[i]==0))
		{
			if (*blow < f[i])
			{
				*blow = f[i];
				*ilow = i;
			}
		}//blow
	}
}
void train_model(svm_sample *train, svm_model *model)
{
	cudaEvent_t start, stop;
	cudaEventCreate ( &start );
	cudaEventCreate ( &stop  );

	int nTV = train->nTV;
	int nfeatures = model->nfeatures;
	//Grid configuration
	int nthreads = MAXTHREADS;
	int nblocks = min(MAXBLOCKS, (nthreads + nTV - 1)/nthreads);

	float *d_TV = 0;//training vectors
	cudaMalloc((void**) &d_TV, nTV*nfeatures*sizeof(float));
	cudaMemcpy(d_TV, train->TV, nTV*nfeatures*sizeof(float),cudaMemcpyHostToDevice);

	float *d_SV = 0;//support vectors

	float C = model->C;// regularisation parameter

	float gamma = model->coef_gamma;// regularisation parameter

	int *d_y = 0;// binary labels
	cudaMalloc((void**) &d_y, nTV*sizeof(int));
	cudaMemcpy(d_y, train->l_TV, nTV*sizeof(int),cudaMemcpyHostToDevice);

	float *d_a = 0; //alphas
	cudaMalloc((void**) &d_a, nTV*sizeof(float));

	float *d_f = 0;//object functions
	cudaMalloc((void**) &d_f, nTV*sizeof(float));

	
	

	//locally reduced thresholds {Bup:Blow}
	float *h_B = (float*)malloc(2*nblocks*sizeof(float));
	float *d_B = 0;
	cudaMalloc((void**) &d_B, 2*nblocks*sizeof(float));

	//indeces of locally reduced Lagrange multipliers {Iup:Ilow}
	unsigned int *h_I = (unsigned int*)malloc(2*nblocks*sizeof(unsigned int));
	unsigned int *d_I = 0; 
	cudaMalloc((void**) &d_I, 2*nblocks*sizeof(unsigned int));

	//global tresholds {Bup:Blow}
	float *h_B_global = (float*)malloc(2*sizeof(float));
	
	unsigned int *h_I_global = (unsigned int*)malloc(2*sizeof(unsigned int));
	unsigned int *d_I_global = 0; 
	cudaMalloc((void**) &d_I_global, 2*sizeof(unsigned int));
	unsigned int *h_I_cache = (unsigned int*)malloc(2*sizeof(unsigned int));
	unsigned int *d_I_cache = 0; 
	cudaMalloc((void**) &d_I_cache, 2*sizeof(unsigned int));

	float *d_delta_a = 0;
	cudaMalloc((void**) &d_delta_a, 2*sizeof(float));

	int *h_active = (int*)malloc(sizeof(int));
	h_active[0] = 0;
	int *d_active = 0;
	cudaMalloc((void**) &d_active, sizeof(int));

	initialization<<<nblocks, nthreads>>>(d_a, d_f, d_y, nTV);
	Reduce_step(d_y, d_a, d_f, d_B, d_I, C, nTV, nblocks, h_B, h_I, h_B_global, h_I_global);
	void* temp;
	size_t rowPitch;
	unsigned int remainingMemory;
	unsigned int totalMemory;
	cudaMallocPitch(&temp, &rowPitch, nTV*sizeof(float), 1);
	cudaFree(temp);

	cudaMemGetInfo(&remainingMemory, &totalMemory);

	printf("%u bytes of memory found on device, %u bytes currently free\n", totalMemory, remainingMemory);

	int sizeOfCache = remainingMemory/(nTV*sizeof(float));

	sizeOfCache = (int)((float)sizeOfCache*KMEM);
	if (nTV < sizeOfCache)
	{
		sizeOfCache = nTV;
	}

	printf("%u rows of kernel matrix will be cached (%u bytes per row)\n", sizeOfCache, (int)rowPitch);

	float* d_k ;
	size_t cachePitch;
	cudaMallocPitch((void**)&d_k, &cachePitch, nTV*sizeof(float), sizeOfCache);cudaCheck
	int width = cachePitch/sizeof(float);
	//float *d_k = 0;// gramm matrix
	//cudaMalloc((void**) &d_k, sizeOfCache*nTV*sizeof(float));
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);cudaCheck
	int iter = 0;
	std::list<std::pair<unsigned int, unsigned int>>cache;
	while (h_B_global[1]>h_B_global[0] + 2*TAU)
	{
		++iter;		
		if(check_cache(h_I_global[0], &h_I_cache[0], &cache, sizeOfCache))		//Iup - second
			get_row<<<nblocks, nthreads>>>(d_k, d_TV, gamma, nfeatures, h_I_global[0], h_I_cache[0], nTV, width);
		if(check_cache(h_I_global[1], &h_I_cache[1], &cache, sizeOfCache))//Ilow - fist
			get_row<<<nblocks, nthreads,0,stream1>>>(d_k, d_TV, gamma, nfeatures, h_I_global[1], h_I_cache[1], nTV, width);
		cudaMemcpy(d_I_cache, h_I_cache, 2*sizeof(unsigned int),cudaMemcpyHostToDevice);
		cudaMemcpy(d_I_global, h_I_global, 2*sizeof(unsigned int),cudaMemcpyHostToDevice);

		Update<<<1,1>>>(d_k, d_y, d_f, d_a, d_delta_a, d_I_global, d_I_cache, C, d_active, nTV, width);
		
		Map<<<nblocks, nthreads>>>(d_f, d_k, d_y, d_delta_a, d_I_global, d_I_cache, nTV, width);
		
		Reduce_step(d_y, d_a, d_f, d_B, d_I, C, nTV, nblocks, h_B, h_I, h_B_global, h_I_global);

	}
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
	model->kernel_type = 0;
	model->svm_type = 0;
	
	cudaDeviceReset();
}


void calc_alphas(float *tv, int *y, float *a, float *f, float *anewup, float *anewlow, int iup, int ilow, int ntv, int nfeatures, float C, float betta, bool *done)
{
	float eps = 0.000001;
	float upold = a[iup];
	float lowold = a[ilow];
	float gamma;
	int s = y[ilow]*y[iup];
	float L;
	float H;
	if (y[ilow] == y[iup])
		gamma = lowold + upold;
	else
		gamma = lowold - upold;

	if (s == 1)
	{
		L = max(0, gamma - C);
		H = min(C, gamma);
	}
	else
	{
		L = max(0, -gamma);
		H = min(C, C - gamma);
	}
	if (H <= L)
		*done = true;
	float nu = 2*getK(tv, nfeatures, betta, ilow, iup) - 2;
	float upnew;
	float lownew;
	if (nu < 0)
	{
		upnew = upold - (y[iup]*(f[ilow]-f[iup])/nu);
		if (upnew < L)
			upnew = L;
		else if (upnew > H)
				upnew = H;
	}
	else
	{
		float slope= y[iup]*(f[ilow]-f[iup]);
		float change= slope * (H-L);
		if(fabs(change)>0.0f)
		{
			if(slope>0.0f)
				upnew= H;
			else
				upnew= L;
		}
		else
			upnew= upold;

		if( upnew > C - eps * C)
			upnew=C;
		else if (upnew < eps * C)
				upnew=0.0f;
	}
	if( fabs( upnew - upold) < eps * ( upnew + upold + eps))
		*done = true;
	if (s == 1)
		lownew = gamma - upnew;
	else
		lownew = gamma + upnew;

	if( lownew > C - eps * C)
		lownew=C;
	else if (lownew < eps * C)
			lownew=0.0f;
	*anewup = upnew;
	*anewlow = lownew;
}
void smo(svm_sample *train, svm_model *model)
{
	int nTV = train->nTV;
	int nfeatures = model->nfeatures;
	float tau = 0.001;
	float C = 1;
	float betta = 1./nfeatures;
	float bup;//alpha_up
	float blow;//alpha_low
	float anewlow, anewup;
	float deltaup, deltalow;
	unsigned int iup, ilow;
	float k12;
	float *f = (float*)malloc(nTV*sizeof(float));// object function
	float *a = (float*)malloc(nTV*sizeof(float));// Lagrange multipliers
	int *y = train->l_TV;//lables
	float *tv = train->TV;
	init(a, y, f, nTV);
	reduction(y, a, f, C, nTV, &blow, &bup, &iup, &ilow);
	int iter = 0;
	bool done = false;
	while (blow>bup + 2*tau)
	{
		iter++;
		k12 = getK(tv, nfeatures, betta, iup, ilow);
		anewup = max(0, min(a[iup] - y[iup]*(f[ilow]-f[iup])/(2*k12-2),C));
		anewlow = max(0, min(a[ilow] - y[iup]*y[ilow]*(anewup-a[iup]), C));
		//calc_alphas(tv, y, a, f, &anewup, &anewlow, iup, ilow, nTV, nfeatures, C, betta, &done);
		deltaup = anewup-a[iup];
		deltalow = anewlow-a[ilow];
		for (int i = 0; i < nTV; i++)
		{
			f[i] += deltalow*y[ilow]*getK(tv, nfeatures, betta, i, ilow)
				    +deltaup*y[iup]*getK(tv, nfeatures, betta, iup, i);
		}
		a[iup] = anewup;
		a[ilow] = anewlow;
		if (done)
			break;
		reduction(y, a, f, C, nTV, &blow, &bup, &iup, &ilow);
	}
	model->b = (float*)malloc(sizeof(float));
	model->b[0] = -(blow+bup)/2; 
	model->C = C;
	int nsv = 0;
	for (int i = 0; i < nTV; i++)
	{
		if (a[i])
			nsv++;
	}
	model->nSV = nsv;
	model->SV_dens = (float*)malloc(nsv*nfeatures*sizeof(float));
	model->l_SV = (float*)malloc(nsv*sizeof(float));
	model->coef_gamma = betta;
	model->kernel_type = 0;
	model->svm_type = 0;
	for (int i = 0, k = 0; i < nTV; i++)
	{
		if (a[i] != 0)
		{
			for (int j = 0; j < nfeatures; j++)
			{
				model->SV_dens[k*nfeatures+j] = tv[i*nfeatures+j];
			}
			model->l_SV[k] = a[i]*y[i];
			k++;
		}
	}
	
}
int main(int argc, char **argv)
{
	FILE *input, *output;
	if (argc==1)
	{
		argc = 4;
		//argv[1] = "C:\\Data\\b.txt";
		//argv[2] = "C:\\Data\\b.model";
		//argv[3] = "10";
		argv[1] = "C:\\Data\\a9a";
		argv[2] = "C:\\Data\\a9a.model";
		argv[3] = "123";
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
	model->C = 1.;
	model->coef_gamma =1./model->nfeatures;
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
	//smo(train, model);
	train_model(train, model);
	printf("time %f cache hits %d\n", cuGetTimer(), cache_hit);
	save_model(output, model);
	//output model
	return 0;
}
