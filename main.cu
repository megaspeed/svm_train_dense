#include "common.cpp"
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <float.h>
#include "device_launch_parameters.h"
//#include "kernels.cu"

//void Reduce_step(int *d_y, float *d_a, float *d_f, float *d_B, unsigned int *d_I, float *d_C, int ntraining, int nblocks,
//				 float *h_B, unsigned int *h_I, float* h_B_global, unsigned int *h_I_global)
//{
//	int smem = MAXTHREADS*(sizeof(float) + sizeof(int));
//	Local_Reduce_Min<<<nblocks, MAXTHREADS, smem>>>(d_y, d_a, d_f, d_B, d_I, d_C, ntraining);
//	Local_Reduce_Max<<<nblocks, MAXTHREADS, smem>>>(d_y, d_a, d_f, &d_B[nblocks], &d_I[nblocks], d_C, ntraining);
//
//	cudaMemcpy(h_B, d_B, 2*nblocks*sizeof(float), cudaMemcpyDeviceToHost);
//	cudaMemcpy(h_I, d_I, 2*nblocks*sizeof(int), cudaMemcpyDeviceToHost);
//	// Global reduction
//	float global_Bup = h_B[0];
//	float global_Blow = h_B[nblocks];
//	int global_Iup = h_I[0];
//	int global_Ilow = h_I[nblocks];
//
//	for (int i = 1; i < nblocks; i++)
//	{
//		if (h_B[i] < global_Bup)
//		{
//			global_Bup = h_B[i];
//			global_Iup = h_I[i];
//		}
//		if (h_B[nblocks + i] > global_Blow)
//		{
//			global_Blow = h_B[nblocks + i];
//			global_Ilow = h_I[nblocks + i];
//		}
//	}
//
//	h_B_global[0] = global_Bup;
//	h_B_global[1] = global_Blow;
//	h_I_global[0] = global_Iup;
//	h_I_global[1] = global_Ilow;
//}
//
//void train_model(svm_sample *train, svm_model *model)
//{
//	float reductiontime = 0;
//	float intervaltime;
//	cudaEvent_t start, stop;
//	cudaEventCreate ( &start );
//	cudaEventCreate ( &stop  );
//
//	int nTV = train->nTV;
//	int nfeatures = model->nfeatures;
//	//Grid configuration
//	int nthreads = MAXTHREADS;
//	int nblocks = min(MAXBLOCKS, (nthreads + nTV - 1)/nthreads);
//
//	float *d_TV = 0;//training vectors
//	cudaMalloc((void**) &d_TV, nTV*nfeatures*sizeof(float));
//	cudaMemcpy(d_TV, train->TV, nTV*nfeatures*sizeof(float),cudaMemcpyHostToDevice);
//
//	float *d_SV = 0;//support vectors
//
//	float *d_C;// regularisation parameter
//	cudaMalloc((void**) &d_C, sizeof(float));
//	cudaMemcpy(d_C, &model->C, sizeof(float),cudaMemcpyHostToDevice);/////////////////////////////
//
//	int *d_y = 0;// binary labels
//	cudaMalloc((void**) &d_y, nTV*sizeof(int));
//	cudaMemcpy(d_y, train->l_TV, nTV*sizeof(int),cudaMemcpyHostToDevice);
//
//	float *d_a = 0; //alphas
//	cudaMalloc((void**) &d_a, nTV*sizeof(float));
//
//	float *d_f = 0;//object functions
//	cudaMalloc((void**) &d_f, nTV*sizeof(float));
//
//	float *h_k = (float*)malloc(nTV*nTV*sizeof(float));
//	float *d_k = 0;// gramm matrix
//	cudaMalloc((void**) &d_k, nTV*nTV*sizeof(float));
//	float alfa = 1.0;
//	float betta = 0;
//
//	//locally reduced Lagrange multipliers {Bup:Blow}
//	float *h_B = (float*)malloc(2*nblocks*sizeof(float));
//	float *d_B = 0;
//	cudaMalloc((void**) &d_B, 2*nblocks*sizeof(float));
//
//	//indeces of locally reduced Lagrange multipliers {Iup:Ilow}
//	unsigned int *h_I = (unsigned int*)malloc(2*nblocks*sizeof(unsigned int));
//	unsigned int *d_I = 0; 
//	cudaMalloc((void**) &d_I, 2*nblocks*sizeof(unsigned int));
//
//	//global Lagrange multipliers {Bup:Blow}
//	float *h_B_global = (float*)malloc(2*sizeof(float));
//	unsigned int *h_I_global = (unsigned int*)malloc(2*sizeof(unsigned int));
//
//	unsigned int *d_I_cache = 0; 
//	cudaMalloc((void**) &d_I_cache, 2*sizeof(unsigned int));
//
//	float *h_delta_a = (float*)malloc(2*sizeof(float));
//	float *d_delta_a = 0;
//	cudaMalloc((void**) &d_delta_a, 2*sizeof(float));
//
//	int *h_active = (int*)malloc(sizeof(int));
//	h_active[0] = 0;
//	int *d_active = 0;
//	cudaMalloc((void**) &d_active, sizeof(int));
//	cudaMemset(d_active, 0, sizeof(int));
//
//	cublasHandle_t handle;
//	cublasCreate_v2(&handle);
//	//cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_DEVICE);
//	//cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_N, nTV, nTV, nfeatures, &alfa, d_TV, nfeatures, d_TV, nfeatures, &betta, d_k, nTV);
//	//	cudaDeviceReset();
//	//return;
//	for (int i = 0; i < nTV; i++)
//	{		
//		for (int j = 0; j < nTV; j++)
//		{
//			float sum = 0;
//			for (int k = 0; k < nfeatures; k++)
//			{
//				sum += train->TV[i*nfeatures+k]*train->TV[j+k*nfeatures];
//			}
//			h_k[i*nTV+j] = sum;
//		}
//	}
//	cudaMemcpy(d_k, h_k, nTV*nTV*sizeof(float),cudaMemcpyHostToDevice);
//	initialization<<<nblocks, nthreads>>>(d_a, d_f, d_y, nTV);
//
//	int iter = 0;
//	while (true)
//	{
//		++iter;
//		
//		Reduce_step(d_y, d_a, d_f, d_B, d_I, d_C, nTV, nblocks, h_B, h_I, h_B_global, h_I_global);
//		cudaMemcpy(d_I_cache, h_I_global, 2*sizeof(int),cudaMemcpyHostToDevice);
//		//cache
//		Update<<<1,1>>>(d_k, d_y, d_f, d_a, d_delta_a, d_I_cache, d_I_cache, d_C, d_active, nTV);
//		cudaMemcpy(h_active, d_active, sizeof(int), cudaMemcpyDeviceToHost);
//		if (h_active[0])
//			break;
//		Map<<<nblocks, nthreads>>>(d_f, d_k, d_y, d_delta_a, d_I, d_I_cache, nTV); 
//	}
//	float *l_sv = (float*)malloc(nTV*sizeof(float));
//	float *SV = (float*)malloc(nTV*nfeatures*sizeof(float));
//	cudaMemcpy(l_sv, d_a, sizeof(float), cudaMemcpyDeviceToHost);
//	float threshold = -(l_sv[h_I_global[0]]-l_sv[h_I_global[1]])/2;
//	model->b = &threshold;
//
//	int nSV = 0;
//	int offset = 0;
//	for (int i = 0; i < nTV; i++)
//	{
//		if (l_sv[i] != 0)
//		{
//			if (i != nSV)
//			{
//				l_sv[nSV] = l_sv[i];
//				for (int j = 0; j < nfeatures; j++)
//					SV[nSV*nfeatures+j] = train->TV[i*nfeatures+j];
//			}			
//			++nSV;
//		}
//	}
//	model->l_SV=(float*)(l_sv, nSV*sizeof(float));
//	model->SV_dens=(float*)realloc(SV, nSV*nfeatures*sizeof(float));
//
//
//	cublasDestroy_v2(handle);
//}
void init(float *a, int *y, float *f, int ntv)
{
	for (int i = 0; i < ntv; i++)
	{
		a[i] = 0.;
		f[i] = -1.*y[i]; 
	}
}
void reduction(int *y, float *a, float *f, float C, int nTV, float *blow, float *bup, int *iup, int *ilow)
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

float scal(float *x, float *y, int n)
{
	float val = 0;
	for (int i = 0; i < n; i++)
	{
		val += x[i]*y[i];
	}
	return val;
}
float getK(float *tv, int ncol, int *y, float gamma, int i, int j)
{
	float val = scal(&tv[i*ncol], &tv[i*ncol], ncol)+
				scal(&tv[j*ncol], &tv[j*ncol], ncol)-
				2*scal(&tv[i*ncol], &tv[j*ncol], ncol);
	return y[i]*y[j]*exp(-gamma*val);
}
void smo(svm_sample *train, svm_model *model)
{
	int nTV = train->nTV;
	int nfeatures = model->nfeatures;
	float tau = 0.001;
	float C = 1;
	float gamma = 1./nfeatures;
	float bup;//alpha_up
	float blow;//alpha_low
	float deltaup, deltalow;
	int iup, ilow;
	float k1, k2, k12;
	float *f = (float*)malloc(nTV*sizeof(float));// object function
	float *a = (float*)malloc(nTV*sizeof(float));// Lagrange multipliers
	int *y = train->l_TV;//lables
	float *tv = train->TV;
	init(a, y, f, nTV);
	reduction(y, a, f, C, nTV, &blow, &bup, &iup, &ilow);
	int iter = 0;
	while (blow>bup + 2*tau)
	{
		iter++;
		k12 = getK(tv, nfeatures, y, gamma, iup, ilow);
		a[iup] = max(0, min(bup - y[iup]*(f[ilow]-f[iup])/(2*k12-2),C));
		a[ilow] = max(0, min(blow + y[iup]*y[ilow]*(bup-a[iup]), C));
		deltaup = a[iup]-bup;
		deltalow = a[ilow]-blow;
		for (int i = 0; i < nTV; i++)
		{
			f[i] += deltalow*y[ilow]*getK(tv, nfeatures, y, gamma, i, ilow)
				    +deltaup*y[iup]*getK(tv, nfeatures, y, gamma, iup, i);
		}
		reduction(y, a, f, C, nTV, &blow, &bup, &iup, &ilow);
	}
	model->b = (float*)malloc(sizeof(float));
	model->b[0] = (blow+bup)/2; 
	model->C = C;
	int nsv = 0;
	for (int i = 0; i < nTV; i++)
	{
		if (a[i])
		{
			nsv++;
		}
	}
	model->nSV = nsv;
	model->SV_dens = (float*)malloc(nsv*nfeatures*sizeof(float));
	model->l_SV = (float*)malloc(nsv*sizeof(float));
	model->coef_gamma = gamma;
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
	}
	if(argc<4)
		exit_with_help();
	struct svm_model *model = (svm_model*)malloc(sizeof(svm_model));
	struct svm_sample *train = (svm_sample*)malloc(sizeof(svm_sample));
	sscanf(argv[3],"%d",&model->nfeatures);
	model->C = 1.;
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
	
	smo(train, model);
	//train_model(train, model);
	save_model(output, model);
	//output model
	return 0;
}
