#include <cuda_runtime_api.h>
#include <float.h>
#include "svm_data.h"
#include "device_launch_parameters.h"
/**
 * Set initial values of the obj functions and alphas
 * @param d_a device pointer to the array with the alphas
 * @param d_f device pointer to the intermediate values of f 
 * @param d_y device pointer to the array with binary labels
 * @param ntraining number of training samples in the training set
 */
__global__ void initialization( float *d_a, float *d_f, int *d_y, int ntraining)
{
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

	while ( i < ntraining)
	{
		d_a[i] = 0.;
		d_f[i] = -1.*d_y[i];
		i += blockDim.x*gridDim.x;
	}
	__syncthreads();
}

__global__ void Local_Reduce_Min(int* d_y, float* d_a, float *d_f, float *d_bup_local,
								 unsigned int* d_Iup_local, float d_C, int ntraining)
{
	extern __shared__ float reducearray[];
	unsigned int tid = threadIdx.x;
	unsigned int blocksize = blockDim.x;
	unsigned int gridsize = blocksize*gridDim.x;
	unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
	float *minreduction = (float*)reducearray;
	unsigned int *minreductionid = (unsigned int*)&reducearray[blocksize];
	minreduction[tid] = (float)FLT_MAX;
	minreductionid[tid] = i;
	float alpha_i;
	int y_i;
	while ( i < ntraining)
	{
		alpha_i = d_a[i];
		y_i = d_y[i];
		if ((   (y_i==1  && alpha_i>0 && alpha_i<d_C) ||
				(y_i==-1 && alpha_i>0 && alpha_i<d_C)) ||
				(y_i==1  && alpha_i==0)|| (y_i==-1 && alpha_i==d_C))
		{
			if (minreduction[tid] > d_f[i])
			{
				minreduction[tid] = d_f[i];
				minreductionid[tid] = i;
			}
		}
		i += gridsize;
	}
	__syncthreads();
	if (blocksize >= 512){if(tid < 256){if(minreduction[tid] >  minreduction[tid+256])
										{  minreduction[tid] =   minreduction[tid+256];
										 minreductionid[tid] = minreductionid[tid+256];}}
						__syncthreads();}
	if (blocksize >= 256){if(tid < 128){if(minreduction[tid] >  minreduction[tid+128])
										{  minreduction[tid] =   minreduction[tid+128];
										 minreductionid[tid] = minreductionid[tid+128];}}
						__syncthreads();}
	if (blocksize >= 128){if(tid < 64){if(minreduction[tid] >  minreduction[tid+64])
										{ minreduction[tid] =   minreduction[tid+64];
										minreductionid[tid] = minreductionid[tid+64];}}
						__syncthreads();}

	if (tid < 32){	if(blocksize >= 64){if(minreduction[tid] >  minreduction[tid+32])
										{  minreduction[tid] =   minreduction[tid+32];
										 minreductionid[tid] = minreductionid[tid+32];}}
					if(blocksize >= 32){if(minreduction[tid] >  minreduction[tid+16])
										{  minreduction[tid] =   minreduction[tid+16];
										 minreductionid[tid] = minreductionid[tid+16];}}
					if(blocksize >= 16){if(minreduction[tid] >  minreduction[tid+ 8])
										{  minreduction[tid] =   minreduction[tid+ 8];
										 minreductionid[tid] = minreductionid[tid+ 8];}}
					if(blocksize >= 8){if( minreduction[tid] >  minreduction[tid+ 4])
										{  minreduction[tid] =   minreduction[tid+ 4];
										 minreductionid[tid] = minreductionid[tid+ 4];}}
					if(blocksize >= 4){if( minreduction[tid] >  minreduction[tid+ 2])
										{  minreduction[tid] =   minreduction[tid+ 2];
										 minreductionid[tid] = minreductionid[tid+ 2];}}
					if(blocksize >= 2){if( minreduction[tid] >  minreduction[tid+ 1])
										{  minreduction[tid] =   minreduction[tid+ 1];
										 minreductionid[tid] = minreductionid[tid+ 1];}}}

	if (tid == 0)
	{
		d_bup_local[blockIdx.x] = minreduction[tid];
		d_Iup_local[blockIdx.x] = minreductionid[tid];
	}
}

__global__ void Local_Reduce_Max(int* d_y, float* d_a, float *d_f, float *d_blow_local,
								 unsigned int* d_Ilow_local, float d_C, int ntraining)
{
	extern __shared__ float reducearray[];
	unsigned int tid = threadIdx.x;
	unsigned int blocksize = blockDim.x;
	unsigned int gridsize = blocksize*gridDim.x;
	unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
	float *maxreduction = (float*)reducearray;
	int *maxreductionid = (int*)&reducearray[blocksize];
	maxreduction[tid] = -1.*(float)FLT_MAX;	
	maxreductionid[tid] = i;
	float alpha_i;
	int y_i;
	while ( i < ntraining)
	{
		alpha_i = d_a[i];
		y_i = d_y[i];
		if ((   (y_i==1  && alpha_i>0 && alpha_i<d_C) ||
				(y_i==-1 && alpha_i>0 && alpha_i<d_C)) ||
				(y_i==1  && alpha_i==d_C)|| (y_i==-1 && alpha_i==0))
		{
			if (maxreduction[tid] < d_f[i])
			{
				maxreduction[tid] = d_f[i];
				maxreductionid[tid] = i;
			}
		}
		i += gridsize;
	}
	__syncthreads();
	if (blocksize >= 512){if(tid < 256){if(maxreduction[tid] <  maxreduction[tid+256])
										{  maxreduction[tid] =   maxreduction[tid+256];
										 maxreductionid[tid] = maxreductionid[tid+256];}}
						__syncthreads();}
	if (blocksize >= 256){if(tid < 128){if(maxreduction[tid] <  maxreduction[tid+128])
										{  maxreduction[tid] =   maxreduction[tid+128];
										 maxreductionid[tid] = maxreductionid[tid+128];}}
						__syncthreads();}
	if (blocksize >= 128){if(tid < 64){if(maxreduction[tid] <  maxreduction[tid+64])
										{ maxreduction[tid] =  maxreduction[tid+64];
										maxreductionid[tid] = maxreductionid[tid+64];}}
						__syncthreads();}

	if (tid < 32){	if(blocksize >= 64){if(maxreduction[tid] <  maxreduction[tid+32])
										{  maxreduction[tid] =   maxreduction[tid+32];
										 maxreductionid[tid] = maxreductionid[tid+32];}}
					if(blocksize >= 32){if(maxreduction[tid] <  maxreduction[tid+16])
										{  maxreduction[tid] =   maxreduction[tid+16];
										 maxreductionid[tid] = maxreductionid[tid+16];}}
					if(blocksize >= 16){if(maxreduction[tid] <  maxreduction[tid+ 8])
										{  maxreduction[tid] =   maxreduction[tid+ 8];
										 maxreductionid[tid] = maxreductionid[tid+ 8];}}
					if(blocksize >= 8){if( maxreduction[tid] <  maxreduction[tid+ 4])
										{  maxreduction[tid] =   maxreduction[tid+ 4];
										 maxreductionid[tid] = maxreductionid[tid+ 4];}}
					if(blocksize >= 4){if( maxreduction[tid] <  maxreduction[tid+ 2])
										{  maxreduction[tid] =   maxreduction[tid+ 2];
										 maxreductionid[tid] = maxreductionid[tid+ 2];}}
					if(blocksize >= 2){if( maxreduction[tid] <  maxreduction[tid+ 1])
										{  maxreduction[tid] =   maxreduction[tid+ 1];
										 maxreductionid[tid] = maxreductionid[tid+ 1];}}}

	if (tid == 0)
	{
		d_blow_local[blockIdx.x] = maxreduction[tid];
		d_Ilow_local[blockIdx.x] = maxreductionid[tid];
	}
}

__global__ void Map( float *d_f, float *d_k, int *d_y, float *d_delta_a, unsigned int* d_I_global,
					unsigned int *d_I_cache, int ntraining, int width)
{
	unsigned int gridsize = blockDim.x*gridDim.x;
	unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;

	while ( i < ntraining)
	{
		d_f[i] += d_delta_a[0]*d_y[d_I_global[0]]*d_k[d_I_cache[0]*width+i] +  /*up */
				  d_delta_a[1]*d_y[d_I_global[1]]*d_k[d_I_cache[1]*width+i];   /*low*/
		i += gridsize;
	}
}

__global__ void Update(float *d_k, int *d_y, float *d_f, float *d_a, float *d_delta_a, 
					   unsigned int *d_I_global, unsigned int *d_I_cache, float d_C, int *d_active, int ntraining, int width)
{
	int g_Iup = d_I_global[0];
	int g_Ilow = d_I_global[1];
	float alpha_up_old =d_a[g_Iup];
	float alpha_low_old =d_a[g_Ilow];
	float alpha_up_new = max(0, min(alpha_up_old + 
		(d_y[g_Iup]*(d_f[g_Ilow]-d_f[g_Iup])/
		(2- 2*d_k[d_I_cache[1]*width+g_Iup])), d_C));

	float alpha_low_new = max(0, min(alpha_low_old+
		d_y[g_Iup]*d_y[g_Ilow]*(alpha_up_old-alpha_up_new), d_C));
	d_delta_a[0] = alpha_up_new-alpha_up_old;
	d_delta_a[1] = alpha_low_new-alpha_low_old;
	d_a[g_Iup] = alpha_up_new;
	d_a[g_Ilow] = alpha_low_new;
	if ((alpha_low_new-alpha_up_new) < 2*TAU)
	{
		d_active[0] = 1;
	}
}
__global__ void get_dot(float *x, float *dot, int n, int width)
{
	unsigned int gridsize = blockDim.x*gridDim.x;
	unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
	extern __shared__ float val[];
	while ( i < n)
	{
	    val[threadIdx.x] = 0;
		for (int j = 0; j < width; j++)
		{
			val[threadIdx.x] +=	x[i*width+j]*x[i*width+j];
		}
		dot[i] = val[threadIdx.x];
		i += gridsize;
	}
}

__global__ void get_row(float *d_k, float *tv, float gamma, int nfeatures, unsigned int irow, unsigned int icache, int ntraining, int width)
{
	unsigned int gridsize = blockDim.x*gridDim.x;
	unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;

	while ( i < ntraining)
	{
		float val = 0;
		for (int j = 0; j < nfeatures; j++)
		{
			val +=	tv[irow*nfeatures+j]*tv[irow*nfeatures+j]+
					tv[i*nfeatures+j]*tv[i*nfeatures+j]-
					2*tv[i*nfeatures+j]*tv[irow*nfeatures+j];
		}
		d_k[icache*width+i] = exp(-gamma*val);
		i += gridsize;
	}
}
