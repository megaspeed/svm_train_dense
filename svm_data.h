#ifndef _SVM_DATA_H_
#define _SVM_DATA_H_
#define MAXTHREADS 128
#define MAXBLOCKS 49152/MAXTHREADS
#define MAXBLOCKS_TV 49152/MAXTHREADS/MAXTHREADS
#define KMEM 0.65
#define TAU 0.001
#define min(a, b)  (((a) < (b)) ? (a) : (b))
#define max(a, b)  (((a) > (b)) ? (a) : (b))

struct svm_sample
{
	int nTV;				/*# of test vectors/samples */
	int *l_TV;				/*	TV's labels				*/
	float *TV;				/*	TVs in dense format		*/
};

struct svm_model
{
	int nr_class;		/*	number of classes		*/
	int nSV;			/*	# of SV					*/
	int nfeatures;		/*	# of SV's features		*/
	float *SV_dens;		/*	SVs in dense format		*/
	float *l_SV;		/*	SV's labels				*/
	float b;			/*	classification parametr	*/	
	int *label_set;		/*  intput lables			*/
	int *SVperclass;	/* number of SVs for each class*/
	int svm_type;
	int kernel_type;
	float coef_d;
	float coef_gamma;
	float coef_b;
	float C;
	float *params;		/*	params C_i, gamma_i for RBF*/
	float* mass_b;
	int	ntasks;
};

#endif

