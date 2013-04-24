#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "svm_data.h"

#ifdef _WIN32

#include <windows.h>

static LARGE_INTEGER t;
static float         f;
static int           freq_init = 0;

void cuResetTimer(void) {
  if (!freq_init) {
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    f = (float) freq.QuadPart;
    freq_init = 1;
  }
  QueryPerformanceCounter(&t);
}

float cuGetTimer(void) {
  LARGE_INTEGER s;
  float d;
  QueryPerformanceCounter(&s);

  d = ((float)(s.QuadPart - t.QuadPart)) / f;

  return (d*1000.0f);
}

#else

#include <sys/time.h>

static struct timeval t;

/**
 * Resets timer
 */
void cuResetTimer() {
  gettimeofday(&t, NULL);
}


/**
 * Gets time since reset
 */
float cuGetTimer() { // result in miliSec
  static struct timeval s;
  gettimeofday(&s, NULL);

  return (s.tv_sec - t.tv_sec) * 1000.0f + (s.tv_usec - t.tv_usec) / 1000.0f;
}

#endif

const char *svm_type_table[] = { "c_svc","nu_svc","one_class","epsilon_svr","nu_svr",0 };
const char *kernel_type_table[] = { "rbf","linear","polynomial","sigmoid","precomputed",0 };
void set_labels(svm_sample *train, svm_model *model)
{
	for (int i = 0; i < train->nTV; i++)
	{
		if (train->l_TV[i] == model->label_set[0])
			train->l_TV[i] = 1;
		else
			train->l_TV[i] = -1;
	}
}

static char* readline(FILE *input, char* line, int *max_line_len)
{
	int len;
	if(fgets(line,*max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		*max_line_len *= 2;
		line = (char *) realloc(line,*max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,*max_line_len-len,input) == NULL)
			break;
	}
	return line;
}
void free_model(struct svm_model *model)
{
	free(model->SV_dens);
	free(model->l_SV);
	free(model->b);
	free(model);	
}
void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}
/**
* Parses data from file in the libsvm format
* @param inputfilename pointer to the char array that contains the file name
* @param h_xdata host pointer to the array that will store the training set
* @param h_ldata host pointer to the array that will store the labels of the training set
* @param nsamples number of samples in the training set
* @param nfeatures number of features per sample in the training set
* @param nclasses number of classes
*/
int parse_SV(FILE* inputFilePointer, float** h_xdata, float** h_ldata, int nsamples, int nfeatures)
{
	char* stringBuffer = (char*)malloc(65536);
	static char* line;

	*h_xdata = (float*) calloc( nsamples*nfeatures,sizeof(float));
	*h_ldata = (float*) calloc( nsamples,sizeof(float));

	for(int i = 0; i < nsamples; i++)
	{
		char c;
		int pos=0;
		char* bufferPointer = stringBuffer;

		do
		{
			c = fgetc(inputFilePointer);

			if((c== ' ') || (c == '\n'))
			{
				if(pos==0)
				{
					//Label found
					*(bufferPointer) = 0;
					float value;
					sscanf(stringBuffer, "%f", &value);
					(*h_ldata)[i]=value;
					pos++;
				}
				else
				{
					//Feature found
					*(bufferPointer) = 0;
					float value;
					sscanf(stringBuffer, "%f", &value);
					(*h_xdata)[i*nfeatures + (pos-1)]= value;
				}
				bufferPointer = stringBuffer;
			}
			else if(c== ':')
			{
				//Position found
				*(bufferPointer) = 0;
				int value;
				sscanf(stringBuffer, "%i", &value);
				pos= value;
				bufferPointer = stringBuffer;
			}
			else
			{
				*(bufferPointer) = c;
				bufferPointer++;
			}

		}
		while (c != '\n');
	}
	free(stringBuffer);
	free(line);
	fclose(inputFilePointer);
	return 1;
}
int parse_TV(FILE* inputFilePointer, svm_sample *train, svm_model *model)
{
	int nfeatures = model->nfeatures;
	char* stringBuffer = (char*)malloc(65536);
	static int max_line_len = 1024;
	static char* line;
	int nsamples = 0;
	line = (char*)malloc(max_line_len * sizeof(char));
	while(readline(inputFilePointer, line, &max_line_len)!=NULL)
	{
		char *p = strtok(line," \t"); // label
		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
		}
		++nsamples;
	}
	rewind(inputFilePointer);
	//*nsamples = NTV;
	float *h_xdata = (float*) calloc( nsamples*nfeatures,sizeof(float));
	int *h_ldata = (int*) calloc( nsamples,sizeof(int));
	int *set_labels = (int*)malloc(2*sizeof(int));

	int label_number = 0;
	set_labels[label_number] = 0;

	for(int i = 0; i < nsamples; i++)
	{
		char c;
		int pos=0;
		char* bufferPointer = stringBuffer;

		do
		{
			c = fgetc(inputFilePointer);

			if((c== ' ') || (c == '\n'))
			{
				if(pos==0)
				{
					//Label found
					*(bufferPointer) = 0;
					int value;
					sscanf(stringBuffer, "%d", &value);
					bool newlabel = true;
					for (int k = 0; newlabel && label_number!=2; k++)
					{					
						if(set_labels[k] == value)
						{
							newlabel = false;
						}
						else
						{
							set_labels[label_number] = value;
							++label_number;
							newlabel = false;
						}
					}
					h_ldata[i]=value;
					++pos;
				}
				else
				{
					//Feature found
					*(bufferPointer) = 0;
					float value;
					sscanf(stringBuffer, "%f", &value);
					h_xdata[i*nfeatures + (pos-1)]= value;
				}
				bufferPointer = stringBuffer;
			}
			else if(c== ':')
			{
				//Position found
				*(bufferPointer) = 0;
				int value;
				sscanf(stringBuffer, "%i", &value);
				pos= value;
				bufferPointer = stringBuffer;
			}
			else
			{
				*(bufferPointer) = c;
				bufferPointer++;
			}

		}
		while (c != '\n');
	}
	train->TV = h_xdata;
	train->l_TV = h_ldata;
	train->nTV = nsamples;
	model->label_set = set_labels;
	model->nr_class = label_number;

	free(stringBuffer);
	free(line);
	fclose(inputFilePointer);
	return 1;
}
int read_model(const char* model_file_name, svm_model *model, int nfeatures)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return 0;
	const char *svm_type_table[] = { "c_svc","nu_svc","one_class","epsilon_svr","nu_svr",0 };
	const char *kernel_type_table[] = { "rbf","linear","polynomial","sigmoid","precomputed",0 };
	// read parameters
	model->nSV = NULL;
	model->SV_dens = NULL;
	model->l_SV = NULL;
	model->label_set = NULL;
	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);

		if(strcmp(cmd,"svm_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0; svm_type_table[i]!=0;i++)
			{
				if(strcmp(svm_type_table[i],cmd)==0)
				{
					model->svm_type=i;
					break;
				}
			}
			if(svm_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown svm type.\n");
				free_model(model);
				return 0;
			}
		}
		else if(strcmp(cmd,"kernel_type")==0)
		{		
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;kernel_type_table[i];i++)
			{
				if(strcmp(kernel_type_table[i],cmd)==0)
				{
					model->kernel_type=i;
					break;
				}
			}
			if(kernel_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown kernel function.\n");
				free_model(model);
				return 0;
			}
		}
		else if(strcmp(cmd,"degree")==0)
			fscanf(fp,"%f",&model->coef_d);
		else if(strcmp(cmd,"gamma")==0)
			fscanf(fp,"%f",&model->coef_gamma);
		else if(strcmp(cmd,"coef0")==0)
			fscanf(fp,"%f",&model->coef_b);
		else if(strcmp(cmd,"nr_class")==0)
			fscanf(fp,"%d",&model->nr_class);
		else if(strcmp(cmd,"total_sv")==0)
			fscanf(fp,"%d",&model->nSV);
		else if(strcmp(cmd,"rho")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->b = (float*)malloc(n*sizeof(float));
			for(int i=0;i<n;i++)
				fscanf(fp,"%f",&model->b[i]);
		}
		else if(strcmp(cmd,"label")==0)
		{
			int n = model->nr_class;
			model->label_set = (int*)malloc(n*sizeof(int));
			for(int i=0;i<n;i++)
				fscanf(fp,"%d",&model->label_set[i]);
		}
		else if(strcmp(cmd,"nr_sv")==0)
		{
			int n = model->nr_class;
			int temp;
			for(int i=0;i<n;i++)
				fscanf(fp,"%d",&temp);
		}
		else if(strcmp(cmd,"SV")==0)
		{
			while(1)
			{
				int c = getc(fp);
				if(c==EOF || c=='\n') break;	
			}
			break;
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			free_model(model);
			return 0;
		}
	}
	// read sv_coef and SV
	parse_SV(fp,&model->SV_dens,&model->l_SV,model->nSV,nfeatures);
	return 1;
}
void exit_with_help()
{
	printf("Usage: svm-predict test_file model_file #_of_features\n");
	exit(1);
}
int save_model(FILE *fp, const svm_model *model)
{

	fprintf(fp,"svm_type %s\n", svm_type_table[model->svm_type]);
	fprintf(fp,"kernel_type %s\n", kernel_type_table[model->kernel_type]);

	if(model->kernel_type == 2)
		fprintf(fp,"degree %d\n", model->coef_d);

	if(model->kernel_type == 2 || model->kernel_type == 0 || model->kernel_type == 3)
		fprintf(fp,"gamma %g\n", model->coef_gamma);

	if(model->kernel_type == 2 || model->kernel_type == 3)
		fprintf(fp,"coef0 %g\n", model->coef_b);

	int nr_class = model->nr_class;
	int l = model->nSV;
	fprintf(fp, "nr_class %d\n", nr_class);
	fprintf(fp, "total_sv %d\n",l);
	
	{
		fprintf(fp, "rho");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %f",model->b[i]);
		fprintf(fp, "\n");
	}
	
	if(model->label_set)
	{
		fprintf(fp, "label");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->label_set[i]);
		fprintf(fp, "\n");
	}


	fprintf(fp, "SV\n");
	float *sv_coef = model->l_SV;
	float *SV = model->SV_dens;

	for(int i=0;i<l;i++)
	{
		for(int j=0;j<nr_class-1;j++)
			fprintf(fp, "%.16g ",sv_coef[j*nr_class+i]);

		for (int j = 0; j < model->nfeatures; j++)
			fprintf(fp,"%d:%.8g ", j+1, SV[i*model->nfeatures+j]);

		fprintf(fp, "\n");
	}

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}