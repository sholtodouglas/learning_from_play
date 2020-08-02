#include "CyberGlove.h"
#include "CyberGlove_utils.h"
#include <Windows.h>
#include <stdio.h>
#include <math.h>

static CyberGlove* persistentGlove = NULL;
cgData cgdata;
cgOption option;

// Utilities ======================

// write error message to console, pause and exit
void util_error(const char* msg)
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
		HANDLE hStdoubt = GetStdHandle(STD_OUTPUT_HANDLE);
		SetConsoleTextAttribute(hStdoubt, FOREGROUND_INTENSITY | FOREGROUND_RED);
#endif
		printf("ERROR: %s\n\nPress Enter to exit ...", msg);
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
		SetConsoleTextAttribute(hStdoubt, FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED);
#endif
		// pause, exit
		getchar();
		exit(1);
}

// write warning message to console
void util_warning(const char* msg)
{
		// write to console
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
		HANDLE hStdoubt = GetStdHandle(STD_OUTPUT_HANDLE);
		SetConsoleTextAttribute(hStdoubt, FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_RED);
#endif
		printf("\nWARNING: %s\n\n", msg);
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
		SetConsoleTextAttribute(hStdoubt, FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED);
#endif
}

// allocate memory, align
void* util_malloc(unsigned int sz, unsigned int align)
{
	void* ptr = 0;

	// try to allocate
#if defined(__linux__) || defined(__APPLE__)
   // could use a check for if "sz" is a multiple of "align"
	if (align == 8 || sz % align) {
		ptr = malloc(sz); // malloc is always 8 aligned
	}
	else {
		posix_memalign(&ptr, align, sz);
	}
#else
	ptr = _aligned_malloc(sz, align);
#endif

	// error if null pointer
	if( !ptr ) {
		util_error("could not allocate memory");
   }

	return ptr;
}

// free memory
void util_free(void* ptr)
{
#if defined(__linux__) || defined(__APPLE__)
	free(ptr);
#else
	_aligned_free(ptr);
#endif
}

// Open file with specified path (else locally)
FILE* util_fopen(const char* fileName, const char* mode)
{
	FILE* fp;
	char errmsg[300];
	int i=0;
	fopen_s(&fp, fileName, mode);
	if(fp)
		return fp;
	else //strip out path
	{	
		for(i=(int)strlen(fileName)-1;i>=0;i--)
			if(fileName[i]=='\\')
			{	fileName=fileName+i+1;
				break;
			}
		fopen_s(&fp, fileName, mode);
	}
	if(fp)
		return fp;
	else
	{
		sprintf_s(errmsg, "Problem opening file '%s'", fileName );
		util_error(errmsg);
		return NULL;
	}
}

// Read cgdata from a tab(or space) seperated file
int util_readFile(const char* Fname, cgNum* vec, const int size)
{
	int n_read, i;
	FILE* fp;
	char errmsg[300];
	const int str_n = 5;
	char str[str_n];
	char fmt[100];

	 // open the file, check for errors
	fp = util_fopen(Fname, "r" );
	
   // Read lines for values
	for(i = 0; i<size; i++)
	{	n_read = fscanf_s(fp, "%lf,", &vec[i]);
		if (n_read!=1)
		{	sprintf_s(fmt, "%%%ds", str_n);
			n_read = fscanf_s(fp, fmt, str);
			for(int j = 0; str[j]; j++)
				str[j] = tolower(str[j]);
			if( strcmp(str, "nan")==0) // if is a NaN
				vec[i] = sqrt(-1.0);
			else
			{	sprintf_s(errmsg, "Problem reading file '%s'. Check cgdata format", Fname );
				util_error(errmsg);
			}
		}
	}
	fclose (fp);
	return 0;
}

// vector dot-product... FMA ???
cgNum util_dot(const cgNum* vec1, const cgNum* vec2, const int n)
{
	int i = 0;
	cgNum res = 0;

	for( i=0; i<n; i++ )
		res += vec1[i] * vec2[i];

	return res;	
}

// multiply matrix and vector
void util_mulMatVec(cgNum* res, const cgNum* mat, const cgNum* vec,
				   int nr, int nc)
{
	int r;
	for( r=0; r<nr; r++ )
	   res[r] = util_dot(mat + r*nc, vec, nc);
}

// Normalize the raw sample
void cGlove_nrmRawSample(cgNum* rawSample_nrm, cgNum* rawSample,
							 cgNum* range, int rawSample_sz)
{
	int i;

	// Check range
	for(i=0; i<rawSample_sz; i++)
	{
		if(option.updateRawRange)
		{	// Update range
			if(rawSample[i]<range[i])
				range[i] = rawSample[i];
			else if (rawSample[i]>range[i+rawSample_sz])
				range[i+rawSample_sz] = rawSample[i];
		}
		else
		{	// clamp to range
			if(rawSample[i]<range[i])
				rawSample[i]=range[i];
			else if (rawSample[i]>range[i+rawSample_sz])
				rawSample[i]=range[i+rawSample_sz];
		}
		
		// Normalize rawSamples
		rawSample_nrm[i] = (rawSample[i]-range[i])/(range[i+rawSample_sz]-range[i]);
	}
	rawSample_nrm[rawSample_sz] = 1;
}

// Calibrate Glove using normalized raw sample
void cGlove_calibrateNrmSample(cgNum* calibSample, cgNum* rawSample_nrm, cgNum* AdroitRangeMat,
							 cgNum* calib, int calibSample_sz, int rawSample_sz)
{
	util_mulMatVec(calibSample, calib, rawSample_nrm, calibSample_sz, rawSample_sz+1);
	for (int i=0; i<calibSample_sz; i++)
	{	
		// clamp [0 1]
		if(calibSample[i]<0)
			calibSample[i]=0;
		else if(calibSample[i]>1)
			calibSample[i]=1;

		// Scale back Adroit joint ranges
		calibSample[i] = AdroitRangeMat[i] + (AdroitRangeMat[i+calibSample_sz]-AdroitRangeMat[i])*calibSample[i];
	}

}


// Read Configuration variables ===============

// check for white spaces
int iswhite(char a){return a==' '||a=='\t';}

// Read values
int util_config(const char *fileName, const char *iname, void *var)
{
	const int maxInputSz = 1024;
	int lineCnt=0,icnt=0,q1;
	int numw, err=-1, len;
	char line[maxInputSz],lineRaw[maxInputSz];
	char name[maxInputSz];
	char* wordPtr[16];
	char* input[2];
	FILE* file;
	char errmsg[300];

	// parse input for variable name
	strcpy_s(name, maxInputSz, iname);
	len = (int)strlen(name);
	for(q1=0; q1<len; q1++)
	{
		if((q1==0&&!iswhite(name[0]))||(name[q1-1]==0&&(!iswhite(name[q1]))))
			if(icnt<2)
				input[icnt++]=name+q1;
		if(iswhite(name[q1]))
			name[q1]=0;
	}
  
	// parse file for values
	file = util_fopen(fileName, "r");
	
	while(fgets(line,1024,file))
    {    
		lineCnt++;
		// remove comments
		if( (line[0]=='/') && (line[0]=='/') )
			continue;

		// remove inline comments
		for(q1=(int)strlen(line)-1;q1>=0;q1--)
		{
			if(line[q1]==';')
			{	line[q1]=0;
				break;
			}
			line[q1]=0;
		}
      
		strcpy_s(lineRaw, maxInputSz, line);

		if(q1==-1)
		{	if(strcmp(line,"")!=0) // if not comment or empty line
			{	sprintf_s(errmsg, "No semicolon on line %d(%s) ... skipping\n",lineCnt,lineRaw);
				util_warning(errmsg);
			}	
			continue;
		}			
		for(q1--;q1>=0;q1--)
			if(iswhite(line[q1]))
				line[q1]=0;
			else 
				break;
		if(q1==-1)
		{	sprintf_s(errmsg, "No cgdata on line %d(%s) ... skipping\n",lineCnt,lineRaw);
			util_warning(errmsg);
			continue;
		}
    
		numw=0;
    
		if(line[q1]=='"')
		{
			line[q1]=0;
			for(q1--;q1>=0;q1--)if(line[q1]=='"')break;
			if(q1==-1)
			{	sprintf_s(errmsg, "Expected string on line %d(%s), no matching \" ... skipping\n",lineCnt,lineRaw);
				util_warning(errmsg);
				continue;
			}
			line[q1]=0;
		}
		else if(line[q1]=='\'')
		{
			int cq1=q1;
			line[q1]=0;
			for(q1--;q1>=0;q1--)if(line[q1]=='\'')break;
			if(q1==-1)
			{	sprintf_s(errmsg, "Expected char on line %d(%s), no matching \' ... skipping\n",lineCnt,lineRaw);
				util_warning(errmsg);
				continue;
			}
			line[q1]=0;
			if(cq1-q1!=2)
			{	sprintf_s(errmsg, "Char needs to have length=1 %d(%s), no matching \' ... skipping\n",lineCnt,lineRaw);
				util_warning(errmsg);
				continue;
			}
		}
		else 
		{	for(;q1>=0;q1--)
				if(iswhite(line[q1]))
				{	line[q1]=0;
					break;
				}
		}
    
		wordPtr[numw++]=line+q1+1;
    
		for(q1--;q1>=0;q1--)
		{	if(q1==0||(iswhite(line[q1-1])&&(!iswhite(line[q1]))))
				wordPtr[numw++]=line+q1;
			if(iswhite(line[q1]))
				line[q1]=0;
		}
    
		//printf("------------------------------------\n");
		//for(q1=0;q1<numw;q1++)printf("%d: (%s)\n",q1,wordPtr[q1]);
		//for(q1=0;q1<icnt;q1++)printf("input:%d: (%s)\n",q1,input[q1]);

    
		if(numw!=4)
		{	sprintf_s(errmsg, "Not enough words (4 expected, %d got) on line %d (%s) ... skipping\n",numw,lineCnt,lineRaw);
			util_warning(errmsg);
			continue;
		}

		if(strcmp(wordPtr[1],"="))
		{	sprintf_s(errmsg, "Not equal sign as word 3 on line %d(%s) ... skipping\n",lineCnt,lineRaw);
			util_warning(errmsg);
			continue;
		}    
    
		/*if(strcmp(wordPtr[2],input[1]) || strcmp(wordPtr[3],input[0]))
		{
			sprintf_s(errmsg, "INFO: type/names ([%s] [%s]) and([%s] [%s]) don't match on line %d(%s) ... skipping\n",wordPtr[3],wordPtr[2],input[0],input[1],lineCnt,lineRaw);
			util_warning(errmsg);
			continue;
		} */   
    
		// Parse for variable values
		if(!strcmp(wordPtr[2], input[1])) // match variable name
		{
			// Check variable type			
			if(!strcmp(wordPtr[3],"int"))
			{	sscanf_s(wordPtr[0],"%d",(int*)var);
				err=0;
				break;
			}
			else if(!strcmp(wordPtr[3],"double"))
			{	sscanf_s(wordPtr[0],"%lf",(double*)var);
				err=0;
				break;
			}
			else if(!strcmp(wordPtr[3],"char"))
			{	*((char*)var)=wordPtr[0][0];
				err=0;
				break;
			}
			else if(!strcmp(wordPtr[3],"char*"))
			{	char*t=(char*)malloc((int)strlen(wordPtr[0])+1);
				strcpy_s(t, (int)strlen(wordPtr[0])+1, wordPtr[0]);
				*((char**)var)=t;
				err=0;
				break;
			}
			else if(!strcmp(wordPtr[3],"bool"))
			{
				if(!strcmp("true",wordPtr[0]))
				{	*((bool*)var)=true;
					err=0;
					break;
				}
				else if(!strcmp("false",wordPtr[0]))
				{	*((bool*)var)=false;
					err=0;
					break;
				}
				else 
				{	sprintf_s(errmsg, "Wrong bool literal on line %d(%s) ... skipping\n",lineCnt,lineRaw);
					util_warning(errmsg);
					continue;
				}
			}
			else
			{	sprintf_s(errmsg, "Type (%s) unrecognized on line %d(%s) ... skipping\n",wordPtr[3],lineCnt,lineRaw);
				util_warning(errmsg);
				continue;
			}
		}
    }
	fclose(file);
	if(err!=0)
	{	sprintf_s(errmsg, "Variable (%s %s) not found ... skipping\n", input[0], input[1]);
		util_warning(errmsg);
	}
  return err;
}

// Read options from the config file
cgOption * readOptions(const char* filename)
{
	// Use modes
	util_config(filename, "bool USEGLOVE", &option.USEGLOVE);
	util_config(filename, "bool STREAM_2_VIZ", &option.STREAM_2_VIZ);
	util_config(filename, "bool STREAM_2_DRIVER", &option.STREAM_2_DRIVER);
	util_config(filename, "bool HIRES_DATA", &option.HIRES_DATA);
	
	// Glove variables
	util_config(filename, "char* glove_port", &option.glove_port);
	util_config(filename, "int baudRate", &option.baudRate);
	util_config(filename, "int rawSenor_n", &option.rawSenor_n);
	util_config(filename, "bool updateRawRange", &option.updateRawRange);

	// Hand
	util_config(filename, "char* modelFile", &option.modelFile);
	util_config(filename, "char* logFile", &option.logFile);
	util_config(filename, "int calibSenor_n", &option.calibSenor_n);
	util_config(filename, "char* driver_ip", &option.driver_ip);
	util_config(filename, "char* driver_port", &option.driver_port);

	// Mujoco
	util_config(filename, "char* viz_ip", &option.viz_ip);
	util_config(filename, "int skip", &option.skip);

	// Calibration 
	util_config(filename, "char* calibFile", &option.calibFile);
	util_config(filename, "char* userRangeFile", &option.userRangeFile);
	util_config(filename, "char* handRangeFile", &option.handRangeFile);

	return &option;
}



// Glove API ==================================

// Allocate cgdata
void cGlove_initData(cgData* d, cgOption* o)
{
	// config buffers
	d->rawSample = (cgNum*)util_malloc(sizeof(cgNum)*o->rawSenor_n, 8);			// raw glove samples
	d->rawSample_nrm = (cgNum*)util_malloc(sizeof(cgNum)*(o->rawSenor_n+1), 8);	// normalized raw glove samples
	d->calibSample = (cgNum*)util_malloc(sizeof(cgNum)*o->calibSenor_n, 8);		// Mujoco convension calibrate samples

	// allocate and load calibration + ranges
	d->calibMat = (cgNum*)util_malloc(sizeof(cgNum)*o->calibSenor_n*(o->rawSenor_n+1), 8);
	d->userRangeMat = (cgNum*)util_malloc(sizeof(cgNum)*o->rawSenor_n*2, 8);
	d->handRangeMat = (cgNum*)util_malloc(sizeof(cgNum)*o->calibSenor_n*2, 8);
	util_readFile(o->calibFile, d->calibMat, o->calibSenor_n*(o->rawSenor_n+1));
	util_readFile(o->userRangeFile, d->userRangeMat, o->rawSenor_n*2);
	util_readFile(o->handRangeFile, d->handRangeMat, o->calibSenor_n*2);
	d->valid = true;
}


// free cgdata
void cGlove_freeData(cgData* d)
{
	d->valid = false;
	util_free(d->rawSample);
	util_free(d->rawSample_nrm);
	util_free(d->calibSample);
	util_free(d->calibMat);
	util_free(d->userRangeMat);
	util_free(d->handRangeMat);
}


// Clean up
void cGlove_clean(char* errorInfo) 
{
	printf("cGlove:>\t Cleaning up cgGlove..\n");
	
	// wait for thread
	if(cgdata.updateGlove)
	{
		printf("cGlove:>\t Waiting for glove update thread to exit\n");
		cgdata.updateGlove = false;
		if(cgdata.glove_th.joinable())
		{
			cgdata.glove_th.join();
			cgdata.glove_th.~thread();
		}
		printf("cGlove:>\t Glove update thread exited\n");
	}
	
	// remove glove
	if(persistentGlove != NULL)
		delete persistentGlove;
	persistentGlove = NULL;

	// clear cgdata
	cGlove_freeData(&cgdata);

	if( errorInfo != NULL)
		util_error(errorInfo);
	else
		printf("cGlove:>\t Clean up cgGlove :: Successful\n");
} 


// connect to glove 
void cGlove_connect(cgOption* o)
{
	printf("cGlove:>\t Trying to open glove port: %s\n", o->glove_port);
	try 
	{
		persistentGlove = new CyberGlove(o->glove_port, o->baudRate);
	}
	catch (std::runtime_error name)
	{
		printf("cGlove:>\t Connection problem: %s\n", name.what());
		printf("cGlove:>\t Retrying... \n");
		try 
		{
			persistentGlove = new CyberGlove(o->glove_port, o->baudRate);
		}
		catch (std::runtime_error name)
		{
			printf("cGlove:>\t Connection problem. %s\n", name.what());
		}
	}
	if(persistentGlove == NULL) 
	{
		cGlove_clean("Couldn't create glove interface.\n");
	}
	printf("cGlove:>\t Created interface for glove.\n");
}


// Update and calibrate cgdata from the glove
void cGlove_update(cgData* d, cgOption* o)
{
	printf("cGlove:>\t cGlove update thread started\n");

	cgNum* calibSample_local = (cgNum*)util_malloc(100000+sizeof(cgNum)*o->calibSenor_n, 8); // Mujoco convension 
	int n_samples = persistentGlove->SampleSize();
	static std::vector<unsigned int> inputSample(n_samples);

	// start streaming and start update loop
	persistentGlove->StartStreaming(o->HIRES_DATA);
	while(d->updateGlove)
	{
		try
		{
			if(o->HIRES_DATA)
				persistentGlove->GetHiResSample(&inputSample.front(),	persistentGlove->SampleSize(), d->timestamp);
			else
				persistentGlove->GetSample(&inputSample.front(), persistentGlove->SampleSize(), d->timestamp);  
		}
		catch (std::runtime_error name)
		{
			printf("cGlove:>\t Error getting sample:: %s\n", name.what());
		}

		// Note : No mutex: partial info can be read by the graphics  
		for (int s=0; s<n_samples; s++)
			d->rawSample[s] = (cgNum)inputSample[s];
		
		// ??? hack to avoid clipping. Remove when resolved
		if(o->HIRES_DATA)
			for (int s=0; s<n_samples; s++)
				d->rawSample[s] *= .1;

		// normalize and calibrate
		cGlove_nrmRawSample(d->rawSample_nrm, d->rawSample, d->userRangeMat, o->rawSenor_n);
		cGlove_calibrateNrmSample(calibSample_local, d->rawSample_nrm, d->handRangeMat, d->calibMat, o->calibSenor_n, o->rawSenor_n);

		// update
		d->cgGlove.lock();
		memcpy(d->calibSample, calibSample_local, o->calibSenor_n*sizeof(cgNum));
		d->cgGlove.unlock();
	}

	// Clear buffers
	util_free(calibSample_local);
	printf("cGlove:>\t cGlove update thread exiting\n");
}


// initialize glove using options
 bool cGlove_init(cgOption* options)
{
	// Connect to Glove
	cGlove_connect(&option);

	// make cgdata
	cGlove_initData(&cgdata, &option);

	// start thread for fast udpate
	cgdata.updateGlove = true;
	cgdata.glove_th = std::thread(cGlove_update, &cgdata, &option);

	return true;
}


// get most recent glove cgdata.
void cGlove_getData(cgNum *buff, const int n_buff)
{
	if(n_buff<option.calibSenor_n)
	{
		printf("Warning:: Buffer too small to update. Minimum size should be %d", option.calibSenor_n);
		return;
	}
	cgdata.cgGlove.lock();
	memcpy(buff, cgdata.calibSample, option.calibSenor_n*sizeof(cgNum));
	cgdata.cgGlove.unlock();
}

