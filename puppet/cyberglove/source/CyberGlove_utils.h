#ifndef _CYBERGLOVE_UTILS_H_
#define _CYBERGLOVE_UTILS_H_

#include <thread>
#include <mutex>

	typedef double cgNum;
	typedef struct _options
	{
		// Use modes
		bool USEGLOVE = true;
		bool USEFEEDBACK = true;
		bool STREAM_2_VIZ = true;
		bool STREAM_2_DRIVER = true;
		bool HIRES_DATA = false;

		// Glove variables
		char* glove_port = "COM1";
		int baudRate = 115200; 
		int rawSenor_n = 22;
		bool updateRawRange = false;

		// Hand
		char* modelFile = "humanoid.xml";
		int calibSenor_n = 24;
		char* driver_ip = "128.208.4.243";
		char* driver_port = "COM1";
		char* logFile ="none";

		// Calibration 
		char* calibFile = "";
		char* userRangeFile = "";
		char* handRangeFile = "";

		// Mujoco
		char* viz_ip = "128.208.4.243";
		int skip = 1;		// update teleOP every skip steps(1: updates tracking every mj_step)

		// feedback
		char* DOChan = "Dev2/port0/line0:7";
		int pulseWidth = 20; // width of feedback pulse in ms;

	}cgOption;
	
	typedef struct _data
	{
		bool valid = false;		// is data valid?
		unsigned int* timestamp;// data time stamp
		cgNum* rawSample;		// raw samples from the glove
		cgNum* rawSample_nrm;	// normalized raw glove samples
		cgNum* calibSample;		// Mujoco convension calibrate samples

		cgNum* calibMat;		// Calibration matrix
		cgNum* userRangeMat;	// User glove range
		cgNum* handRangeMat;	// Hand joint ranges

		std::thread glove_th;   // Glove background update thread 
		bool updateGlove;		// update glove with latest data?
		std::mutex cgGlove;		// Mutex on the calibSamples
	}cgData;

	extern cgData cgdata;
	extern cgOption option;

	// Glove API ===============================

	// Get the latest data from the glove
	void cGlove_getData(cgNum *buff, const int n_buff);

	//  Clean up glove
	void cGlove_clean(char* errorInfo);

	// initialize glove
	bool cGlove_init(cgOption* options);

	// Utilities ==============================

	// write message to console, pause and exit
	void util_error(const char* msg);

	// write warning message to console
	void util_warning(const char* msg);

	// allocate memory, align
	void* util_malloc(unsigned int sz, unsigned int align);

	// free memory
	void util_free(void* buf);

	// Read data from a tab(or space) seperated file
	int util_readFile(const char* Fname, cgNum* vec, const int size);
	
	// read configuration
	int util_config(const char *fileName, const char *iname, void *var);

	// Read options from the config file
	cgOption * readOptions(const char* filename);



#endif