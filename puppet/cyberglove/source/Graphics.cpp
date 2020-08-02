#include "matplotpp.h"
#include "CyberGlove_utils.h"

extern cgData cgdata;
extern cgOption option;

// Configure plotting ========================================================  
void Plot::DISPLAY()
{	
	vector<double> raw(option.rawSenor_n), raw_nrm(option.rawSenor_n), calib(option.calibSenor_n);

	if(!cgdata.valid)
		return;

	for(int i=0; i<option.rawSenor_n; i++)
	{	raw[i] = (double)cgdata.rawSample[i];
		raw_nrm[i] = (double)cgdata.rawSample_nrm[i];
	}
	
	cgdata.cgGlove.lock();
	for(int i=0; i<option.calibSenor_n; i++)
	{	calib[i] = (double)cgdata.calibSample[i];
	}
	cgdata.cgGlove.unlock();

	subplot(3,1,1); title("Raw Samples");
	bar(raw);
	axis(0, option.rawSenor_n+1, -5, 300);

	subplot(3,1,2); title("Normalized Raw Samples");
	bar(raw_nrm);
	axis(0, option.rawSenor_n+1, -.5, 1.5);

	subplot(3,1,3); title("Calibrated Samples");
	bar(calib);
	axis(0, option.calibSenor_n+1, -3, 3);
}