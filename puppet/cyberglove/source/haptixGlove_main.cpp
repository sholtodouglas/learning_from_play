#include <stdio.h>
#include <cstdlib>
#include <conio.h>
#include <signal.h>

#include "socket.h"
#include "cyberGlove_utils.h"
#include "matplotpp.h"
#include "haptix.h"

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

using namespace std;

// GLOVE ======================================================
cgOption* o;				// options



// VISUALIZER =================================================

// clean up vizualizer connections
void viz_clean()
{
	if( mj_connected() )
		mj_close();
}

// Connect to visualizer
void viz_connect(const char* viz_ip, const char* filename)
{
	mjInfo info;
	printf("Main:>\t Connecting to the Vizualizer\n");
	mj_connect(viz_ip);
	if( !mj_connected() )
	{
		printf("Main:>\t ERROR: Coudn't connect to the Vizualizer. Check if Mujoco haptix is running.\n");
		cGlove_clean("Coudn't connect to the Vizualizer.");
	}
	else
	{
		 mj_info(&info);
		 if(info.nq==0 && info.ngeom==0)
		 {	printf("Main:>\t ERROR: No model found. Please load appropriate Mujoco model.\n");
			cGlove_clean("No model found.");
		 }
	}
}

// Stream controls to visualizer
void viz_update(double* ctrl, double time)
{
	mjControl u;
	u.nu = o->calibSenor_n;
	u.time = (float)time;
	for(int i=0; i<o->calibSenor_n; i++)
		u.ctrl[i] = (float)ctrl[i];
	mj_set_control(&u);
}


// HARDWRE ====================================================

// connect to a socket
int socConnect(mjSocket *soc, const char* ip, int timeOut, bool isServer)
{
	int err;
	if(isServer)
		err = soc->connectServer(timeOut, true, ip);
	else
		err = soc->connectClient(timeOut, (const char*)&ip);

	if(err)
		printf("Connected");
	else
	{	soc->clear();
		printf("Timeout");		
	}
	return err;
}

// connect to the driver
void driver_connect(mjSocket *soc)
{
	soc->mjInitSockets();
	strcpy_s((char*)(soc->portListen), 100*sizeof(char), o->driver_port);
	printf("Main:> Soc connecting to Driver:: ");
	if(!socConnect(soc, o->driver_ip, 2500, true))
	{	soc->clear();
		cGlove_clean("Error Connecting to the driver.");
	}
}

// Communicate data to driver
void driver_update(mjSocket *soc)
{
	// Socket buffer
	int soc_err = 0;	
	int buf_sz = o->calibSenor_n; // Jnt;
	static cgNum* buff = (cgNum*)util_malloc(buf_sz*sizeof(cgNum), 8); 
	
	// laod buffer
	cGlove_getData(buff, buf_sz);

	soc_err = 0;
	// Send buffer 
	if(soc->getState())
		soc_err = soc->sendBuffer((char*)buff, sizeof(cgNum)*buf_sz, 2);
	else
	{	printf("Main:> Driver\n");
		soc_err = socConnect(soc, o->driver_ip, 10, true);
	}
	if(soc_err!=0)
		soc->clear();
}


// maindriver ==================================================

// main (Initiate communication with the glove and the visualizer, and update)
int main(int argc, char** argv)
{   
	clock_t start_t, end_t, total_t;
	int frame_i =0;
	mjState state;

	// Driver sockets
	mjSocket socDriver;

	// Connect to Glove ----------------------------------
	o = readOptions("cyberglove.config");
	cGlove_init(o);

	// Connect to visualizer -------------------------------
	if(o->STREAM_2_VIZ)
		viz_connect(o->viz_ip, o->modelFile);

	// Connect to hardware -------------------------------
	if(o->STREAM_2_DRIVER)
		driver_connect(&socDriver);

	// Graphs ----------------------------------
	if(o->USEGRAPHICS)
	{	printf("Main:>\t Starting graphics\n");
		Graphics_init(argc, argv, 20, 200, 600, 500, "Data Glove"); //Set up graphics
	}

	// Acquire data ----------------------------
	start_t = clock();
	printf("Main:>\t Acquiring data...\n");
	cgNum* desPos = (cgNum*)util_malloc(sizeof(cgNum)*o->calibSenor_n, 8);
	while( (Graphics_Key()!=27) && (!_kbhit()) )
	{
		// Get data from the glove
		cGlove_getData(desPos, o->calibSenor_n);

		// Stream data to  vizualizer
		if(o->STREAM_2_VIZ)
		{
			mj_get_state(&state);
			viz_update(desPos, state.time);
		}

		// Stream data to  driver
		if(o->STREAM_2_DRIVER)
			driver_update(&socDriver);
		frame_i++;
	}

	// Finalize -------------------------------
	end_t = clock();
	printf("Main:>\t Done\n");

	total_t = (long)(end_t - start_t) / CLOCKS_PER_SEC;
	printf("\n\nTotal time taken by CPU: %f sec\n", (double)total_t);
	printf("Total loop frame_i %d\n", frame_i);
	printf("Update rate %f\n", (double)frame_i/(double)(total_t));

	// Close and clean up -------------------------------
	if(o->USEGRAPHICS)
		Graphics_Close();
	
	cGlove_clean(NULL);
	//Sleep(2500);
	return 0;
}

