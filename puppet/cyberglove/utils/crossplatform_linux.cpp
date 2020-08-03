//-----------------------------------//
//  This file is part of MuJoCo.     //
//  Copyright 2009-2014 Roboti LLC.  //
//  Using with cyberglve project with special permission from Emo Todorov //
//-----------------------------------//

//#include "common/errmem.h"
#include "crossplatform.h"

#include "unistd.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <sys/time.h>
#include <sys/types.h>
#include <sys/sysctl.h>



//---------------------------- Critical Section -----------------------------------------
/*
// constructor
mjCriticalSection::mjCriticalSection()
{
  pthread_mutex_init(&mtx, 0);
}



// destructor
mjCriticalSection::~mjCriticalSection()
{
  pthread_mutex_destroy(&mtx);
}



// enter
void mjCriticalSection::Enter(void)
{
  pthread_mutex_lock(&mtx);
}



// try to enter
bool mjCriticalSection::TryEnter(void)
{
	return (bool)pthread_mutex_trylock(&mtx);
}



// leave
void mjCriticalSection::Leave(void)
{
  pthread_mutex_unlock(&mtx);
}



// lock consructor
mjCriticalSectionLock::mjCriticalSectionLock(mjCriticalSection* _pcs)
{
  pcs = _pcs;
  pcs->Enter();
}



// lock destructor
mjCriticalSectionLock::~mjCriticalSectionLock()
{
  pcs->Leave();
  pcs = 0;
}



//---------------------------- Event: Single --------------------------------------------

// constructor: create
mjEvent::mjEvent()
{
	// create mutex and condition variable, clear state
	pthread_mutex_init(&mtx, 0);
	pthread_cond_init(&cnd, 0);
	state = false;
}



// destructor: close
mjEvent::~mjEvent()
{
	pthread_cond_destroy(&cnd);
	pthread_mutex_destroy(&mtx);
}



// get event state
bool mjEvent::Get(void)
{
	pthread_mutex_lock(&mtx);
	bool result = state;
	pthread_mutex_unlock(&mtx);

	return result;
}



// set event
void mjEvent::Set(void)
{
	pthread_mutex_lock(&mtx);
	if( !state )
	{
		state = true;
		pthread_cond_broadcast(&cnd);
	}
	pthread_mutex_unlock(&mtx);
}



// reset event
void mjEvent::Reset(void)
{
	pthread_mutex_lock(&mtx);
	state = false;
	pthread_mutex_unlock(&mtx);
}



// wait for event to be set by another thread
bool mjEvent::Wait(int msec)
{
	bool res = false;

	pthread_mutex_lock(&mtx);

	if( !state )
	{
		if( msec<0 )
			res = (pthread_cond_wait(&cnd, &mtx) == 0);
		else
		{
			tmout.tv_sec = msec/1000;
			tmout.tv_nsec = 100000 * (long)(msec%1000);
			res = (pthread_cond_timedwait(&cnd, &mtx, &tmout) == 0);
		}
	}

	pthread_mutex_unlock(&mtx);

	return res;
}



//---------------------------- Semaphores -----------------------------------------------

// constructor
mjSemaphore::mjSemaphore(int initCount)
{
	sem_init(&sem, 0, initCount);
}



// destructor
mjSemaphore::~mjSemaphore()
{
	sem_destroy(&sem);
}



// timed wait
bool mjSemaphore::Wait(int msec)
{
	if( msec<0 )
		return( sem_wait(&sem)==0 );
	else if( msec==0 )
		return( sem_trywait(&sem)==0 );
	else
	{
		tmout.tv_sec = msec/1000;
		tmout.tv_nsec = 100000 * (long)(msec%1000);
		return( sem_timedwait(&sem, &tmout)==0 );
	}
}



// increase semaphore count
void mjSemaphore::Increase(int count)
{
	for( int i=0; i<count; i++ )
		sem_post(&sem);
}



// get number of processors
int mjGetNCores(void)
{
	int num = 1;

#if defined(__APPLE__)
	size_t length = sizeof(int); 
	sysctlbyname("hw.ncpu", &num, &length, NULL, 0); 

#elif defined(_SC_NPROCESSORS_ONLN)
	num = sysconf(_SC_NPROCESSORS_ONLN);

#else
	FILE* fp = 0;
	system("cat /proc/cpuinfo | grep -c processor > CountProc.txt");
	fp = fopen("CountProc.txt", "rt");
	if( fp )
	{
		fscanf(fp, "%d", &num);
		fclose(fp);
		system("rm CountProc.txt");
	}
#endif

	return num;
}

*/

//---------------------------- Timing ---------------------------------------------------

static bool _tmInitialized = false;
static struct timeval _tmBase;


// record base time
void mjBeginTime(void)
{
	if( _tmInitialized )
		return;

	gettimeofday(&_tmBase, 0);
	_tmInitialized = true;
}



// nothing to do in Linux
void mjEndTime(void)
{
}



// get time in milliseconds scince initialization
int mjGetTime(void)
{
	if( !_tmInitialized )
		mjBeginTime();

	struct timeval now;
	gettimeofday(&now, 0);
	return (int)(now.tv_sec*1000 - _tmBase.tv_sec*1000 +  
				 now.tv_usec/1000 - _tmBase.tv_usec/1000);
}



// get time in microseconds scince initialization
long long int mjGetTimeHR(void)
{
	if( !_tmInitialized )
		mjBeginTime();

	struct timeval now;
	gettimeofday(&now, 0);
	return (int)(now.tv_sec*1000000 - _tmBase.tv_sec*1000000 +  
				 now.tv_usec - _tmBase.tv_usec);
}



// sleep for given number of milliseconds
void mjSleep(unsigned int msec)
{
  usleep(msec*1000);
}
