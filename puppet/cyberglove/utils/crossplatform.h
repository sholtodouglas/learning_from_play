//-----------------------------------//
//  This file is part of MuJoCo.     //
//  Copyright 2009-2015 Roboti LLC.  //
//  Using with cyberglve project with special permission from Emo Todorov //
//-----------------------------------//

#pragma once


//------------------------- OS-specific include, libraries, defines --------------------

#ifdef _WIN32
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
	#include <mmsystem.h>
	
	#define PATHSYMBOL '\\'

#else
	#include <pthread.h>
	#include <semaphore.h>
	#include <sys/types.h>
	#include <errno.h>
	#include <netdb.h>
	#include <unistd.h>
	#include <X11/Xlib.h>

	#define PATHSYMBOL '/'
#endif


//---------------------------- Critical Section -----------------------------------------

// section object
class mjCriticalSection
{
public:
	mjCriticalSection();
	~mjCriticalSection();

	void Enter(void);
	void Leave(void);
	bool TryEnter(void);

private:
#ifdef _WIN32
	CRITICAL_SECTION cs;
#else
	pthread_mutex_t mtx;
#endif
};


// lock critical section
class mjCriticalSectionLock
{
public:
	mjCriticalSectionLock(mjCriticalSection* pcs);
	~mjCriticalSectionLock();

private:
	mjCriticalSection* pcs;
};


//---------------------------- Events -------------------------------------------------

// single event
class mjEvent
{
public:
	mjEvent();
	~mjEvent();

	bool Get(void);
	void Set(void);
	void Reset(void);
	bool Wait(int msec = -1);

private:
#ifdef _WIN32
	HANDLE ev;
#else
	pthread_mutex_t mtx;
	pthread_cond_t cnd;
	struct timespec tmout;
	bool state;
#endif
};


//---------------------------- Semaphores -----------------------------------------------

class mjSemaphore
{
public:
	mjSemaphore(int initCount = 0);
	~mjSemaphore();

	bool Wait(int msec = -1);		// -1: infinite wait; 0: return immediately
	void Increase(int count = 1);

private:
#ifdef _WIN32
	HANDLE sem;
#else
	sem_t sem;
	struct timespec tmout;
#endif
};


// get the number of processor cores on the local machine
int mjGetNCores(void);


//---------------------------- Timing ---------------------------------------------------

// initialize 1 msec timer (on Windows), record base time
void mjBeginTime(void);

// close timer (on Windows)
void mjEndTime(void);

// get time in milliseconds scince mjBeginTime
int mjGetTime(void);

// get time in microseconds since mjBeginTime
long long int mjGetTimeHR(void);

// sleep for given number of milliseconds
void mjSleep(unsigned int msec);
