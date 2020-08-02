//-----------------------------------//
//  This file is part of MuJoCo.     //
//  Copyright 2009-2015 Roboti LLC.  //
//  Using with cyberglve project with special permission from Emo Todorov //
//-----------------------------------//

#include "crossplatform.h"
#include "stdio.h"

#pragma comment(lib, "winmm.lib")


//---------------------------- Critical Section -----------------------------------------

// constructor
mjCriticalSection::mjCriticalSection()
{
	InitializeCriticalSection(&cs);
}



// destructor
mjCriticalSection::~mjCriticalSection()
{
	DeleteCriticalSection(&cs);
}



// enter
void mjCriticalSection::Enter(void)
{
	EnterCriticalSection(&cs);
}



// try to enter
bool mjCriticalSection::TryEnter(void)
{
	return (TryEnterCriticalSection(&cs)!=0);
}



// leave
void mjCriticalSection::Leave(void)
{
	LeaveCriticalSection(&cs);
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

// constructor: create event and reset
mjEvent::mjEvent()
{
	// create event in reset state
	ev = CreateEvent(0, true, false, 0);
}



// destructor: close handle
mjEvent::~mjEvent()
{
	CloseHandle(ev);
}



// get event state
bool mjEvent::Get(void)
{
	return( WaitForSingleObject(ev, 0) == WAIT_OBJECT_0 );
}



// set event
void mjEvent::Set(void)
{
	SetEvent(ev);
}



// reset event
void mjEvent::Reset(void)
{
	ResetEvent(ev);
}



// wait for event to be set by another thread
bool mjEvent::Wait(int msec)
{
	if( msec<0 )
		return( WaitForSingleObject(ev, INFINITE) == WAIT_OBJECT_0 );
	else
		return( WaitForSingleObject(ev, msec) == WAIT_OBJECT_0 );
}



//---------------------------- Semaphores -----------------------------------------------

// constructor
mjSemaphore::mjSemaphore(int initCount)
{
	sem = CreateSemaphore(NULL, initCount, 10000, NULL);		// max count 10000
}



// destructor
mjSemaphore::~mjSemaphore()
{
	CloseHandle(sem);
}



// timed wait
bool mjSemaphore::Wait(int msec)
{
	if( msec<0 )
		return( WaitForSingleObject(sem, INFINITE)==WAIT_OBJECT_0 );
	else
		return( WaitForSingleObject(sem, msec)==WAIT_OBJECT_0 );
}



// increase semaphore count
void mjSemaphore::Increase(int count)
{
	ReleaseSemaphore(sem, count, NULL);
}



// get the number of processor cores on the local machine
int mjGetNCores(void)
{
	SYSTEM_INFO info;
	GetSystemInfo(&info);
	return info.dwNumberOfProcessors;
}



//---------------------------- Timing ---------------------------------------------------

static bool _tmInitialized = false;
static int _tmBase;
static LARGE_INTEGER _tmFreqHR;
static LARGE_INTEGER _tmBaseHR;


// initialize 1 msec timer, record base time
void mjBeginTime(void)
{
	if( _tmInitialized )
		return;

	// initialize multimedia timer
	timeBeginPeriod(1);
	_tmBase = timeGetTime();

	// initialize high performance timer
	QueryPerformanceFrequency(&_tmFreqHR);
	QueryPerformanceCounter(&_tmBaseHR);

	_tmInitialized = true;
}



// close timer
void mjEndTime(void)
{
	if( _tmInitialized )
	{
		timeEndPeriod(1);
		_tmInitialized = false;
	}
}



// get time in milliseconds scince initialization
int mjGetTime(void)
{
	if( !_tmInitialized )
		mjBeginTime();

	return (timeGetTime() - _tmBase);
}



// get time in microseconds scince initialization
long long int mjGetTimeHR(void)
{
	if( !_tmInitialized )
		mjBeginTime();

	LARGE_INTEGER tm;
	QueryPerformanceCounter(&tm);
	return (long long int) (((tm.QuadPart - _tmBaseHR.QuadPart)*1000000)/_tmFreqHR.QuadPart);
}



// sleep for given number of milliseconds
void mjSleep(unsigned int msec)
{
	if( !_tmInitialized )
		mjBeginTime();

	Sleep(msec);
}
