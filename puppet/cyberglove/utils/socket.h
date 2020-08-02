//-----------------------------------//
//  This file is part of MuJoCo.     //
//  Copyright 2009-2015 Roboti LLC.  //
//  Using with cyberglve project with special permission from Emo Todorov //
//-----------------------------------//

#pragma once

#ifdef _WIN32
	#define WIN32_LEAN_AND_MEAN
	#include <winsock2.h>
	#include <ws2tcpip.h>

#else
	#include <sys/socket.h>
	#include <netinet/in.h>
	#include <netinet/tcp.h>
	#include <arpa/inet.h>
	#include <errno.h>
	#include <fcntl.h>

	#define SOCKET int
	#define INVALID_SOCKET -1
	#define SOCKET_ERROR -1
	#define WSAEINPROGRESS EINPROGRESS
	#define WSAEWOULDBLOCK EWOULDBLOCK
	#define closesocket close
#endif


//------------------------- Socket functions -------------------------------------------

typedef enum _mjtSoc		// results from socket send/receive
{
	mjSOC_OK = 0,			// success
	mjSOC_TIMEOUT,			// timeout
	mjSOC_CLOSED			// socket is in closed (or otherwise bad) state
} mjtSoc;


// type of socket error callback
typedef void (*mjfSocErr)(const char* format, int i);


class mjSocket				// socket functionality
{
public:
	mjSocket();
	~mjSocket();

	//-------------- initialization and error functions 

	// user-defined error function pointer
	void (*_errorfunc)(const char* format, int i);

	// error function
	void _error(const char* format, int i = 0);

	// last socket error
	int mjSocketError(void);

	// init sockets (needed in Windows only)
	void mjInitSockets(void);

	// clean sockets (needed in Windows only)
	void mjClearSockets(void);

	// set blocking mode for socket
	void mjSetBlocking(SOCKET s, bool block);

	//-------------- main API

	// active: try to connect
	bool connectClient(int tmout = -1, const char* host = 0); 

	// passive: listen for connections
	bool connectServer(int tmout = -1, bool print = false, 
					   const char* hostServer = 0);

	// clear socket
	void clear(void);

	// flush input
	void flushInput(void);

	// send buffer; return mjtSoc or socket error if negative
	int sendBuffer(const char* buf, int len, int tmout = -1);

	// receive buffer; return mjtSoc or socket error if negative
	int recvBuffer(char* buf, int len, int tmout = -1);

	// read-only access
	SOCKET getSoc(void)	{return soc;}
	bool getState(void)	{return state;}

	//---------------- utility functions

	// getaddrinfo for host (default: this machine)
	struct addrinfo* getHost(bool passive, const char* port, const char* host = 0);

	// make connection
	bool tryConnect(const char* port, const char* host, int tmout);

	// use select() to wait for socket read/write, true if not timeout
	bool waitSocket(SOCKET s, bool read, int tmout);


	//---------------- settings

	char portListen[100];			// port for listening (user can change the default)
	bool verbose;					// print diagnostics
	int tmoutTryConnect;			// timeout for each connect attempt
	bool (*userexit)(void);			// check if user wants to exit before listen timeout

private:
	SOCKET	soc;					// socket object
	bool	state;					// last known state (true: connected)
	char	dummy[1000];			// used to flush input
};
