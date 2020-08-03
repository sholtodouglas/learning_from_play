//-----------------------------------//
//  This file is part of MuJoCo.     //
//  Copyright 2009-2015 Roboti LLC.  //
//  Using with cyberglve project with special permission from Emo Todorov //
//-----------------------------------//

#define _CRT_SECURE_NO_WARNINGS
#include "socket.h"
#include "crossplatform.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


//---------------------------- OS specific socket functions -----------------------------

#ifdef _WIN32
#pragma comment(lib,"Ws2_32.lib")


// last socket error
int mjSocket::mjSocketError(void)
{
	return WSAGetLastError();
}


// init sockets
void mjSocket::mjInitSockets(void)
{
	WSADATA wsaData;
	if( WSAStartup(0x0202, &wsaData) )
		_error("error initializing Winsock");
}


// clean sockets
void mjSocket::mjClearSockets(void)
{
	if( WSACleanup() )
		_error("error cleaning up Winsock");
}


// set blocking mode for socket
void mjSocket::mjSetBlocking(SOCKET s, bool block)
{
	unsigned long value = (!block);
	if( ioctlsocket(s, FIONBIO, &value) )
		_error("set blocking mode failed with error %d", mjSocketError());
}


#else

// last socket error
int mjSocket::mjSocketError(void)
{
	return errno;
}


// init sockets (needed in Windows only)
void mjSocket::mjInitSockets(void)
{
}


// clean sockets (needed in Windows only)
void mjSocket::mjClearSockets(void)
{
}


// set blocking mode for socket
void mjSocket::mjSetBlocking(SOCKET s, bool block)
{
	int flags = fcntl(s, F_GETFL, 0); 
	if( block )
		fcntl(s, F_SETFL, flags & (!O_NONBLOCK));
	else
		fcntl(s, F_SETFL, flags | O_NONBLOCK);
}

#endif



//---------------------------- Main API -------------------------------------------------

// constructor
mjSocket::mjSocket()
{
	// clear
	soc = INVALID_SOCKET;
	state = false;
	_errorfunc = 0;

	// default settings
	strcpy(portListen, "50505");
	verbose = false;
	tmoutTryConnect = 2000;
	userexit = 0;
}



// destructor
mjSocket::~mjSocket()
{
	clear();
}



// call user-defined error, or print message and exit
void mjSocket::_error(const char* format, int i)
{
	if( _errorfunc )
		_errorfunc(format, i);
	else
	{
		char msg[1000];
		sprintf(msg, format, i);
		printf("%s\n\n", msg);
		printf("Press Enter to exit\n");
		getchar();
		exit(1);
	}
}



// try to make connection to specified host
bool mjSocket::connectClient(int tmout, const char* host)
{
	// record start time
	int tmStart = mjGetTime();
	
	// try to connect
	while( tmout<0 || mjGetTime()-tmStart<tmout )
		if( tryConnect(portListen, host, tmoutTryConnect) )
			break;

	return state;
}



// listen for connections
bool mjSocket::connectServer(int tmout, bool print, const char* hostServer)
{
	// what if we are already connected ???
	// get address info for local machine
	struct addrinfo* info = getHost(true, portListen, hostServer);
	if( !info )
		_error("getaddrinfo failed with error %d", mjSocketError());

	// print wait message
	if( print )
	{
		char adr[INET_ADDRSTRLEN];
		printf("waiting for connection on %s : %s\n", 
			inet_ntop(AF_INET, &(((struct sockaddr_in*)info->ai_addr)->sin_addr), 
				adr, INET_ADDRSTRLEN), portListen);
	}

	// create a socket for listening
	SOCKET ListenSocket = socket(info->ai_family, info->ai_socktype, info->ai_protocol);
	if( ListenSocket == INVALID_SOCKET )
		_error("socket creation failed with error %d", mjSocketError());

	// get rid of reuse error
	int on = 1;
	if( setsockopt(ListenSocket, SOL_SOCKET, SO_REUSEADDR, (char*)&on, sizeof(int)) )
		_error("setsockopt failed with error %d", mjSocketError());

	// bind, release addrinfo
	if( bind(ListenSocket, info->ai_addr, (int)info->ai_addrlen) )
		_error("socket bind failed with error %d", mjSocketError());
	freeaddrinfo(info);

	// set keepalive and non-blocking
	if( setsockopt(ListenSocket, SOL_SOCKET, SO_KEEPALIVE, (char*)&on, sizeof(int)) )
		_error("setsockopt failed with error %d", mjSocketError());
	mjSetBlocking(ListenSocket, false);

	// init timing info
	int tmStart = mjGetTime();

	// start listening
	if( listen(ListenSocket, SOMAXCONN) )
		_error("listen socket failed with error %d", mjSocketError());

	// wait/accept connections
	int remaining;
	while( tmout<0 || (remaining=tmout-(mjGetTime()-tmStart))>0 )
	{	if( waitSocket(ListenSocket, true, remaining < 100 ? remaining : 100) )
		{

			// accept connection
			soc = accept(ListenSocket, NULL, NULL);
			if( soc==INVALID_SOCKET )
				_error("accept socket failed with error %d", mjSocketError());
			state = true;
			printf("connection accepted\n");

			// set keepalive, no-delay and non-blocking
			if( setsockopt(soc, SOL_SOCKET, SO_KEEPALIVE, (char*)&on, sizeof(on)) )
				_error("setsockopt failed with error %d", mjSocketError());
			if (setsockopt(soc, IPPROTO_TCP, TCP_NODELAY, (char*)&on, sizeof(on)) )
				_error("setsockopt tcp_nodelay failed with error %d", mjSocketError());
			mjSetBlocking(soc, false);

			break;
		}

		// exit if user function says so
		else if( userexit )
		{
			if( userexit() )
				break;
		}
	}
	closesocket(ListenSocket);
	return state;
}



// close socket
void mjSocket::clear(void)
{
	if( state && closesocket(soc) )
		if( verbose )
			printf("close socket error %d", mjSocketError());

	state = false;
}



// flush input buffer
void mjSocket::flushInput(void)
{
	while( waitSocket(soc, true, 0) )
		recvBuffer(dummy, 1000, 0);
}



// send buffer
int mjSocket::sendBuffer(const char *buf, int len, int tmout)
{
	int n, ndone = 0, tmStart = mjGetTime();

	if( verbose )
		printf("SEND BUFFER %d\n", len);

	// socket assumed closed
	if( !state )
		return mjSOC_CLOSED;

	// send in chunks (usually necessary)
	while( ndone<len )
	{
		// prepare timeout info
		int remaining;
		if( tmout>=0 )
		{
			remaining = tmout - (mjGetTime() - tmStart);

			// handle timeout
			if( remaining<0 )
				return mjSOC_TIMEOUT;
		}
		else
			remaining = -1;

		// wait for socket to become writable, write
		if( waitSocket(soc, false, remaining) )
		{
			if( verbose )
				printf(" trying to send %d bytes\n", len-ndone);

			// send
			n = send(soc, buf+ndone, len-ndone, 0);

			// handle socket error
			if( n<=0 )
			{
				int err = mjSocketError();
				clear();

				if( verbose )
					printf("error in sendBuffer: %d\n", err);

				return err;
			}
			else if( verbose )
				printf(" sent %d bytes\n", len-ndone);


			// add to done
			ndone += n;
		}
	}

	return mjSOC_OK;
}



// receive buffer
int mjSocket::recvBuffer(char *buf, int len, int tmout)
{
	int n, ndone = 0, tmStart = mjGetTime();

	if( verbose )
		printf("RECEIVE BUFFER %d\n", len);

	// socket assumed closed
	if( !state )
	{
		if( verbose )
			printf("--- bad state\n");
		return mjSOC_CLOSED;
	}

	// receive in chunks (usually necessary)
	while( ndone<len )
	{
		// prepare timeout info
		int remaining;
		if( tmout>0 )		//??? Vik - its was if( tmout>=0 )
		{
			remaining = tmout - (mjGetTime() - tmStart);

			// handle timeout
			if( remaining<=0 )
			{
				if( verbose )
					printf("--- timeout: %d, %d, %d, %d\n", 
						tmout, remaining, mjGetTime(), tmStart);
				return mjSOC_TIMEOUT;
			}
		}
		else
			remaining = -1;

		// wait for socket to become readable, read
		if( waitSocket(soc, true, remaining) )
		{
			if( verbose )
				printf(" waiting for %d bytes\n", len-ndone);

			// recv
			n = recv(soc, buf+ndone, len-ndone, 0);

			// handle socket error
			if( n<=0 )
			{
				int err = (n==0) ? 2 : mjSocketError();
				clear();

				if( verbose )
					printf("error in recvBuffer: %d\n", err);

				return err;
			}
			else if( verbose )
				printf(" received %d bytes\n", n);

			// add to done
			ndone += n;
		}
		else if( verbose )
			printf("could not wait: %d\n", remaining);
	}

	return mjSOC_OK;
}



//-------------------------mjSocket utility functions ---------------------------------

// getaddrinfo for host (default: this machine)
struct addrinfo* mjSocket::getHost(bool passive, const char* port, const char* host)
{
	struct addrinfo* info = 0;

	// set hints
	struct addrinfo hints;
	memset(&hints, 0, sizeof(struct addrinfo));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;
	if( passive )
		hints.ai_flags = AI_PASSIVE;

	// get address info
	char autoname[100];
	gethostname(autoname, 100);
	
	if( getaddrinfo((host ? host: autoname), port, &hints, &info) )
		return 0;
	else
		return info;	
}



// make one active connection
bool mjSocket::tryConnect(const char* port, const char* host, int tmout)
{
	// resolve the server address and port
	struct addrinfo* info = getHost(false, port, host);
	if( !info )
	{
		if( verbose )
			printf("tryConnect could not getaddrinfo for %s:%s\n", host, port);
		return false;
	}

	// create socket, free addrinfo
	soc = socket(info->ai_family, info->ai_socktype, info->ai_protocol);
	if( soc==INVALID_SOCKET )
		_error("socket creation failed with error %d", mjSocketError());

	// set keepalive and non-blocking
	int on = 1;
	if( setsockopt(soc, SOL_SOCKET, SO_KEEPALIVE, (char*)&on, sizeof(on)) )
		_error("setsockopt failed with error %d", mjSocketError());
	if (setsockopt(soc, IPPROTO_TCP, TCP_NODELAY, (char*)&on, sizeof(on)) )
		_error("setsockopt tcp_nodelay failed with error %d", mjSocketError());
	mjSetBlocking(soc, false);

	// print
	if( verbose )
	{
		char adr[INET_ADDRSTRLEN];
		printf("\ntryConnect: host = %s, port = %s, ip = %s\n", 
			host, port, inet_ntop(AF_INET, &(((struct sockaddr_in*)info->ai_addr)->sin_addr), 
			adr, INET_ADDRSTRLEN));
	}

	// try to connect (expect EINPROGRESS or WWOULDBLOCK), free info
	int res = connect(soc, info->ai_addr, (int)info->ai_addrlen);
	int err = mjSocketError();
	if( res && (err!=WSAEINPROGRESS && err!=WSAEWOULDBLOCK) )
		_error("connect socket failed with error %d", mjSocketError());
	freeaddrinfo(info);

	// if connection was not made immediately, wait
	if( res==0 || waitSocket(soc, false, tmout) )
	{
		// check for error
		int err = 0;
		socklen_t size = sizeof(int);
		if( getsockopt(soc, SOL_SOCKET, SO_ERROR, (char*)&err, &size) )
			_error("getsockopt failed with error %d", mjSocketError());
		
		// no error: connection successful
		if( !err )
		{
			if( verbose )
				printf("   successful connection\n");
			state = true;
			return true;
		}
	}

	// failure
	closesocket(soc);
	return false;
}



// use select() to wait for socket operation, true if not timeout
bool mjSocket::waitSocket(SOCKET s, bool read, int tmout)
{
	// make set with socket s
	fd_set set;
	FD_ZERO(&set);
	FD_SET(s, &set);

	// timeout structure
	struct timeval tm;
	tm.tv_sec = tmout/1000;
	tm.tv_usec = ((long)(tmout%1000)) * 1000;

	// call with read or write set
	int result = 0;
	if( read )
		result = select((int)s+1, &set, 0, 0, tmout>=0 ? &tm : 0);
	else
		result = select((int)s+1, 0, &set, 0, tmout>=0 ? &tm : 0);

	// check for error
	if( result==SOCKET_ERROR )
		_error("select socket failed with error %d", mjSocketError());

	return (result==1);
}
