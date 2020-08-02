#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <stdexcept>
#include "CyberGlove.h"

const char* CyberGlove::Control::Reset           = "\x12";	// CTRL_R
const char  CyberGlove::Control::Cancel          = '\x03';	// CTRL_C
const char  CyberGlove::Control::StreamSamples   = 'S';
const char  CyberGlove::Control::GetSingleSample = 'G';

const char CyberGlove::Parameter::TimeStamp           = 'D';
const char CyberGlove::Parameter::Filter              = 'F';
const char CyberGlove::Parameter::SwitchControlsLight = 'J';
const char CyberGlove::Parameter::Light               = 'L';
const char CyberGlove::Parameter::SendQuantized       = 'Q';
const char CyberGlove::Parameter::GloveStatus         = 'U';
const char CyberGlove::Parameter::Switch              = 'W';
const char CyberGlove::Parameter::ExternalSynch       = 'Y';


const char* CyberGlove::Query::BaudRate            = "?B";
const char* CyberGlove::Query::Calibration         = "?C";
const char* CyberGlove::Query::SoftwareSensorMask  = "?M";
const char* CyberGlove::Query::SampleSize          = "?N";
const char* CyberGlove::Query::ParameterFlags      = "?P";
const char* CyberGlove::Query::SamplePeriod        = "?T";
const char* CyberGlove::Query::GloveValid          = "?G";
const char* CyberGlove::Query::Information         = "?i";
const char* CyberGlove::Query::HardwareSensorMask  = "?K";
const char* CyberGlove::Query::RightHanded         = "?R";
const char* CyberGlove::Query::SensorCount         = "?S";
const char* CyberGlove::Query::Version             = "?V";
const char* CyberGlove::Query::TimeStampEnabled    = "?D";
const char* CyberGlove::Query::FilterStatus        = "?F";
const char* CyberGlove::Query::SwitchControlsLight = "?J";
const char* CyberGlove::Query::Light               = "?L";

const char* CyberGlove::CG3::EnableUsbStream	   = "1eu";
const char* CyberGlove::CG3::StreamHiRes           = "1S";

	//if(value == 'e')
	//{// Result: Error
	//    value = ReadByte();
	//    SynchInput(value);
	//    switch(value)
	//    {
	//    case '?': throw std::runtime_error("Unknown command");
	//    case 'n': throw std::runtime_error("Too many numbers entered");
	//    case 'y': throw std::runtime_error("Synch input-rate too fast");
	//    case 'g': throw std::runtime_error("Glove not plugged in");
	//    case 's': throw std::runtime_error("Sampling rate too fast");
	//    default:  throw std::runtime_error("Unknown error");
	//    }
	//}

#undef min


CyberGlove::CyberGlove(const std::string &portName, int baudRate)
{
	LPCTSTR lpFileName = (LPCTSTR)portName.c_str();
	
	port = CreateFile(lpFileName,GENERIC_READ|GENERIC_WRITE,0,0,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,0);
	if(port == INVALID_HANDLE_VALUE)
	{
		int ecode = GetLastError();
		printf("CG:>\tCannot open port. Error code: %d\n", ecode);
		throw std::runtime_error("Cannot open port");
		exit(-1);
	}
	
	DCB params;
	GetCommState(port, &params);
	params.DCBlength = sizeof(DCB);
	params.BaudRate  = baudRate;
	params.ByteSize  = 8;
	params.StopBits  = ONESTOPBIT;
	params.Parity    = NOPARITY;

	if(!SetCommState(port, &params))
	{
		CloseHandle(port);
		throw std::runtime_error("Cannot set serial-port state");
	}

	COMMTIMEOUTS timeouts; // In milliseconds
	timeouts.ReadIntervalTimeout         = 1000;
	timeouts.ReadTotalTimeoutConstant    = 1000;
	timeouts.ReadTotalTimeoutMultiplier  = 1000;
	timeouts.WriteTotalTimeoutConstant   = 1000;
	timeouts.WriteTotalTimeoutMultiplier = 1000;
	
	if(!SetCommTimeouts(port, &timeouts))
	{
		CloseHandle(port);
		throw std::runtime_error("Cannot set serial-port timeouts");
	}

	//Sleep(5000);
	Reset();
}

CyberGlove::~CyberGlove()
{
	if(port != INVALID_HANDLE_VALUE)
		CloseHandle(port);
}

void CyberGlove::Reset(bool hardwareReset)
{
	// Purge and reset the hardware
	try
	{	StopStreaming();
		SynchInput();
		printf("CG:>\tCleaning old streams\n");
	}
	catch(std::runtime_error name)
	{
		printf("CG:>\tNo steam found\n");
	}
	
	if(hardwareReset)
	{	
		printf("CG:>\tRebooting hardware\n");
		//SendCommand(Control::Reset,NULL,0); // Doesn't work as per the documentation
		printf("CG:>\tWaiting for hardware to reboot\n");
		while(!IsGloveConnected())
			printf("CG:>\tIs Golve: %d ============= \n", (int)IsGloveConnected());
	}
	else
		ClearInput();

	isStreaming = false;
	sensorCount = (size_t)QueryByte(Query::SensorCount);
	sampleSize = (size_t)QueryByte(Query::SampleSize);
	sampleSize = 22;  //??? Hack for the moment till we fix it !
	timeStampsEnabled = (bool)QueryByte(Query::TimeStampEnabled);
	timeStampsEnabled = false; //??? Hack for the moment till we fix it !

	// Set USB as streaminig destination 
	//WriteCommand(CG3::EnableUsbStream);
	//SynchInput();

	// Set the sampling rate
	// WriteByte('T');
	// unsigned short w1 = 1440; // 80 Hz; 1440 = 115200 / 80
	// unsigned short w2 = 1;
	// WriteByte((uchar)(w1>>8));
	// WriteByte((uchar)w1);
	// WriteByte((uchar)(w2>>8));
	// WriteByte((uchar)w2);
	// printf("CG:>\t\nSampling rate==========="); 
	// SynchInput();
}

unsigned char CyberGlove::ReadByte()
{
	DWORD bytesRead;
	char result;
	if(!ReadFile(port, &result, 1, &bytesRead, NULL) || bytesRead != 1)
		throw std::runtime_error("Could not read data from serial-port");
	else
		return result;
}

void CyberGlove::WriteByte(unsigned char value)
{
	DWORD bytesWritten;
	if(!WriteFile(port, &value, 1, &bytesWritten, NULL))
		throw std::runtime_error("Could not write data to serial-port");
}

void CyberGlove::SynchInput(char value)
{
	int i=0;
	static unsigned char purged[1000];
	while(value)
	{	value = ReadByte();
		purged[i] = value;
		i = i<999? i+1:i;
		//printf("CG:>\t\%c", value);
	}
	if(i>1)
	{
		purged[i] = '\0';
		printf("CG:>\tPurged %d bytes: %s\n", i, purged);
	}
	//printf("CG:>\t (%d bytes purged)", i);
}

void CyberGlove::SynchInput1(char value)
{
	int i=0;
	static unsigned char purged[1000];
	
	printf("CG:>\t");

	for(int j=0; j<200; j++)
	{	value = ReadByte();
		purged[i] = value;
		if(value=='\n')
			printf("%c", '_');
		else
			printf("%c", value);
	}
	printf("\n");

}

void CyberGlove::ClearInput(void)
{
	static char nul[32];
	DWORD bytesRead;

	do
	{
		if(!ReadFile(port, nul, sizeof(nul), &bytesRead, NULL))
			throw std::runtime_error("Could not read data from serial-port");
	} while(bytesRead == 32);
}

void CyberGlove::WriteCommand(const char* command)
{
	if(!command)
		throw std::runtime_error("Null command");
	size_t length = strlen(command);    

	DWORD bytes;
	if(!WriteFile(port, command, length, &bytes, NULL) || bytes != length)
		throw std::runtime_error("Could not write data to serial-port");

	char value = ReadByte();

	if(value != command[0])
	{
		SynchInput(value);
		throw std::runtime_error("Response is not synchronized");
	}

	// Read past command-echo
	for(size_t idx = 1; idx < length; ++idx)
	{
		value = ReadByte();
		if(value != command[idx])
		{
			SynchInput(value);
			throw std::runtime_error("Response is not synchronized");
		}
	}
}

void CyberGlove::SetParameter(const char parameter, bool enabled)
{
	WriteByte(parameter);
	WriteByte(enabled ? 1 : 0);

	char value = ReadByte();
	SynchInput(value);

	if(value != parameter)
	{
		SynchInput(value);
		throw std::runtime_error("Response is not synchronized");
	}
}

size_t CyberGlove::SendCommand(const char* command, char* response, size_t responseMaxLength)
{
	WriteCommand(command);

	char value = ReadByte();
	size_t length = 0;
	if(response)
	{
		for(length = 0; length < responseMaxLength && value; ++length)
		{
			response[length] = value;
			value = ReadByte();
		}

		if(length < responseMaxLength)
			response[length] = '\0';
	}

	// Read the rest of the command
	SynchInput(value);

	return length;
}

unsigned char CyberGlove::QueryByte(const char* command)
{
	WriteCommand(command);

	char result = ReadByte();

	SynchInput();

	return result;
}

void CyberGlove::GetSample(unsigned int *sample, size_t size, unsigned int *timeStamp)
{
	if(!isStreaming)
	{
		WriteByte(Control::GetSingleSample);
		char value = ReadByte();
		if(value != Control::GetSingleSample)
			throw std::runtime_error("Input not synchronized");
	}
	else
	{
		char value = ReadByte();
		if(value != Control::StreamSamples)
			throw std::runtime_error("Input not synchronized");
	}

	// Number of samples to be read into the destination
	size_t sampleCount = std::min(size, sampleSize);

	// Read the samples into the destination
	for(size_t idx = 0; idx < size; ++idx)
	{	sample[idx] = (unsigned int)ReadByte();
	}

	// If more samples than available were requested, set them to 0
	for(size_t idx = sampleCount; idx < size; ++idx)
		sample[idx] = 0;

	// Read time-stamp    
	if(timeStamp)
		*timeStamp = 0;

	if(timeStampsEnabled && timeStamp)
	{
		// Flush any unneeded samples so we can get the time stamp
		for(size_t idx = size+1; idx < sampleSize; ++idx)
			ReadByte();

		unsigned char nulEncoding = ReadByte();
		*timeStamp|= (ReadByte() ^ (nulEncoding & 1));
		*timeStamp|= (ReadByte() ^ (nulEncoding & 2)) << 8;
		*timeStamp|= (ReadByte() ^ (nulEncoding & 4)) << 16;
		*timeStamp|= (ReadByte() ^ (nulEncoding & 8)) << 24;
	}

	// TODO: Read glove-status
	
	// Handle errors

	SynchInput();
}

void CyberGlove::GetHiResSample(unsigned int *sample, size_t size, unsigned int *timeStamp)
{
	if(!isHiResStream)
	{
		throw std::runtime_error("Not configured for hi-resolution data");
	}
	else if(!isStreaming)
		throw std::runtime_error("No stream found. Hi-resolution data can only be gatherned in streaming mode");
	else
	{
		// each record is 61 bytes long
		int PACKET_SIZE = 61;
		int bytesRead = 0;
	
		char bytes[255];
		for(int iread=0; iread<PACKET_SIZE; iread++)
			bytes[iread] = ReadByte();

		// extract time from record
		// hh:mm:ss:ff
		// 01234567890
		bytes[2] = 0;
		bytes[5] = 0;
		bytes[8] = 0;

		// extract time
		m_currDataTime.hour		= atoi( bytes );
		m_currDataTime.minute	= atoi( bytes + 3 );
		m_currDataTime.second	= atoi( bytes + 6 );
		m_currDataTime.frame	= atoi( bytes + 9 );
		m_currDataTime.subFrame = atoi( bytes + 11 );

		char outString[255];
		int pos = sprintf( outString, "%.2d:%.2d:%.2d:%.2d:%.2d\t", m_currDataTime.hour, m_currDataTime.minute, m_currDataTime.second,m_currDataTime.frame,m_currDataTime.subFrame  );	// extract data

		int dataOffset = 14;
		int index  = 0;
		for( int i = 0; i < 23; i++ ) 
		{
			if( i != 7 ) 
			{
				m_currHiResData[i] = ((unsigned int) bytes[dataOffset+2*index] << 8) + ((unsigned char) bytes[dataOffset+2*index+1]);
				sample[index] = m_currHiResData[i];
				index++;
			}
			else 
			{
				// index abduction, set to 0 for now will be calculated during calibration
				m_currHiResData[i] = 0;
			}
			pos += sprintf( outString+pos, "%d ", m_currHiResData[i]  );
		}
		sprintf( outString+pos, "\n" );
		//printf("%s\n", outString);
		//OutputDebugStringA( outString );
	}
}

void CyberGlove::StartStreaming(bool streamHighRes)
{
	// Start Streaming
	if(!streamHighRes)
		WriteByte(Control::StreamSamples);
	else
	{
		// set glove streaming and then start HiRes stream
		setEnableStreaming( false, 1, true, 1, false, 1 );
		WriteCommand(CG3::StreamHiRes);
	}

	isHiResStream = streamHighRes;
	isStreaming = true;
}

// Streaming options for HiRes data
int CyberGlove::setEnableStreaming( bool _wireless, int _wfm, bool _usb, int _ufm, bool _sdcard, int _sfm )
{
	char command[255];

	// frame rate multiplier of 3 (3* 30Hz = 90 Hz)
	command[0] = '1';
	command[1] = 'm';
	command[2] = 0x3;
	command[3] = '\0';
	WriteCommand(command);
	SynchInput();

	// Wireless connection
	if( _wireless ) 
	{
		WriteCommand("1ew");
		SynchInput();
		
		command[0] = '1';
		command[1] = 'w';
		command[2] = (unsigned char)_wfm;
		command[3] = '\0';
		WriteCommand(command);
		SynchInput();
	} else 
	{
		WriteCommand( "1dw");
		SynchInput();
	}

	// USB connection 
	if( _usb ) {
		WriteCommand( "1eu");
		SynchInput();

		command[0] = '1';
		command[1] = 'u';
		command[2] = (unsigned char)_ufm;
		command[3] = '\0';
		WriteCommand( command);
		SynchInput();
	} else 
	{
		WriteCommand( "1du");
		SynchInput();
	}

	// SD Card
	if( _sdcard ) {
		WriteCommand( "1es");
		SynchInput();

		command[0] = '1';
		command[1] = 's';
		command[2] = (unsigned char)_sfm;
		command[3] = '\0';
		WriteCommand( command);
		SynchInput();

	} else 
	{
		WriteCommand( "1ds");
		SynchInput();
	}

	return 0;
}

// Stop active stream
void CyberGlove::StopStreaming()
{
	WriteByte(Control::Cancel);

	ClearInput();

	isStreaming = false;
}

void CyberGlove::TimeStamp(bool enabled) 
{ 
	SetParameter(Parameter::TimeStamp, enabled); 
	timeStampsEnabled = enabled; 
}

bool   CyberGlove::IsGloveConnected()   { return (bool)QueryByte(Query::GloveValid); }
bool   CyberGlove::IsRightHanded()      { return (bool)QueryByte(Query::RightHanded); }
size_t CyberGlove::SensorCount()        { return sensorCount; }
size_t CyberGlove::SampleSize()         { return sampleSize; }











