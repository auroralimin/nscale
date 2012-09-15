/*
 * DataBuffer.cpp
 *
 *  Created on: Aug 14, 2012
 *      Author: tcpan
 */

#include "DataBuffer.h"
#include <cstdlib>
#include <cstdio>

namespace cci {
namespace rt {

const int DataBuffer::READY = 11;
const int DataBuffer::STOP = 10;
const int DataBuffer::EMPTY = 12;
const int DataBuffer::FULL = 13;
const int DataBuffer::BAD_DATA = -11;



DataBuffer::DataBuffer(int _capacity) : capacity(_capacity), status(DataBuffer::READY) {

}

DataBuffer::~DataBuffer() {
	if (!buffer.empty()) {
		Debug::print("WARNING:  DataBuffer is not empty.  likely to have leaked memory.\n");
	}
}

void DataBuffer::dumpBuffer() {
	while (~buffer.empty()) {
		DataType d = buffer.front();
		buffer.pop();
		free(d.second);
	}
}


int DataBuffer::push(DataType const data) {

	if (isStopped()) return STOP;
	if (isFull()) return FULL;
	if (data.first == 0 || data.second == NULL) return BAD_DATA;

	if (this->canPush()) buffer.push(data);

	//Debug::print("DataBuffer: push called.  %d load\n", buffer.size());

	return status;  // should have value READY.
}

int DataBuffer::pop(DataType &data) {
	if (!canPop()) return EMPTY;

	data = buffer.front();
	buffer.pop();

	//Debug::print("DataBuffer: pop called.  %d load\n", buffer.size());

	return status;
}


} /* namespace rt */
} /* namespace cci */