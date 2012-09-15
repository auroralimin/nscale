/*
 * ADIOSSave_Reduce.cpp
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#include "ADIOSSave_Reduce.h"
#include "Debug.h"
#include "mpi.h"

#include "FileUtils.h"

#include "CVImage.h"
#include "UtilsADIOS.h"


namespace cci {
namespace rt {
namespace adios {


ADIOSSave_Reduce::ADIOSSave_Reduce(MPI_Comm const * _parent_comm, int const _gid,
		DataBuffer *_input, DataBuffer *_output,
		std::string &outDir, std::string &iocode, int total, int _buffer_max,
		int tile_max, int imagename_max, int filename_max,
		ADIOSManager *_iomanager, cciutils::SCIOLogSession *_logsession) :
		Action_I(_parent_comm, _gid, _input, _output, _logsession), iomanager(_iomanager),
		local_iter(0), local_total(0),
		c(0),
		buffer_max(_buffer_max) {

	assert(_input != NULL);

	// determine if we are using AMR, if so, don't append time points
	bool appendInTime = true;
	if (strcmp(iocode.c_str(), "MPI_AMR") == 0 ||
		strcmp(iocode.c_str(), "gap-MPI_AMR") == 0) appendInTime = false;

	// always overwrite.
	bool overwrite = true;

	// and the stages to capture.
	std::vector<int> stages;
	for (int i = 0; i < 200; i++) {
		stages.push_back(i);
	}

	int minRank = 0;
	MPI_Allreduce(&rank, &minRank, 1, MPI_INT, MPI_MIN, comm);
	if (rank == minRank &&
			!(strcmp(iocode.c_str(), "NULL") == 0 ||
			strcmp(iocode.c_str(), "gap-NULL") == 0)) {
		// create the directory
		FileUtils futils;
		futils.mkdirs(outDir);
		printf("made directories for %s\n", outDir.c_str());
	}


	writer = iomanager->allocateWriter(outDir, std::string("bp"), overwrite,
			appendInTime, stages,
			total, buffer_max,
			tile_max, imagename_max, filename_max,
			comm, groupid);
	writer->setLogSession(this->logsession);

}

ADIOSSave_Reduce::~ADIOSSave_Reduce() {
	Debug::print("%s destructor called.  total written out is %d\n", getClassName(), local_total);


	if (writer) writer->persistCountInfo();

	iomanager->freeWriter(writer);

//	MPI_Barrier(comm);
}

int ADIOSSave_Reduce::run() {

	long long t1, t2;

	t1 = ::cciutils::event::timestampInUS();

	int max_iter = 0;

	// status is set to WAIT or READY, since we can be DONE only if everyone is DONE
	int status = (this->inputBuf->isEmpty() ? Communicator_I::WAIT : Communicator_I::READY);

	int buffer[2], gbuffer[2];

//	if (test_input_status == DONE)
//		Debug::print("TEST start input status = %d\n", input_status);


	// first get the local states - active or inactive
	if (this->inputBuf->isFinished()) {
		buffer[0] = 0;
		c++;
	} else {
		buffer[0] = 1;
	}
	// next predict the local iterations.  write either when full, or when done.
	if (this->inputBuf->isFull() ||
			(!(this->inputBuf->isEmpty()) && this->inputBuf->isStopped())) {
		// not done and has full buffer, or done and has some data
		// increment self and accumulate
		buffer[1] = local_iter + 1;
	} else {
		buffer[1] = local_iter;
	}
	MPI_Allreduce(buffer, gbuffer, 2, MPI_INT, MPI_MAX, comm);
//	if (test_input_status == DONE)
//		Debug::print("TEST 1 input status = %d\n", input_status);
	if (gbuffer[0] == 0) {
		// everyone is done. - reached when all inputBuf are finished.
		status = Communicator_I::DONE;
	}
	max_iter = gbuffer[1];


	if (status == Communicator_I::DONE) Debug::print("%s call_count = %ld, input_status = %d, status = %d, max_iter = %d, local_iter = %d, buffer size = %d\n", getClassName(), c, status, max_iter, local_iter, this->inputBuf->getBufferSize());

//	if (test_input_status == DONE)
//		Debug::print("TEST 2 input status = %d\n", input_status);

	t2 = ::cciutils::event::timestampInUS();
	//if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO MPI update iter"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));


	// local_count > 0 && max_iter <= local_iter;  //new data to write
	// local_count == 0 && max_iter > local_iter; //catch up with empty
	// local_count > 0 && max_iter > local_iter;  //catch up with some writes
	// local_count == 0 && max_iter <= local_iter;  // update remote, nothing to write locally.

	//	Debug::print("%s rank %d max iter = %d, local_iter = %d\n", getClassName(), rank, max_iter, local_iter);

	/**
	 *  catch up.
	 */
	// catch up.  so flush whatever's in buffer.
	while (max_iter > local_iter) {
		Debug::print("%s write out: IO group %d rank %d, write iter %d, max_iter = %d, tile count %d\n", getClassName(), groupid, rank, local_iter, max_iter, this->inputBuf->getBufferSize());
		process();
	}

//	if (test_input_status == DONE)
//		Debug::print("TEST end input status = %d\n", input_status);

	//Debug::print("%s rank %d returning input status %d\n", getClassName(), rank, input_status);
	return status;
}

int ADIOSSave_Reduce::process() {

	// move data from action's buffer to adios' buffer

	DataBuffer::DataType data;
	int input_size;  // allocate output vars because these are references
	void *input;
	int result;

	while (!this->inputBuf->isEmpty()) {
		result = this->inputBuf->pop(data);
		input_size = data.first;
		input = data.second;

		if (input != NULL) {
			CVImage *adios_img = new CVImage(input_size, input);
			writer->saveIntermediate(adios_img, 1);

			delete adios_img;
			free(input);
			input = NULL;

			++local_total;
		}
	}

//	Debug::print("%s calling ADIOS to persist the files\n", getClassName());
	writer->persist(local_iter);
	//MPI_Barrier(comm);
	writer->clearBuffer();
	++local_iter;

//	Debug::print("%s persist completes at rank %d\n", getClassName(), rank);

	return 0;
}

} /* namespace adios */
} /* namespace rt */
} /* namespace cci */