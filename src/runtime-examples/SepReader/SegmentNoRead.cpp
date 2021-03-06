/*
 * SegmentNoRead.cpp
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#include "SegmentNoRead.h"
#include "Debug.h"
#include "opencv2/opencv.hpp"
#include "CVImage.h"
#include <string>
#include "SCIOHistologicalEntities.h"
#include "FileUtils.h"
#include "TypeUtils.h"

#include "CmdlineParser.h"

namespace cci {
namespace rt {
namespace adios {

bool SegmentNoRead::initParams() {
	params.add_options()
			("device_type,r", boost::program_options::value< std::string >()->default_value(std::string("cpu")), "processing device type. cpu/gpu")
//			("device_id,w", boost::program_options::value<int>()->default_value(1), "device ID. (applies to GPU only)")
			;
	return true;
}

boost::program_options::options_description SegmentNoRead::params("Compute Options");
bool SegmentNoRead::param_init = SegmentNoRead::initParams();


SegmentNoRead::SegmentNoRead(MPI_Comm const * _parent_comm, int const _gid,
		DataBuffer *_input, DataBuffer *_output,
		boost::program_options::variables_map &_vm,
		cci::common::LogSession *_logsession) :
				Action_I(_parent_comm, _gid, _input, _output, _logsession), output_count(0)
 	 {
	assert(_input != NULL);
	assert(_output != NULL);

	compressing = cci::rt::CmdlineParser::getParamValueByName<bool>(_vm, cci::rt::DataBuffer::PARAM_COMPRESSION);
	std::string proctype = cci::rt::CmdlineParser::getParamValueByName<std::string>(_vm, "device_type");


	if (strcmp(proctype.c_str(), "cpu")) proc_code = cci::common::type::DEVICE_CPU;
	else if (strcmp(proctype.c_str(), "gpu")) {
		proc_code = cci::common::type::DEVICE_GPU;
	}

}

SegmentNoRead::~SegmentNoRead() {
	cci::common::Debug::print("%s destructor called.\n", getClassName());
}

int SegmentNoRead::compute(int const &input_size , void * const &input,
			int &output_size, void * &output) {
	if (input_size == 0 || input == NULL) return -1;

	long long t1, t2;

	t1 = ::cci::common::event::timestampInUS();

	CVImage * img;
	if (compressing) img = new CVImage(input_size, input, CVImage::ENCODE_Z);
	else img = new CVImage(input_size, input, CVImage::ENCODE_RAW);

	int dummy1, dummy2;
	std::string fn(img->getSourceFileName(dummy1, dummy2));
	std::string imagename(img->getImageName(dummy1, dummy2));
	int tilex = img->getMetadata().info.x_offset;
	int tiley = img->getMetadata().info.y_offset;

	cv::Mat im = img->getImage();

	//sleep(rand() % 3 + 1);
	t2 = ::cci::common::event::timestampInUS();
	char len[21];  // max length of uint64 is 20 digits
	memset(len, 0, 21);
	sprintf(len, "%lu", (long)(im.dataend) - (long)(im.datastart));
	if (logsession != NULL) logsession->log(cci::common::event(0, std::string("deserialize"), t1, t2, std::string(len), ::cci::common::event::FILE_I));


	if (!im.data) {
		im.release();
		delete img;
		return nscale::SCIOHistologicalEntities::INVALID_IMAGE;
	}

	t1 = ::cci::common::event::timestampInUS();

	// real computation:
	int status = ::nscale::SCIOHistologicalEntities::SUCCESS;
	int *bbox = NULL;
	int compcount;
	cv::Mat mask = cv::Mat::zeros(im.size(), CV_32SC1);
//	if (proc_code == cci::common::type::DEVICE_GPU ) {
//		nscale::gpu::SCIOHistologicalEntities *seg = new nscale::gpu::SCIOHistologicalEntities(fn);
//		status = seg->segmentNuclei(std::string(input), std::string(mask), compcount, bbox, NULL, session, writer);
//		delete seg;
//
//	} else {
	//cci::common::Debug::print("%s running for %s\n", getClassName(), fn.c_str());

/// //DEBUGGING ONLY
	nscale::SCIOHistologicalEntities *seg = new nscale::SCIOHistologicalEntities(fn);
	status = seg->segmentNuclei(im, mask, compcount, bbox, NULL, NULL);
	delete seg;
	//cci::common::Debug::print("%s complete for %s\n", getClassName(), fn.c_str());
//	}

	t2 = ::cci::common::event::timestampInUS();
	std::string eventName;
	if (status == nscale::SCIOHistologicalEntities::SUCCESS) {
		eventName.assign("computeFull");
	} else if (status == nscale::SCIOHistologicalEntities::BACKGROUND) {
		eventName.assign("computeNoFG");
	} else if (status == nscale::SCIOHistologicalEntities::NO_CANDIDATES_LEFT) {
		eventName.assign("computeNoNU");
	} else {
		eventName.assign("computeOTHER");
	}
	if (logsession != NULL) logsession->log(cci::common::event(90, eventName, t1, t2, std::string("1"), ::cci::common::event::COMPUTE));

	if (status == ::nscale::SCIOHistologicalEntities::SUCCESS) {
		t1 = ::cci::common::event::timestampInUS();
		CVImage *oimg = new CVImage(mask, imagename, fn, tilex, tiley);
//		CVImage *oimg = new CVImage(im, imagename, fn, tilex, tiley);
		if (compressing) oimg->serialize(output_size, output, CVImage::ENCODE_Z);
		else oimg->serialize(output_size, output);
		// clean up
		delete oimg;


		t2 = ::cci::common::event::timestampInUS();
		memset(len, 0, 21);
		sprintf(len, "%lu", (long)output_size);
		if (logsession != NULL) logsession->log(cci::common::event(90, std::string("serialize"), t1, t2, std::string(len), ::cci::common::event::MEM_IO));

	}
	if (bbox != NULL) free(bbox);
	im.release();
	delete img;
	mask.release();
	return status;
}

int SegmentNoRead::run() {

	if (this->inputBuf->isFinished()) {
		cci::common::Debug::print("%s input DONE.  input count = %d, output count = %d\n", getClassName(), call_count, output_count);
		this->outputBuf->stop();

		return Communicator_I::DONE;
	} else if (this->outputBuf->isStopped()) {
		cci::common::Debug::print("%s output DONE.  input count = %d, output count = %d\n", getClassName(), call_count, output_count);
		this->inputBuf->stop();

		if (!this->inputBuf->isFinished()) cci::common::Debug::print("WARNING: %s input buffer is not empty.\n", getClassName());
		return Communicator_I::DONE;
	} else if (!this->inputBuf->canPop() || !this->outputBuf->canPush()) {
		return Communicator_I::WAIT;
	}

	DataBuffer::DataType data;
	int output_size, input_size;
	void *output = NULL, *input = NULL;


	int bstat = this->inputBuf->pop(data);
	if (bstat == DataBuffer::EMPTY) {
		return Communicator_I::WAIT;
	}
	input_size = data.first;
	input = data.second;

//		cci::common::Debug::print("%s READY and getting input:  call count= %d\n", getClassName(), call_count);

	int result = compute(input_size, input, output_size, output);
	call_count++;


	if (result == ::nscale::SCIOHistologicalEntities::SUCCESS) {
//			cci::common::Debug::print("%s bufferring output:  call count= %d\n", getClassName(), call_count);
		++output_count;
		bstat = this->outputBuf->push(std::make_pair(output_size, output));

		if (bstat == DataBuffer::STOP) {
			cci::common::Debug::print("ERROR: %s can't push into buffer.  status STOP.  Should have caught this earlier. \n", getClassName());
			this->inputBuf->push(data);
			this->inputBuf->stop();
			free(output);
			return Communicator_I::DONE;
		} else if (bstat == DataBuffer::FULL) {
			cci::common::Debug::print("WARNING: %s can't push into buffer.  status FULL.  Should have caught this earlier.\n", getClassName());
			this->inputBuf->push(data);
			free(output);
			return Communicator_I::WAIT;
		} else {
			if (input != NULL) {
				free(input);
				input = NULL;
			}
			return Communicator_I::READY;
		}
	} else {
		if (input != NULL) {
			free(input);
			input = NULL;
		}
		return Communicator_I::READY;
	}


}

}
} /* namespace rt */
} /* namespace cci */
