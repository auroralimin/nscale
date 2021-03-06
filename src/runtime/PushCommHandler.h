/*
 * PushCommHandler.h
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#ifndef PUSHCOMMHANDLER_H_
#define PUSHCOMMHANDLER_H_

#include "CommHandler_I.h"

namespace cci {
namespace rt {

class PushCommHandler: public cci::rt::CommHandler_I {
public:
	PushCommHandler(MPI_Comm const * _parent_comm, int const _gid,
			MPIDataBuffer *_buffer, Scheduler_I * _scheduler,
			cci::common::LogSession *_logsession = NULL);

	virtual ~PushCommHandler();

	virtual const char* getClassName() { return "PushCommHandler"; };

	virtual int run();

private:
	int send_count;
};

} /* namespace rt */
} /* namespace cci */
#endif /* PUSHCOMMHANDLER_H_ */
