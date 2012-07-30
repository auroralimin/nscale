/*
 * ProcessConfigurator_I.h
 *
 *  Created on: Jun 28, 2012
 *      Author: tcpan
 */

#ifndef PROCESSCONFIGURATOR_I_H_
#define PROCESSCONFIGURATOR_I_H_

#include "Communicator_I.h"
#include "mpi.h"
#include "Process.h"
#include <vector>

namespace cci {
namespace rt {

class Process;

class ProcessConfigurator_I {
public:
	ProcessConfigurator_I(cciutils::SCIOLogger *_logger) : logger(_logger) {};
	virtual ~ProcessConfigurator_I() {};

	virtual bool init() = 0;
	virtual bool finalize() = 0;

	virtual bool configure(MPI_Comm &comm, Process * proc) = 0;

protected:
	cciutils::SCIOLogger *logger;

};

} /* namespace rt */
} /* namespace cci */
#endif /* PROCESSCONFIGURATOR_I_H_ */