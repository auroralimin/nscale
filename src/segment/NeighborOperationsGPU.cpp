/*
 * ScanlineOperations.cpp
 *
 *  Created on: Aug 2, 2011
 *      Author: tcpan
 */

#include "NeighborOperations.h"
#include "PixelOperations.h"
#include <limits>
#include "Logger.h"
#include "gpu_utils.h"


#if defined (WITH_CUDA)
#include "opencv2/gpu/stream_accessor.hpp"
#include "cuda/neighbor-ops.cuh"
#endif

namespace nscale {

using namespace cv;

namespace gpu {

using namespace cv::gpu;


#if !defined (WITH_CUDA)
template <typename T>
GpuMat NeighborOperations::border(const GpuMat& img, T background, int connectivity, Stream& stream) { throw_nogpu(); return GpuMat(); }

#else


template <typename T>
GpuMat NeighborOperations::border(const GpuMat& img, T background, int connectivity, Stream& stream) {
	// write the raw image
	CV_Assert(img.channels() == 1);
	CV_Assert(std::numeric_limits<T>::is_integer);

	// make border
    GpuMat imgFloat;
    img.convertTo(imgFloat, CV_32FC1); 
	GpuMat preInput = createContinuous(img.rows + 2, img.cols + 2, CV_32FC1);
	copyMakeBorder(imgFloat, preInput, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(background), stream);
	stream.waitForCompletion();

    GpuMat input, img2;
    preInput.convertTo(input, img.type());
    imgFloat.convertTo(img2, img.type());

    GpuMat result = createContinuous(input.size(), input.type());

    borderCaller<T>(input.rows, input.cols, input, result, background, connectivity, StreamAccessor::getStream(stream));
    stream.waitForCompletion();

    input.release();

    return result(Rect(1,1, img2.cols, img2.rows));
}

#endif

template GpuMat NeighborOperations::border<int>(const GpuMat&, int, int, Stream&);


}

}


