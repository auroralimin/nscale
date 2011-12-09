/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
#include "cv.h"
#include "highgui.h"
#include <iostream>
#include <dirent.h>
#include <vector>
#include <string>
#include <errno.h>
#include <time.h>
#include "MorphologicOperations.h"
#include "PixelOperations.h"
#include "NeighborOperations.h"

#include "utils.h"
#include <stdio.h>

#include "opencv2/gpu/gpu.hpp"
#include "precomp.hpp"


using namespace cv;
using namespace cv::gpu;


int main (int argc, char **argv){

//	Mat seg_big = imread("/home/tcpan/PhD/path/Data/segmentation-tests/astroII.1/astroII.1.ndpi-0000008192-0000008192-15.mask.pbm", -1);
//	Mat seg_big = imread("/home/tcpan/PhD/path/Data/segmentation-tests/gbm2.1/gbm2.1.ndpi-0000004096-0000004096-15.mask.pbm", -1);
//	Mat seg_big = imread("/home/tcpan/PhD/path/Data/segmentation-tests/normal.3/normal.3.ndpi-0000028672-0000012288-15.mask.pbm", -1);
//	Mat seg_big = imread("/home/tcpan/PhD/path/Data/segmentation-tests/oligoastroIII.1/oligoastroIII.1.ndpi-0000053248-0000008192-15.mask.pbm", -1);
	Mat seg_big = imread("/home/tcpan/PhD/path/Data/segmentation-tests/oligoIII.1/oligoIII.1.ndpi-0000012288-0000028672-15.mask.pbm", -1);

//	Mat img = imread("/home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1/astroII.1.ndpi-0000008192-0000008192.tif", -1);
//	Mat img = imread("/home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/gbm2.1/gbm2.1.ndpi-0000004096-0000004096.tif", -1);
//	Mat img = imread("/home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/normal.3/normal.3.ndpi-0000028672-0000012288.tif", -1);
//	Mat img = imread("/home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/oligoastroIII.1/oligoastroIII.1.ndpi-0000053248-0000008192.tif", -1);
	Mat img = imread("/home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/oligoIII.1/oligoIII.1.ndpi-0000012288-0000028672.tif", -1);
	// original
	Stream stream;

	uint64_t t1, t2;

	// distance transform:  matlab code is doing this:
	// invert the image so nuclei candidates are holes
	// compute the distance (distance of nuclei pixels to background)
	// negate the distance.  so now background is still 0, but nuclei pixels have negative distances
	// set background to -inf

	// really just want the distance map.  CV computes distance to 0.
	// background is 0 in output.
	// then invert to create basins
	Mat dist(seg_big.size(), CV_32FC1);

	// opencv: compute the distance to nearest zero
	// matlab: compute the distance to the nearest non-zero
	distanceTransform(seg_big, dist, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	double mmin, mmax;
	minMaxLoc(dist, &mmin, &mmax);

	// invert and shift (make sure it's still positive)
	//dist = (mmax + 1.0) - dist;
	dist = - dist;  // appears to work better this way.

//	cciutils::cv::imwriteRaw("test/out-dist", dist);

	// then set the background to -inf and do imhmin
	//Mat distance = Mat::zeros(dist.size(), dist.type());
	// appears to work better with -inf as background
	Mat distance(dist.size(), dist.type(), -std::numeric_limits<float>::max());
	dist.copyTo(distance, seg_big);
//	cciutils::cv::imwriteRaw("test/out-distance", distance);


	// then do imhmin. (prevents small regions inside bigger regions)
	Mat distance2 = nscale::imhmin<float>(distance, 1.0f, 8);

//cciutils::cv::imwriteRaw("test/out-distanceimhmin", distance2);


	/*
	 *
		seg_big(watershed(distance2)==0) = 0;
		seg_nonoverlap = seg_big;
     *
	 */



	Mat nuclei = Mat::zeros(img.size(), img.type());
//	Mat distance3 = distance2 + (mmax + 1.0);
//	Mat dist4 = Mat::zeros(distance3.size(), distance3.type());
//	distance3.copyTo(dist4, seg_big);
//	Mat dist5(dist4.size(), CV_8U);
//	dist4.convertTo(dist5, CV_8U, (std::numeric_limits<uchar>::max() / mmax));
//	cvtColor(dist5, nuclei, CV_GRAY2BGR);
	img.copyTo(nuclei, seg_big);

	t1 = cciutils::ClockGetTime();

	// watershed in openCV requires labels.  input foreground > 0, 0 is background
	// critical to use just the nuclei and not the whole image - else get a ring surrounding the regions.
	Mat watermask = nscale::watershed2(nuclei, distance2, 8);
//	cciutils::cv::imwriteRaw("test/out-watershed", watermask);

	t2 = cciutils::ClockGetTime();
	std::cout << "cpu watershed loop took " << t2-t1 << "ms" << std::endl;

	// cpu version of watershed.
	double mn, mx;
	minMaxLoc(watermask, &mn, &mx);
	watermask = (watermask - mn) * (255.0 / (mx-mn));

	imwrite("test/out-cpu-watershed-oligoIII.1.png", watermask);
	dist.release();
	distance.release();
	watermask.release();



	// gpu version of watershed
	//Stream stream;
	GpuMat g_distance2, g_watermask, g_dummy;
	stream.enqueueUpload(distance2, g_distance2);
	stream.waitForCompletion();
	std::cout << "finished uploading" << std::endl;

	t1 = cciutils::ClockGetTime();
	g_watermask = nscale::gpu::watershedDW(g_dummy, g_distance2, 8, stream);
	stream.waitForCompletion();
	t2 = cciutils::ClockGetTime();
	std::cout << "gpu watershed DW loop took " << t2-t1 << "ms" << std::endl;
t1 = cciutils::ClockGetTime();
	GpuMat g_border = nscale::gpu::NeighborOperations::border(g_watermask, 0, stream);
	stream.waitForCompletion();
t2 = cciutils::ClockGetTime();
std::cout << "gpu border detection took " << t2 - t1 << "ms" << std::endl;

	printf("watermask size: %d %d,  type %d\n", g_watermask.rows, g_watermask.cols, g_watermask.type());
	printf("g_border size: %d %d,  type %d\n", g_border.rows, g_border.cols, g_border.type());
	Mat watermask2(g_border.size(), g_border.type());
	stream.enqueueDownload(g_border, watermask2);
	stream.waitForCompletion();
	printf("here\n");

	g_watermask.release();
	g_border.release();

//	minMaxLoc(watermask2, &mn, &mx);
//	watermask2 = (watermask2 - mn) * (255.0 / (mx-mn));

	// to show the segmentation, use modulus to separate adjacent object's values
	//watermask2 = nscale::PixelOperations::mod(watermask2, 16) * 16;

	imwrite("test/out-gpu-watershed-dw-oligoIII.1.png", watermask2);




//	t1 = cciutils::ClockGetTime();
//	g_watermask = nscale::gpu::watershedCA(g_dummy, g_distance2, 8, stream);
//	stream.waitForCompletion();
//	t2 = cciutils::ClockGetTime();
//	std::cout << "gpu watershed CA loop took " << t2-t1 << "ms" << std::endl;
//	g_watermask.download(watermask2);
//	g_watermask.release();
//
//	minMaxLoc(watermask2, &mn, &mx);
//	watermask2 = (watermask2 - mn) * (255.0 / (mx-mn));
//
//	imwrite("test/out-gpu-watershed-ca-oligoIII.1.png", watermask2);
//
//
//	// this would not work.  tested in matlab - over segment - it finds regions that are uniform.  nuclei are not uniform
//	// compute the gradient mag
//	// followed by hmin - this is input image
//	// seed if inputimage with localmin, then labelled.
//
//	Mat gray(img.size(), CV_8U);
//	cvtColor(img, gray, CV_BGR2GRAY);
//	GpuMat g_gray(img.size(), CV_8U);
//	stream.enqueueUpload(gray, g_gray);
//    stream.waitForCompletion();
//
//	Mat gray_nuclei(nuclei.size(), gray.type());
//	gray.copyTo(gray_nuclei, seg_big);
//	GpuMat g_nuclei(gray_nuclei.size(), gray_nuclei.type());
//	stream.enqueueUpload(gray_nuclei, g_nuclei);
//
//	GpuMat dx(g_nuclei.size(), CV_32F);
//	Sobel(g_nuclei, dx, CV_32F, 0, 1);
//	GpuMat dy(g_nuclei.size(), CV_32F);
//	Sobel(g_nuclei, dy, CV_32F, 1, 0);
//	GpuMat gradMag(g_nuclei.size(), CV_32F);
//	magnitude(dx, dy, gradMag, stream);
//	std::cout << "computed grad mag" << std::endl;
//	dx.release();
//	dy.release();
//
//	GpuMat hmin = nscale::gpu::imhmin(gradMag, 1.0f, 8, stream);
//	gradMag.release();
//	g_watermask = nscale::gpu::watershedCA(g_dummy, hmin, 8, stream);
//	hmin.release();
//	g_watermask.download(watermask2);
//	g_watermask.release();
//
//	minMaxLoc(watermask2, &mn, &mx);
//	watermask2 = (watermask2 - mn) * (255.0 / (mx-mn));
//
//	imwrite("test/out-gpu-watershed-gradmag-oligoIII.1.png", watermask2);
//	g_nuclei.release();
//	gray_nuclei.release();
//	gray.release();
//	g_gray.release();

	g_dummy.release();
	g_distance2.release();
	watermask2.release();

	seg_big.release();
	img.release();

	return 0;
}

