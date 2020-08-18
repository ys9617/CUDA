#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/program_options.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace cv;
using namespace boost::program_options;
using namespace std;

#define ERROR_HANDLE(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void gpuAssert(cudaError_t cu_err, const char *file, int line) {
	if (cu_err != cudaSuccess) {
		cerr << "GPU Assert : "
			<< cudaGetErrorString(cu_err) << " " << file << " " << line << endl;

		exit(EXIT_FAILURE);
	}
}

__global__
void wave(uchar *ptr, const int window_size, int ticks) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < window_size && y < window_size) {
		float fx = x - window_size / 2;
		float fy = y - window_size / 2;

		float d = sqrtf(fx * fx  + fy * fy);

		ptr[x + window_size * y] = 
			(uchar)(128.f + 127.f * cos(d/10.f - ticks/7.f) / (d/10.f + 1.f));
	}
}


int main(int argc, char* argv[]) {
	options_description desc("all options");

	desc.add_options()
		("help,h", "produce a help screen")
		("window_size", value<int>(), "Disply Window Size")
		("block_width", value<uint>(), "CUDA Grid Block Width")
		("block_height", value<uint>(), "CUDA Grid Block Height")
		("time", value<uint>(), "Animation Time (sec)");

	variables_map vm;
	store(parse_command_line(argc, argv, desc), vm);

	if (vm.count("help")) {
		cout << desc;
		return 1;
	}

	int window_size = 800;
	uint block_width = 16;
	uint block_height = 16;
	uint ani_time = 10;

	if (vm.count("window_size")) {
		window_size = vm["window_size"].as<int>();
	}

	if (vm.count("block_width")) {
		block_width = vm["block_width"].as<uint>();
	}

	if (vm.count("block_height")) {
		block_height = vm["block_height"].as<uint>();
	}

	if (vm.count("time")) {
		ani_time = vm["time"].as<uint>();
	}


	Mat img = Mat::zeros(window_size, window_size, CV_8UC1);

	const int imgByte = window_size * window_size * sizeof(uchar);
	uchar *dImg;

	ERROR_HANDLE(cudaMalloc(&dImg, imgByte));

	const dim3 block(block_width, block_height);
	const dim3 grid(window_size / block.x + 1, window_size / block.y + 1);

	namedWindow("WAVE", WINDOW_NORMAL);
	resizeWindow("WAVE", window_size, window_size);

	for (uint t = 0; t < ani_time * 100; ++t) {
		wave<<<grid, block>>>(dImg, window_size, t);

		ERROR_HANDLE(cudaMemcpy(img.data, dImg, imgByte, cudaMemcpyDeviceToHost));

		imshow("WAVE", img);
		waitKey(10);

		if (t%100 == 0) {
			cout << "Elapsed time : " << t /100 << "(s)" << endl;
		}
	}

	waitKey(0);

	cudaFree(dImg);

    return 0;
}