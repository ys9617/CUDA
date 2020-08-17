#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/program_options.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define ERROR_HANDLE(ans) { gpuAssert((ans), __FILE__, __LINE__); }


using namespace cv;
using namespace boost::program_options;
using namespace std;


void gpuAssert(cudaError_t cu_err, const char *file, int line) {
	if (cu_err != cudaSuccess) {
		cerr << "GPU Assert : "
			<< cudaGetErrorString(cu_err) << " " << file << " " << line << endl;

		exit(EXIT_FAILURE);
	}
}

struct cuComplex {
	float r, i;

	__device__ cuComplex(float a, float b) : r{a}, i(b) {}

	__device__ float magnitude2() { return r * r + i * i; }
	
	__device__ cuComplex operator*(const cuComplex& a) {
		return cuComplex(r*a.r - i * a.i, i*a.r + r * a.i);
	}

	__device__ cuComplex operator+(const cuComplex& a) {
		return cuComplex(r + a.r, i + a.i);
	}
};

__device__
int julia(int x, int y, int wnd_size) {
	const float scale = 1.5;

	float jx = scale * (float)(wnd_size / 2 - x) / (wnd_size / 2);
	float jy = scale * (float)(wnd_size / 2 - y) / (wnd_size / 2);

	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	for (int i = 0; i < 200; ++i) {
		a = a * a + c;

		if (a.magnitude2() > 1000) {
			return 0;
		}
	}

	return 1;
}

__global__
void juliaSet(uchar *ptr, const int window_size) {
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < window_size && row < window_size) {
		ptr[row * window_size + col] = 255 * julia(col, row, window_size);
	}
}


int main(int argc, char* argv[]) {
	options_description desc("all options");

	desc.add_options()
		("help,h", "produce a help screen")
		("window_size", value<int>(), "Disply Window Size")
		("block_width", value<uint>(), "CUDA Grid Block Width")
		("block_height", value<uint>(), "CUDA Grid Block Height");

	variables_map vm;
	store(parse_command_line(argc, argv, desc), vm);

	if (vm.count("help")) {
		cout << desc;
		return 1;
	}

	int window_size = 1000;
	uint block_width = 16;
	uint block_height = 16;

	if (vm.count("window_size")) {
		window_size = vm["window_size"].as<int>();
	}

	if (vm.count("block_width")) {
		block_width = vm["block_width"].as<uint>();
	}

	if (vm.count("block_height")) {
		block_height = vm["block_height"].as<uint>();
	}

	Mat img = Mat::zeros(window_size, window_size, CV_8UC1);

	const int imgByte = window_size * window_size * sizeof(uchar);
	uchar *dImg;

	ERROR_HANDLE(cudaMalloc(&dImg, imgByte));

	const dim3 block(block_width, block_height);
	const dim3 grid(window_size / block.x + 1, window_size / block.y + 1);

	juliaSet<<<grid, block>>>(dImg, window_size);

	ERROR_HANDLE(cudaMemcpy(img.data, dImg, imgByte, cudaMemcpyDeviceToHost));

	namedWindow("JULIA SET", WINDOW_NORMAL);
	resizeWindow("JULIA SET", window_size, window_size);
	imshow("JULIA SET", img);

	waitKey(0);

    return 0;
}