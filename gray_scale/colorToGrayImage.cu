#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <boost/program_options.hpp>


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
void cuColorToGray(uchar *d_out, uchar *d_in, const int width, const int height) {
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < width && row < height) {
		uchar b = d_in[row * width * 3 + col * 3];
		uchar g = d_in[row * width * 3 + col * 3 + 1];
		uchar r = d_in[row * width * 3 + col * 3 + 2];

		d_out[row * width + col] = (uchar)(r * 0.299f + g * 0.587f + b * 0.114f);
	}
}

Mat colorToGray(Mat &colorImg, size_t block_width, size_t block_height) {
	const int width = colorImg.cols, height = colorImg.rows;

	Mat grayImg = Mat::zeros(height, width, CV_8UC1);

	const int grayByte = width * height * sizeof(uchar);
	const int bgrByte = grayByte * 3;
	
	uchar *dImg, *dGrayImg;

	ERROR_HANDLE(cudaMalloc(&dImg, bgrByte));
	ERROR_HANDLE(cudaMalloc(&dGrayImg, grayByte));

	ERROR_HANDLE(cudaMemcpy(dImg, colorImg.data, bgrByte, cudaMemcpyHostToDevice));

	const dim3 block(block_width, block_height);
	const dim3 grid(width / block.x + 1, height / block.y + 1);

	cuColorToGray<<<grid, block>>>(dGrayImg, dImg, width, height);

	ERROR_HANDLE(cudaMemcpy(grayImg.data, dGrayImg, grayByte, cudaMemcpyDeviceToHost));

	ERROR_HANDLE(cudaFree(dImg));
	ERROR_HANDLE(cudaFree(dGrayImg));

	return grayImg;
}


int main(int argc, char* argv[]) {
	// arg parser
	options_description desc("all options");

	desc.add_options()
		("help,h", "produce a help screen")
		("image_path", value<string>(), "Image File Path")
		("window_size", value<int>(), "Disply Window Size")
		("block_width", value<uint>(), "CUDA Grid Block Width")
		("block_height", value<uint>(), "CUDA Grid Block Height");

	variables_map vm;
	store(parse_command_line(argc, argv, desc), vm);

	if (vm.count("help")) {
		cout << desc;
		return 1;
	}

	string filePath = "";
	Mat img;

	int window_width = 800;
	uint block_width = 16;
	uint block_height = 16;

	if (vm.count("image_path")) {
		filePath = vm["image_path"].as<string>();

		img = imread(filePath, CV_LOAD_IMAGE_COLOR);

		if (!img.data) {
			cout << "Invalid Image Path" << endl;
			return 0;
		}

		cout << "Image Load : " << filePath << endl;
	}

	if (vm.count("window_size")) {
		window_width = vm["window_size"].as<int>();
	}

	if (vm.count("block_width")) {
		block_width = vm["block_width"].as<uint>();
	}

	if (vm.count("block_height")) {
		block_height = vm["block_height"].as<uint>();
	}

	// color to gray
	Mat grayImg = colorToGray(img, block_width, block_height);

	const int width = img.cols;
	const int height = img.rows;

	const int window_height = int(height * float(window_width) / width);

	namedWindow("COLOR_IMAGE", WINDOW_NORMAL);
	namedWindow("GRAY_IMAGE", WINDOW_NORMAL);

	resizeWindow("COLOR_IMAGE", window_width, window_height) ;
	resizeWindow("GRAY_IMAGE", window_width, window_height);

	imshow("COLOR_IMAGE", img);
	imshow("GRAY_IMAGE", grayImg);

	waitKey(0);

	return 0;
}