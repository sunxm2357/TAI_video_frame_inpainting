#include <THC.h>
#include <THCGeneral.h>
#include <stdio.h>

#define VEC_0(ARRAY) ((ARRAY).x)
#define VEC_1(ARRAY) ((ARRAY).y)
#define VEC_2(ARRAY) ((ARRAY).z)
#define VEC_3(ARRAY) ((ARRAY).w)

#define IDX_1(ARRAY, X)          ((ARRAY)[((X) * (ARRAY##_stride.x))])
#define IDX_2(ARRAY, X, Y)       ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y))])
#define IDX_3(ARRAY, X, Y, Z)    ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y)) + ((Z) * (ARRAY##_stride.z))])
#define IDX_4(ARRAY, X, Y, Z, W) ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y)) + ((Z) * (ARRAY##_stride.z)) + ((W) * (ARRAY##_stride.w))])

#ifdef __cplusplus
	extern "C" {
#endif

__global__ void kernel_SeparableConvolution_updateOutput(
	const int n,
	const float* input, const long4 input_size, const long4 input_stride,
	const float* vertical, const long4 vertical_size, const long4 vertical_stride,
	const float* horizontal, const long4 horizontal_size, const long4 horizontal_stride,
	float* output, const long4 output_size, const long4 output_stride,
	int ks
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

	float dblOutput = 0.0;

	int intBatch = ( intIndex / VEC_3(output_size) / VEC_2(output_size) / VEC_1(output_size) ) % VEC_0(output_size);
	int intDepth = ( intIndex / VEC_3(output_size) / VEC_2(output_size)                      ) % VEC_1(output_size);
	int intY     = ( intIndex / VEC_3(output_size)                                           ) % VEC_2(output_size);
	int intX     = ( intIndex                                                                ) % VEC_3(output_size);

	for (int intFilterY = 0; intFilterY < ks; intFilterY += 1) {
		for (int intFilterX = 0; intFilterX < ks; intFilterX += 1) {
			dblOutput += IDX_4(input, intBatch, intDepth, intY + intFilterY, intX + intFilterX) * IDX_4(vertical, intBatch, intFilterY, intY, intX) * IDX_4(horizontal, intBatch, intFilterX, intY, intX);
		}
	}

	output[intIndex] = dblOutput;
}

__global__ void kernel_SeparableConvolution_updateGradV(
	const int n,
	const float* grad_output, const long4 grad_output_size, const long4 grad_output_stride,
	const float* input, const long4 input_size, const long4 input_stride,
	const float* vertical, const long4 vertical_size, const long4 vertical_stride,
	const float* horizontal, const long4 horizontal_size, const long4 horizontal_stride,
	float* grad_vertical, const long4 grad_vertical_size, const long4 grad_vertical_stride,
	float* grad_horizontal, const long4 grad_horizontal_size, const long4 grad_horizontal_stride,
	int ks
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

	int intBatch = ( intIndex / VEC_3(grad_vertical_size) / VEC_2(grad_vertical_size) / VEC_1(grad_vertical_size) 	) % VEC_0(grad_vertical_size);
	int intDepth = ( intIndex / VEC_3(grad_vertical_size) / VEC_2(grad_vertical_size)                      			) % VEC_1(grad_vertical_size);
	int intY     = ( intIndex / VEC_3(grad_vertical_size)                                           				) % VEC_2(grad_vertical_size);
	int intX     = ( intIndex                                                                						) % VEC_3(grad_vertical_size);

	float dblGradVertical = 0.0;

	//if (intDepth == 11){
        //printf("grad_vertical[%d,%d,%d,%d]: \n", intBatch, intDepth, intY, intX);
        //}

	for (int depth = 0; depth < VEC_1(grad_output_size) ; depth +=1){
		for (int intFilter = 0; intFilter < ks; intFilter += 1) {
		    //if (intDepth ==11){
		      //  printf("    grad_output[%d,%d,%d,%d] * input[[%d,%d,%d,%d] * horizontal[%d,%d,%d,%d](%f)\n", intBatch, depth, intY, intX, intBatch, depth, intY + intDepth, intX + intFilter, intBatch, intFilter, intY, intX, IDX_4(horizontal, intBatch, intFilter, intY, intX));
		    //}
			dblGradVertical += IDX_4(grad_output, intBatch, depth, intY, intX) * IDX_4(input, intBatch, depth, intY + intDepth, intX + intFilter) * IDX_4(horizontal, intBatch, intFilter, intY, intX);
		}
	}

	grad_vertical[intIndex] = dblGradVertical;
}

__global__ void kernel_SeparableConvolution_updateGradH(
	const int n,
	const float* grad_output, const long4 grad_output_size, const long4 grad_output_stride,
	const float* input, const long4 input_size, const long4 input_stride,
	const float* vertical, const long4 vertical_size, const long4 vertical_stride,
	const float* horizontal, const long4 horizontal_size, const long4 horizontal_stride,
	float* grad_vertical, const long4 grad_vertical_size, const long4 grad_vertical_stride,
	float* grad_horizontal, const long4 grad_horizontal_size, const long4 grad_horizontal_stride,
	int ks
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

	int intBatch = ( intIndex / VEC_3(grad_horizontal_size) / VEC_2(grad_horizontal_size) / VEC_1(grad_horizontal_size) 	) % VEC_0(grad_horizontal_size);
	int intDepth = ( intIndex / VEC_3(grad_horizontal_size) / VEC_2(grad_horizontal_size)                      			) % VEC_1(grad_horizontal_size);
	int intY     = ( intIndex / VEC_3(grad_horizontal_size)                                           				) % VEC_2(grad_horizontal_size);
	int intX     = ( intIndex                                                                						) % VEC_3(grad_horizontal_size);

	float dblGradHorizontal = 0.0;

	for (int depth = 0; depth < VEC_1(grad_output_size) ; depth +=1){
		for (int intFilter = 0; intFilter < ks; intFilter += 1) {
			dblGradHorizontal += IDX_4(grad_output, intBatch, depth, intY, intX) * IDX_4(input, intBatch, depth, intY + intFilter, intX + intDepth) * IDX_4(vertical, intBatch, intFilter, intY, intX);
		}
	}

	grad_horizontal[intIndex] = dblGradHorizontal;
}

__global__ void kernel_SeparableConvolution_updateGradI(
	const int n,
	const float* grad_output, const long4 grad_output_size, const long4 grad_output_stride,
	const float* input, const long4 input_size, const long4 input_stride,
	const float* vertical, const long4 vertical_size, const long4 vertical_stride,
	const float* horizontal, const long4 horizontal_size, const long4 horizontal_stride,
	float* grad_input, const long4 grad_input_size, const long4 grad_input_stride,
	int ks
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

	int intBatch = ( intIndex / VEC_3(grad_input_size) / VEC_2(grad_input_size) / VEC_1(grad_input_size) 	) % VEC_0(grad_input_size);
	int intDepth = ( intIndex / VEC_3(grad_input_size) / VEC_2(grad_input_size)                      		) % VEC_1(grad_input_size);
	int intY     = ( intIndex / VEC_3(grad_input_size)                                           			) % VEC_2(grad_input_size);
	int intX     = ( intIndex                                                                				) % VEC_3(grad_input_size);

	float dblGradInput = 0.0;

	//if (intY == 22 && intX == 35){
      //  printf("grad_input[%d,%d,%d,%d]: \n", intBatch, intDepth, intY, intX);
      //  }

	for (int intFilterX = 0; intFilterX < ks; intFilterX +=1){
		for (int intFilterY = 0; intFilterY < ks; intFilterY += 1) {
			int X = intX - (ks-1) + intFilterX;
			int Y = intY - (ks-1) + intFilterY;
			if ((X < 0) || (Y < 0) || (Y >= VEC_2(grad_output_size)) || (X >= VEC_3(grad_output_size))){
				continue;
			} else {
			    //if (intY == 22 && intX == 35){
			      //  printf("    grad_output[%d,%d,%d,%d](%f) * vertical[%d,%d,%d,%d](%f) * horizontal[%d,%d,%d,%d](%f) \n", intBatch, intDepth, Y, X, IDX_4(grad_output, intBatch, intDepth, Y, X), intBatch, (51-1) - intFilterY, Y, X, IDX_4(vertical, intBatch, (51-1) - intFilterY, Y, X), intBatch, (51-1) - intFilterX, Y, X, IDX_4(horizontal, intBatch, (51-1) - intFilterX, Y, X));
			    //}
				dblGradInput += IDX_4(grad_output, intBatch, intDepth, Y, X) * IDX_4(vertical, intBatch, (ks-1) - intFilterY, Y, X) * IDX_4(horizontal, intBatch, (ks-1) - intFilterX, Y, X);
			}
		}
	}

	grad_input[intIndex] = dblGradInput;
}

void SeparableConvolution_kernel_forward(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* horizontal,
	THCudaTensor* output,
	int ks
) {
	int n = 0;

	n = THCudaTensor_nElement(state, output);
	kernel_SeparableConvolution_updateOutput<<< (n + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		n,
		THCudaTensor_data(state, input), make_long4(input->size[0], input->size[1], input->size[2], input->size[3]), make_long4(input->stride[0], input->stride[1], input->stride[2], input->stride[3]),
		THCudaTensor_data(state, vertical), make_long4(vertical->size[0], vertical->size[1], vertical->size[2], vertical->size[3]), make_long4(vertical->stride[0], vertical->stride[1], vertical->stride[2], vertical->stride[3]),
		THCudaTensor_data(state, horizontal), make_long4(horizontal->size[0], horizontal->size[1], horizontal->size[2], horizontal->size[3]), make_long4(horizontal->stride[0], horizontal->stride[1], horizontal->stride[2], horizontal->stride[3]),
		THCudaTensor_data(state, output), make_long4(output->size[0], output->size[1], output->size[2], output->size[3]), make_long4(output->stride[0], output->stride[1], output->stride[2], output->stride[3]),
		ks
	);

	THCudaCheck(cudaGetLastError());
}

void SeparableConvolution_kernel_backward(
	THCState* state,
	THCudaTensor* grad_output,
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* horizontal,
	THCudaTensor* grad_input,
	THCudaTensor* grad_vertical,
	THCudaTensor* grad_horizontal,
	int ks
) {
	int n = 0;

	n = THCudaTensor_nElement(state, grad_vertical);
	kernel_SeparableConvolution_updateGradV<<< (n + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		n,
		THCudaTensor_data(state, grad_output), make_long4(grad_output->size[0], grad_output->size[1], grad_output->size[2], grad_output->size[3]), make_long4(grad_output->stride[0], grad_output->stride[1], grad_output->stride[2], grad_output->stride[3]),
		THCudaTensor_data(state, input), make_long4(input->size[0], input->size[1], input->size[2], input->size[3]), make_long4(input->stride[0], input->stride[1], input->stride[2], input->stride[3]),
		THCudaTensor_data(state, vertical), make_long4(vertical->size[0], vertical->size[1], vertical->size[2], vertical->size[3]), make_long4(vertical->stride[0], vertical->stride[1], vertical->stride[2], vertical->stride[3]),
		THCudaTensor_data(state, horizontal), make_long4(horizontal->size[0], horizontal->size[1], horizontal->size[2], horizontal->size[3]), make_long4(horizontal->stride[0], horizontal->stride[1], horizontal->stride[2], horizontal->stride[3]),
		THCudaTensor_data(state, grad_vertical), make_long4(grad_vertical->size[0], grad_vertical->size[1], grad_vertical->size[2], grad_vertical->size[3]), make_long4(grad_vertical->stride[0], grad_vertical->stride[1], grad_vertical->stride[2], grad_vertical->stride[3]),
		THCudaTensor_data(state, grad_horizontal), make_long4(grad_horizontal->size[0], grad_horizontal->size[1], grad_horizontal->size[2], grad_horizontal->size[3]), make_long4(grad_horizontal->stride[0], grad_horizontal->stride[1], grad_horizontal->stride[2], grad_horizontal->stride[3]),
		ks
	);

	THCudaCheck(cudaGetLastError());

	n = THCudaTensor_nElement(state, grad_horizontal);
	kernel_SeparableConvolution_updateGradH<<< (n + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		n,
		THCudaTensor_data(state, grad_output), make_long4(grad_output->size[0], grad_output->size[1], grad_output->size[2], grad_output->size[3]), make_long4(grad_output->stride[0], grad_output->stride[1], grad_output->stride[2], grad_output->stride[3]),
		THCudaTensor_data(state, input), make_long4(input->size[0], input->size[1], input->size[2], input->size[3]), make_long4(input->stride[0], input->stride[1], input->stride[2], input->stride[3]),
		THCudaTensor_data(state, vertical), make_long4(vertical->size[0], vertical->size[1], vertical->size[2], vertical->size[3]), make_long4(vertical->stride[0], vertical->stride[1], vertical->stride[2], vertical->stride[3]),
		THCudaTensor_data(state, horizontal), make_long4(horizontal->size[0], horizontal->size[1], horizontal->size[2], horizontal->size[3]), make_long4(horizontal->stride[0], horizontal->stride[1], horizontal->stride[2], horizontal->stride[3]),
		THCudaTensor_data(state, grad_vertical), make_long4(grad_vertical->size[0], grad_vertical->size[1], grad_vertical->size[2], grad_vertical->size[3]), make_long4(grad_vertical->stride[0], grad_vertical->stride[1], grad_vertical->stride[2], grad_vertical->stride[3]),
		THCudaTensor_data(state, grad_horizontal), make_long4(grad_horizontal->size[0], grad_horizontal->size[1], grad_horizontal->size[2], grad_horizontal->size[3]), make_long4(grad_horizontal->stride[0], grad_horizontal->stride[1], grad_horizontal->stride[2], grad_horizontal->stride[3]),
		ks
	);

	THCudaCheck(cudaGetLastError());

	int m = 0;

	m = THCudaTensor_nElement(state, grad_input);
	kernel_SeparableConvolution_updateGradI<<< (m + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		m,
		THCudaTensor_data(state, grad_output), make_long4(grad_output->size[0], grad_output->size[1], grad_output->size[2], grad_output->size[3]), make_long4(grad_output->stride[0], grad_output->stride[1], grad_output->stride[2], grad_output->stride[3]),
		THCudaTensor_data(state, input), make_long4(input->size[0], input->size[1], input->size[2], input->size[3]), make_long4(input->stride[0], input->stride[1], input->stride[2], input->stride[3]),
		THCudaTensor_data(state, vertical), make_long4(vertical->size[0], vertical->size[1], vertical->size[2], vertical->size[3]), make_long4(vertical->stride[0], vertical->stride[1], vertical->stride[2], vertical->stride[3]),
		THCudaTensor_data(state, horizontal), make_long4(horizontal->size[0], horizontal->size[1], horizontal->size[2], horizontal->size[3]), make_long4(horizontal->stride[0], horizontal->stride[1], horizontal->stride[2], horizontal->stride[3]),
		THCudaTensor_data(state, grad_input), make_long4(grad_input->size[0], grad_input->size[1], grad_input->size[2], grad_input->size[3]), make_long4(grad_input->stride[0], grad_input->stride[1], grad_input->stride[2], grad_input->stride[3]),
		ks
	);

	THCudaCheck(cudaGetLastError());
}

#ifdef __cplusplus
	}
#endif
