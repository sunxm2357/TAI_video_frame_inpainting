#include <THC.h>
#include <THCGeneral.h>

#include "SeparableConvolution_kernel.h"

extern THCState* state;

int SeparableConvolution_cuda_forward(
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* horizontal,
	THCudaTensor* output,
	int ks
) {
	SeparableConvolution_kernel_forward(
		state,
		input,
		vertical,
		horizontal,
		output,
		ks
	);

	return 1;
}


int SeparableConvolution_cuda_backward(
	THCudaTensor* grad_output,
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* horizontal,
	THCudaTensor* grad_input,
	THCudaTensor* grad_vertical,
	THCudaTensor* grad_horizontal,
	int ks
) {
	SeparableConvolution_kernel_backward(
		state,
		grad_output,
		input,
		vertical,
		horizontal,
		grad_input,
		grad_vertical,
		grad_horizontal,
		ks
	);
	
	return 1;
}
