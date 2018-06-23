#ifdef __cplusplus
	extern "C" {
#endif

void SeparableConvolution_kernel_forward(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* horizontal,
	THCudaTensor* output,
	int ks
);

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
);

#ifdef __cplusplus
	}
#endif
