int SeparableConvolution_cuda_forward(
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* horizontal,
	THCudaTensor* output,
	int ks
);

int SeparableConvolution_cuda_backward(
	THCudaTensor* grad_output,
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* horizontal,
	THCudaTensor* grad_input,
	THCudaTensor* grad_vertical,
	THCudaTensor* grad_horizontal,
	int ks
);
