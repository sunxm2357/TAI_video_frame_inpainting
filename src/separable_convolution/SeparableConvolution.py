import torch
import _ext.cunnex
from torch.autograd import Variable


class SeparableConvolution(torch.autograd.Function):
    def __init__(self, ks=51):
        super(SeparableConvolution, self).__init__()
    #end
    @staticmethod
    def forward(ctx, input, vertical, horizontal, ks=51):
        # save variable for backward
        ctx.save_for_backward(input, vertical, horizontal)
        ctx.constant = ks
        # get the size of input
        intBatches = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)

        # get the size of output
        intFilterSize = min(vertical.size(1), horizontal.size(1))
        intOutputHeight = min(vertical.size(2), horizontal.size(2))
        intOutputWidth = min(vertical.size(3), horizontal.size(3))

        # check the size
        assert(intInputHeight - ks == intOutputHeight - 1)
        assert(intInputWidth - ks == intOutputWidth - 1)
        assert(intFilterSize == ks)

        assert(input.is_contiguous() == True)
        assert(vertical.is_contiguous() == True)
        assert(horizontal.is_contiguous() == True)

        # allocate mem for output
        output = input.new().resize_(intBatches, intInputDepth, intOutputHeight, intOutputWidth).zero_()

        # forward, defined in cfiles
        if input.is_cuda == True:
            _ext.cunnex.SeparableConvolution_cuda_forward(
                input,
                vertical,
                horizontal,
                output,
                ks
            )

        elif input.is_cuda == False:
            raise NotImplementedError() # CPU VERSION NOT IMPLEMENTED

        # end
        return output
    # end
    @staticmethod
    def backward(ctx, grad_output):
        # get saved variables
        input, vertical, horizontal = ctx.saved_variables
        ks = ctx.constant
        # get the input size
        intBatches = input.data.size(0)
        intInputDepth = input.data.size(1)
        intInputHeight = input.data.size(2)
        intInputWidth = input.data.size(3)
        # get the output size
        intFilterSize = min(vertical.data.size(1), horizontal.data.size(1))
        intOutputHeight = min(vertical.data.size(2), horizontal.data.size(2))
        intOutputWidth = min(vertical.data.size(3), horizontal.data.size(3))
        # allocate memory for grads
        grad_input = input.data.new().resize_(intBatches, intInputDepth, intInputHeight, intInputWidth).zero_()
        grad_vertical = vertical.data.new().resize_(intBatches, intFilterSize, intOutputHeight, intOutputWidth).zero_()
        grad_horizontal = horizontal.data.new().resize_(intBatches, intFilterSize, intOutputHeight, intOutputWidth).zero_()

        # backward
        if grad_output.is_cuda == True:
            _ext.cunnex.SeparableConvolution_cuda_backward(
                grad_output.data,
                input.data,
                vertical.data,
                horizontal.data,
                grad_input,
                grad_vertical,
                grad_horizontal,
                ks
            )

        elif grad_output.is_cuda == False:
            raise NotImplementedError() # CPU VERSION NOT IMPLEMENTED

        return Variable(grad_input, volatile=True), Variable(grad_vertical, volatile=True), Variable(grad_horizontal, volatile=True), None

    # end
    # end
