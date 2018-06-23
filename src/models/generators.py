import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from separable_convolution.SeparableConvolution import SeparableConvolution
from util.util import inverse_transform, bgr2gray


##################################################
#############  SPECIAL LOSS MODULES  #############
##################################################

class GDL(nn.Module):
    """Image gradient difference loss as defined by Mathieu et al. (https://arxiv.org/abs/1511.05440)."""

    def __init__(self, c_dim):
        """Constructor

        :param c_dim: The number of image channels (e.g. 3 for RGB)
        """
        super(GDL, self).__init__()
        self.loss = nn.L1Loss()
        self.filter_w = np.zeros([c_dim, c_dim, 1, 2])
        self.filter_h = np.zeros([c_dim, c_dim, 2, 1])
        for i in range(c_dim):
            self.filter_w[i, i, :, :] = np.array([[-1, 1]])
            self.filter_h[i, i, :, :] = np.array([[1], [-1]])

    def forward(self, output, target):
        """Forward method

        :param output: The predicted output
        :param target: The desired output
        """
        filter_w = Variable(torch.from_numpy(self.filter_w).float().cuda())
        filter_h = Variable(torch.from_numpy(self.filter_h).float().cuda())
        output_w = F.conv2d(output, filter_w, padding=(0, 1))
        output_h = F.conv2d(output, filter_h, padding=(1, 0))
        target_w = F.conv2d(target, filter_w, padding=(0, 1))
        target_h = F.conv2d(target, filter_h, padding=(1, 0))
        return self.loss(output_w, target_w) + self.loss(output_h, target_h)


##################################################
##############  MC-NET PRIMITIVES  ###############
##################################################


class MotionEnc(nn.Module):
    """The motion encoder as defined by Villegas et al. (https://arxiv.org/abs/1706.08033).

    This module takes a difference frame and produces an encoded representation with reduced resolution. It also
    produces the intermediate convolutional activations for use with residual layers.
    """

    def __init__(self, gf_dim):
        """Constructor

        :param gf_dim: The number of filters in the first layer
        """
        super(MotionEnc, self).__init__()

        conv1 = nn.Conv2d(1, gf_dim, 5, padding=2)
        relu1 = nn.ReLU()
        self.dyn_conv1 = nn.Sequential(conv1, relu1)

        pool1 = nn.MaxPool2d(2)
        conv2 = nn.Conv2d(gf_dim, gf_dim * 2, 5, padding=2)
        relu2 = nn.ReLU()
        self.dyn_conv2 = nn.Sequential(pool1, conv2, relu2)

        pool2 = nn.MaxPool2d(2)
        conv3 = nn.Conv2d(gf_dim * 2, gf_dim * 4, 7, padding=3)
        relu3 = nn.ReLU()
        self.dyn_conv3 = nn.Sequential(pool2, conv3, relu3)

        self.pool3 = nn.MaxPool2d(2)

    def forward(self, input_diff):
        """Forward method

        :param input_diff: A difference frame [batch_size, 1, h, w]
        :return: [batch_size, gf_dim*4, h/8, w/8]
        """
        res_in = []
        res_in.append(self.dyn_conv1(input_diff))
        res_in.append(self.dyn_conv2(res_in[-1]))
        res_in.append(self.dyn_conv3(res_in[-1]))
        output = self.pool3(res_in[-1])
        return output, res_in


class ContentEnc(nn.Module):
    """The motion encoder as defined by Villegas et al. (https://arxiv.org/abs/1706.08033).

    This module takes a standard frame and produces an encoded representation with reduced resolution. It also
    produces the intermediate convolutional activations for use with residual layers.
    """

    def __init__(self, c_dim, gf_dim):
        """Constructor

        :param c_dim: The number of image channels (e.g. 3 for RGB)
        :param gf_dim: The number of filters in the first layer
        """

        super(ContentEnc, self).__init__()

        conv1_1 = nn.Conv2d(c_dim, gf_dim, 3, padding=1)
        relu1_1 = nn.ReLU()
        conv1_2 = nn.Conv2d(gf_dim, gf_dim, 3, padding=1)
        relu1_2 = nn.ReLU()
        self.cont_conv1 = nn.Sequential(conv1_1, relu1_1, conv1_2, relu1_2)

        pool1 = nn.MaxPool2d(2)
        conv2_1 = nn.Conv2d(gf_dim, gf_dim * 2, 3, padding=1)
        relu2_1 = nn.ReLU()
        conv2_2 = nn.Conv2d(gf_dim * 2, gf_dim * 2, 3, padding=1)
        relu2_2 = nn.ReLU()
        self.cont_conv2 = nn.Sequential(pool1, conv2_1, relu2_1, conv2_2, relu2_2)

        pool2 = nn.MaxPool2d(2)
        conv3_1 = nn.Conv2d(gf_dim * 2, gf_dim * 4, 3, padding=1)
        relu3_1 = nn.ReLU()
        conv3_2 = nn.Conv2d(gf_dim * 4, gf_dim * 4, 3, padding=1)
        relu3_2 = nn.ReLU()
        conv3_3 = nn.Conv2d(gf_dim * 4, gf_dim * 4, 3, padding=1)
        relu3_3 = nn.ReLU()

        self.cont_conv3 = nn.Sequential(pool2, conv3_1, relu3_1, conv3_2, relu3_2, conv3_3, relu3_3)

        self.pool3 = nn.MaxPool2d(2)

    def forward(self, raw):
        """Forward method

        :param raw: A raw image frame [batch_size, c_dim, h, w]
        :return: [batch_size, gf_dim*4, h/8, w/8]
        """
        res_in = []
        res_in.append(self.cont_conv1(raw))
        res_in.append(self.cont_conv2(res_in[-1]))
        res_in.append(self.cont_conv3(res_in[-1]))
        output = self.pool3(res_in[-1])
        return output, res_in


class CombLayers(nn.Module):
    """The combination layers as defined by Villegas et al. (https://arxiv.org/abs/1706.08033).

    This module combines the encoded representations of the past motion frames and the last content frame with
    convolutional layers.
    """

    def __init__(self, gf_dim):
        """Constructor

        :param gf_dim: The number of filters in the first layer
        """

        super(CombLayers, self).__init__()

        conv1 = nn.Conv2d(gf_dim * 8, gf_dim * 4, 3, padding=1)
        relu1 = nn.ReLU()
        conv2 = nn.Conv2d(gf_dim * 4, gf_dim * 2, 3, padding=1)
        relu2 = nn.ReLU()
        conv3 = nn.Conv2d(gf_dim * 2, gf_dim * 4, 3, padding=1)
        relu3 = nn.ReLU()
        self.h_comb = nn.Sequential(conv1, relu1, conv2, relu2, conv3, relu3)

    def forward(self, h_dyn, h_cont):
        """Forward method

        :param h_dyn: The output from the MotionEnc module
        :param h_cont: The output from the ContentEnc module
        """
        input = torch.cat((h_dyn, h_cont), dim=1)
        return self.h_comb(input)


class Residual(nn.Module):
    """The residual layers as defined by Villegas et al. (https://arxiv.org/abs/1706.08033).

    This module combines a pair of "residual" convolutional activations from the MotionEnc and ContentEnc modules with
    convolutional layers.
    """

    def __init__(self, in_dim, out_dim):
        """Constructor

        :param in_dim: The number of channels in the input
        :param out_dim: The number of channels in the output
        """

        super(Residual, self).__init__()

        conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        relu1 = nn.ReLU()
        conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.res = nn.Sequential(conv1, relu1, conv2)

    def forward(self, input_dyn, input_cont):
        """Forward method

        :param input_dyn: A set of intermediate activations from the MotionEnc module
        :param input_cont: A set of intermediate activations from the ContentEnc module
        """
        input = torch.cat((input_dyn, input_cont), dim=1)
        return self.res(input)


class DecCnn(nn.Module):
    """The decoder layers as defined by Villegas et al. (https://arxiv.org/abs/1706.08033).

    This module decodes the output of the CombLayers module into a full-resolution image. Optionally, it can incorporate
    activations from the Residual modules to help preserve spatial information.
    """

    def __init__(self, c_dim, gf_dim):
        """Constructor

        :param c_dim: The number of image channels (e.g. 3 for RGB)
        :param gf_dim: The number of filters in the first layer
        """

        super(DecCnn, self).__init__()

        deconv3_3 = nn.ConvTranspose2d(gf_dim * 4, gf_dim * 4, 3, padding=1)
        relu3_3 = nn.ReLU()
        deconv3_2 = nn.ConvTranspose2d(gf_dim * 4, gf_dim * 4, 3, padding=1)
        relu3_2 = nn.ReLU()
        deconv3_1 = nn.ConvTranspose2d(gf_dim * 4, gf_dim * 2, 3, padding=1)
        relu3_1 = nn.ReLU()
        self.dec3 = nn.Sequential(deconv3_3, relu3_3, deconv3_2, relu3_2, deconv3_1, relu3_1)

        deconv2_2 = nn.ConvTranspose2d(gf_dim * 2, gf_dim * 2, 3, padding=1)
        relu2_2 = nn.ReLU()
        deconv2_1 = nn.ConvTranspose2d(gf_dim * 2, gf_dim, 3, padding=1)
        relu2_1 = nn.ReLU()
        self.dec2 = nn.Sequential(deconv2_2, relu2_2, deconv2_1, relu2_1)

        deconv1_2 = nn.ConvTranspose2d(gf_dim, gf_dim, 3, padding=1)
        relu1_2 = nn.ReLU()
        deconv1_1 = nn.ConvTranspose2d(gf_dim, c_dim, 3, padding=1)
        tanh1_1 = nn.Tanh()
        self.dec1 = nn.Sequential(deconv1_2, relu1_2, deconv1_1, tanh1_1)

    def forward(self, comb, res1=None, res2=None, res3=None):
        """Forward method

        :param comb: The output from the CombLayers module
        :param (res1, res2, res3): Outputs from each Residual module
        """

        # Check that either all or no residual activations are given
        if res1 is not None:
            if (res2 is None) or (res3 is None):
                raise ValueError('3 residual networks should not be None at the same time')
            else:
                enable_res = True
        else:
            if (res2 is None) and (res3 is None):
                enable_res = False
            else:
                raise ValueError('3 residual networks should be None at the same time')

        input3 = self.fixed_unpooling(comb)
        if enable_res:
            input3 += res3
        dec3_out = self.dec3(input3)

        input2 = self.fixed_unpooling(dec3_out)
        if enable_res:
            input2 += res2
        dec2_out = self.dec2(input2)

        input1 = self.fixed_unpooling(dec2_out)
        if enable_res:
            input1 += res1
        dec1_out = self.dec1(input1)

        return dec1_out

    def fixed_unpooling(self, x):
        """Unpools by spreading the values of x across a spaced-out grid. E.g.:

               x0x0x0
        xxx    000000
        xxx -> x0x0x0
        xxx    000000
               x0x0x0
               000000

        :param input: B x C x H x W FloatTensor Variable
        :return:
        """
        x = x.permute(0, 2, 3, 1)
        out = torch.cat((x, x.clone().zero_()), dim=3)
        out = torch.cat((out, out.clone().zero_()), dim=2)
        return out.view(x.size(0), 2*x.size(1), 2*x.size(2), x.size(3)).permute(0, 3, 1, 2)


class ConvLstmCell(nn.Module):
    """A convolutional LSTM cell (https://arxiv.org/abs/1506.04214)."""

    def __init__(self, feature_size, num_features, forget_bias=1, activation=F.tanh, bias=True):
        """Constructor

        :param feature_size: The kernel size of the convolutional layer
        :param num_features: Controls the number of input/output features of cell
        :param forget_bias: The bias for the forget gate
        :param activation: The activation function to use in the gates
        :param bias: Whether to use a bias for the convolutional layer
        """
        super(ConvLstmCell, self).__init__()

        self.feature_size = feature_size
        self.num_features = num_features
        self.forget_bias = forget_bias
        self.activation = activation

        self.conv = nn.Conv2d(num_features * 2, num_features * 4, feature_size, padding=(feature_size - 1) / 2,
                              bias=bias)

    def forward(self, input, state):
        """Forward method

        :param input: The current input to the ConvLSTM
        :param state: The previous state of the ConvLSTM (the concatenated memory cell and hidden state)
        """
        c, h = torch.chunk(state, 2, dim=1)
        conv_input = torch.cat((input, h) , dim=1)
        conv_output = self.conv(conv_input)
        (i, j, f, o) = torch.chunk(conv_output, 4, dim=1)
        new_c = c * F.sigmoid(f + self.forget_bias) + F.sigmoid(i) * self.activation(j)
        new_h = self.activation(new_c) * F.sigmoid(o)
        new_state = torch.cat((new_c, new_h), dim=1)
        return new_h, new_state


##################################################
####################  MC-NET  ####################
##################################################


class Generator(nn.Module):
    """The MC-Net video prediction network as defined by Villegas et al. (https://arxiv.org/abs/1706.08033)."""

    def __init__(self, gf_dim, c_dim, feature_size, forget_bias=1, activation=F.tanh, bias=True):
        """Constructor

        :param gf_dim: The number of filters to use in the MotionEnc and ContentEnc modules
        :param c_dim: The number of image channels (e.g. 3 for RGB)
        :param feature_size: The kernel size of the ConvLSTM
        :param forget_bias: The forget bias to use in the ConvLSTM
        :param activation: The activation function to use in the ConvLSTM
        :param bias: Whether to use a bias for the ConvLSTM
        """
        super(Generator, self).__init__()
        self.c_dim = c_dim

        self.motion_enc = MotionEnc(gf_dim)
        self.convLstm_cell = ConvLstmCell(feature_size, 4 * gf_dim, forget_bias=forget_bias, activation=activation,
                                          bias=bias)
        self.content_enc = ContentEnc(c_dim, gf_dim)
        self.comb_layers = CombLayers(gf_dim)
        self.residual3 = Residual(gf_dim * 8, gf_dim * 4)
        self.residual2 = Residual(gf_dim * 4, gf_dim * 2)
        self.residual1 = Residual(gf_dim * 2, gf_dim * 1)
        self.dec_cnn = DecCnn(c_dim, gf_dim)

    def forward(self, K, T, state, image_size, diff_in, xt):
        """Forward method

        :param K: The number of past time steps
        :param T: The number of future time steps
        :param state: The initial state of the ConvLSTM
        :param image_size: The resolution of the input video
        :param diff_in: The past difference frames
        :param xt: The last past frame
        """

        # Compute the motion encoding at each past time step
        for t in range(K-1):
            enc_h, res_m = self.motion_enc.forward(diff_in[t])
            h_dyn, state = self.convLstm_cell.forward(enc_h, state)

        # Keep track of outputs
        pred = []
        for t in range(T):
            # Compute the representation of the next motion frame
            if t > 0:
                enc_h, res_m = self.motion_enc.forward(diff_in[-1])
                h_dyn, state = self.convLstm_cell.forward(enc_h, state)
            # Compute the representation of the next content frame
            h_cont, res_c = self.content_enc.forward(xt)
            # Combine the motion and content encodings
            h_tpl = self.comb_layers.forward(h_dyn, h_cont)
            # Pass intermediate activations through the residual layers
            res_1 = self.residual1.forward(res_m[0], res_c[0])
            res_2 = self.residual2.forward(res_m[1], res_c[1])
            res_3 = self.residual3.forward(res_m[2], res_c[2])
            # Pass activations through the decoder
            x_hat = self.dec_cnn.forward(h_tpl, res_1, res_2, res_3)

            # Obtain grayscale versions of the predicted frames
            if self.c_dim == 3:
                x_hat_gray = bgr2gray(inverse_transform(x_hat))
                xt_gray = bgr2gray(inverse_transform(xt))
            else:
                x_hat_gray = inverse_transform(x_hat)
                xt_gray = inverse_transform(xt)

            # Compute the next MotionEnc input, which is difference between grayscale frames
            diff_in.append(x_hat_gray - xt_gray)
            # Update last past frame
            xt = x_hat
            # Update outputs
            pred.append(x_hat.view(-1, self.c_dim, image_size[0], image_size[1]))

        return pred


##################################################
######  SEPARABLE ADAPTIVE KERNEL NETWORKS  ######
##################################################


def create_basic_conv_block(num_layers, num_in_channels, num_out_channels):
    """Create a sequence of resolution-preserving convolutional layers.

    Used for the encoder and decoder in the adaptive separable convolution network.

    :param num_layers: The number of convolutional layers in this block
    :param num_in_channels: The number of channels in the input
    :param num_out_channels: The number of channels in the output
    """
    sequence = []
    for i in xrange(num_layers):
        if i == 0:
            sequence.append(torch.nn.Conv2d(in_channels=num_in_channels, out_channels=num_out_channels, kernel_size=3,
                                            stride=1, padding=1))
        else:
            sequence.append(torch.nn.Conv2d(in_channels=num_out_channels, out_channels=num_out_channels, kernel_size=3,
                                            stride=1, padding=1))
        sequence.append(torch.nn.ReLU(inplace=False))

    return torch.nn.Sequential(*sequence)


def create_1d_kernel_generator_block(num_layers, kf_dim, ks):
    """Create a sequence of convolutional blocks that result in a Tensor where each column corresponds to the weights of
     a 1D kernel.

    :param num_layers: The number of intermediate convolutional layers
    :param kf_dim: A number that controls the number of features per layer
    :param ks: The size of the 1D kernel
    """

    sequence = []
    for i in range(num_layers):
        if i == num_layers - 1:
            sequence.append(torch.nn.Conv2d(in_channels=kf_dim*2, out_channels=ks, kernel_size=3, stride=1, padding=1))
        else:
            sequence.append(torch.nn.Conv2d(in_channels=kf_dim*2, out_channels=kf_dim*2, kernel_size=3, stride=1,
                                            padding=1))
        sequence.append(torch.nn.ReLU(inplace=False))
    sequence.append(torch.nn.Upsample(scale_factor=2, mode='bilinear'))
    sequence.append(torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1))

    return torch.nn.Sequential(*sequence)


def create_encoder_blocks(start_i, end_i, layers, if_dim, kf_dim):
    """Create a chain of (end_i - start_i) encoder blocks.

    :param start_i: The starting index of the chain
    :param end_i: The ending index of the chain
    :param layers: The number of layers to use in each block
    :param if_dim: The number of channels in the chain's input
    :param kf_dim: Controls the number of filters in each block
    :return: A list of convolutional blocks and a list of pooling layers in the chain
    """

    moduleConv = []
    modulePool = []

    for i in xrange(start_i, end_i):
        if i == start_i:
            moduleConv.append(create_basic_conv_block(layers, if_dim, kf_dim * (2 ** i)))
        else:
            moduleConv.append(create_basic_conv_block(layers, kf_dim * (2 ** (i - 1)), kf_dim * (2 ** i)))
        modulePool.append(torch.nn.AvgPool2d(kernel_size=2, stride=2))

    return moduleConv, modulePool


def create_decoder_blocks(num_block, kf_dim, layers, rc_loc):
    """Create a chain of decoder blocks.

    :param num_block: The number of decoder blocks in this chain
    :param kf_dim: Controls the number of filters in each block
    :param layers: The number of layers to use in each block
    :param rc_loc: The index of the block to inject temporal information into
    :return: A list of convolutional blocks and a list of upsampling layers in the chain
    """

    moduleDeconv = []
    moduleUpsample = []
    for i in range(num_block):
        eff_in = 2 ** (num_block - i + 1)
        eff_out = 2 ** (num_block - i)
        if i == 0:
            c_in = kf_dim * eff_out
            c_out = kf_dim * eff_out
        else:
            c_in = kf_dim * eff_in
            c_out = kf_dim * eff_out
        moduleDeconv.append(create_basic_conv_block(layers, c_in, c_out))
        if i == rc_loc - 1:
            moduleUpsample.append(torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear'),
                torch.nn.Conv2d(in_channels=c_out+1, out_channels=c_out, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False))
            )
        else:
            moduleUpsample.append(torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear'),
                torch.nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False))
            )

    return moduleDeconv, moduleUpsample


class KernelNet(nn.Module):
    """An extended version of the adaptive separable convolution network for video frame interpolation as proposed by
    Niklaus et al. (https://arxiv.org/abs/1708.01692).

    This module can optionally take in a "ratio" argument in the forward pass, where the ratio represents which time
    step is being generated.
    """

    def __init__(self, c_dim, ks, num_block=5, layers=3, kf_dim=32, rc_loc=-1):
        """Constructor

        :param c_dim: The number of image channels (e.g. 3 for RGB)
        :param ks: The size of the 1D kernel
        :param num_block: Controls the number of blocks to use in the encoder and decoder chains
        :param layers: The number of layers to use in each encoder and decoder block
        :param kf_dim: Controls the number of filters in each encoder and decoder block
        :param rc_loc: The index of the block to inject temporal information into. If -1, do not inject time information
        """

        super(KernelNet, self).__init__()

        assert layers >= 1, 'layers in per block should be no smaller than 1, but layers=[%d]' % layers

        self.kf_dim = kf_dim
        self.ks = ks
        self.layers = layers
        self.num_block = num_block
        self.rc_loc = rc_loc

        # Create the chain of encoder blocks
        moduleConv, modulePool = create_encoder_blocks(0, num_block, layers, c_dim * 2, kf_dim)
        self.moduleConv = torch.nn.ModuleList(moduleConv)
        self.modulePool = torch.nn.ModuleList(modulePool)

        # Create the chain of decoder blocks
        moduleDeconv, moduleUpsample = create_decoder_blocks(num_block - 1, kf_dim, layers, rc_loc)
        self.moduleDeconv = torch.nn.ModuleList(moduleDeconv)
        self.moduleUpsample = torch.nn.ModuleList(moduleUpsample)

        # Create the adaptive kernel blocks
        self.moduleVertical1 = create_1d_kernel_generator_block(self.layers, self.kf_dim, self.ks)
        self.moduleVertical2 = create_1d_kernel_generator_block(self.layers, self.kf_dim, self.ks)
        self.moduleHorizontal1 = create_1d_kernel_generator_block(self.layers, self.kf_dim, self.ks)
        self.moduleHorizontal2 = create_1d_kernel_generator_block(self.layers, self.kf_dim, self.ks)

        self.modulePad = torch.nn.ReplicationPad2d([int(math.floor(self.ks / 2.0)), int(math.floor(self.ks / 2.0)),
                                                    int(math.floor(self.ks / 2.0)), int(math.floor(self.ks / 2.0))])
        self.separableConvolution = SeparableConvolution.apply

    def forward(self, variableInput1, variableInput2, ratio=0):
        """Forward method

        :param variableInput1: The first frame to interpolate between
        :param variableInput2: The second frame to interpolate between
        :param ratio: The value to use for the time input. Not used if self.rc_loc is negative
        """

        variableJoin = torch.cat([variableInput1, variableInput2], 1)

        variableConv = []
        variablePool = []

        # Pass the input through the encoder chain
        for i in range(self.num_block):
            if i == 0:
                variableConv.append(self.moduleConv[i](variableJoin))
            else:
                variableConv.append(self.moduleConv[i](variablePool[-1]))
            variablePool.append(self.modulePool[i](variableConv[-1]))

        # Pass the result through the decoder chain, applying skip connections from the encoder
        variableDeconv = []
        variableUpsample = []
        variableCombine = []
        for i in range(self.num_block-1):
            if i == 0:
                layer_input = variablePool[-1]
            else:
                layer_input = variableCombine[-1]
            variableDeconv.append(self.moduleDeconv[i](layer_input))

            # Inject time information
            if i == self.rc_loc - 1:
                input_size = list(variableDeconv[-1].size())
                rc = Variable(ratio * torch.ones(input_size[0], 1, input_size[2], input_size[3]))
                rc = rc.cuda()
                variableDeconv[-1] = torch.cat([variableDeconv[-1], rc], dim=1)

            # Upsample and apply skip connection
            variableUpsample.append(self.moduleUpsample[i](variableDeconv[-1]))
            variableCombine.append(variableUpsample[-1] + variableConv[self.num_block-i-1])

        # Apply the kernels to the source images
        variableDot1 = self.separableConvolution(self.modulePad(variableInput1),
                                                 self.moduleVertical1(variableCombine[-1]),
                                                 self.moduleHorizontal1(variableCombine[-1]),
                                                 self.ks)
        variableDot2 = self.separableConvolution(self.modulePad(variableInput2),
                                                 self.moduleVertical2(variableCombine[-1]),
                                                 self.moduleHorizontal2(variableCombine[-1]),
                                                 self.ks)
        return variableDot1, variableDot2


class KernelNetShallow(nn.Module):
    """A shallow variant of the adaptive separable convolutional network for video frame interpolation.

    Instead of taking raw frames, this module takes encoded representations of frames at a reduced resolution, as well
    as intermediate activations associated with the encoded representations.

    This module can optionally take in a "ratio" argument in the forward pass, where the ratio represents which time
    step is being generated.
    """

    def __init__(self, gf_dim, ks, num_block=5, layers=3, kf_dim=32, rc_loc=-1):
        """Constructor

        :param gf_dim: The number of channels in the input encodings
        :param ks: The size of the 1D kernel
        :param num_block: Controls the number of blocks to use in the encoder and decoder chains
        :param layers: The number of layers to use in each encoder and decoder block
        :param kf_dim: Controls the number of filters in each encoder and decoder block
        :param rc_loc: The index of the block to inject temporal information into. If -1, do not inject time information
        """

        super(KernelNetShallow, self).__init__()

        assert layers >= 1, 'layers in per block should be no smaller than 1, but layers=[%d]' % layers
        assert num_block >= 4, '# blocks should be no less than 3, but num_block=%d' % num_block

        self.kf_dim = kf_dim
        self.ks = ks
        self.layers = layers
        self.num_block = num_block
        self.rc_loc = rc_loc

        # Create the chain of encoder blocks
        moduleConv, modulePool = create_encoder_blocks(3, num_block, layers, gf_dim * 8 * 2, kf_dim)
        self.moduleConv = torch.nn.ModuleList(moduleConv)
        self.modulePool = torch.nn.ModuleList(modulePool)

        # Create the chain of decoder blocks
        moduleDeconv, moduleUpsample = create_decoder_blocks(num_block - 1, kf_dim, layers, rc_loc)
        self.moduleDeconv = torch.nn.ModuleList(moduleDeconv)
        self.moduleUpsample = torch.nn.ModuleList(moduleUpsample)

        # Create the adaptive kernel blocks
        self.moduleVertical1 = create_1d_kernel_generator_block(self.layers, self.kf_dim, self.ks)
        self.moduleVertical2 = create_1d_kernel_generator_block(self.layers, self.kf_dim, self.ks)
        self.moduleHorizontal1 = create_1d_kernel_generator_block(self.layers, self.kf_dim, self.ks)
        self.moduleHorizontal2 = create_1d_kernel_generator_block(self.layers, self.kf_dim, self.ks)

        self.modulePad = torch.nn.ReplicationPad2d([int(math.floor(self.ks / 2.0)), int(math.floor(self.ks / 2.0)),
                                                    int(math.floor(self.ks / 2.0)), int(math.floor(self.ks / 2.0))])
        self.separableConvolution = SeparableConvolution.apply

    def forward(self, variableInput1, variableInput2, variableDyn1, variableDyn2, variableCont1, variableCont2,
                variableRes=None, ratio=0):
        """Forward method

        :param variableInput1: The encoding of the first frame to interpolate between
        :param variableInput2: The encoding of the second frame to interpolate between
        :param variableDyn1: The intermediate activations from MotionEnc associated with the first frame
        :param variableDyn2: The intermediate activations from MotionEnc associated with the second frame
        :param variableCont1: The intermediate activations from ContentEnc associated with the first frame
        :param variableCont2: The intermediate activations from ContentEnc associated with the second frame
        :param variableRes: The output of the residual layers that combine the residual activations from both frames
        :param ratio: The value to use for the time input. Not used if self.rc_loc is negative
        """

        variableJoin = torch.cat([variableDyn1, variableDyn2, variableCont1, variableCont2], 1)

        variableConv = []
        variablePool = []

        # Pass the input through the encoder chain
        for i in range(self.num_block-3):
            if i == 0:
                variableConv.append(self.moduleConv[i](variableJoin))
            else:
                variableConv.append(self.moduleConv[i](variablePool[-1]))
            variablePool.append(self.modulePool[i](variableConv[-1]))

        # Pass the result through the decoder chain, applying skip connections from the encoder
        variableDeconv = []
        variableUpsample = []
        variableCombine = []
        for i in range(self.num_block-1):
            if i == 0:
                layer_input = variablePool[-1]
            else:
                layer_input = variableCombine[-1]
            variableDeconv.append(self.moduleDeconv[i](layer_input))

            # Inject time information
            if i == self.rc_loc - 1:
                input_size = list(variableDeconv[-1].size())
                rc = Variable(ratio * torch.ones(input_size[0], 1, input_size[2], input_size[3]))
                rc = rc.cuda()
                variableDeconv[-1] = torch.cat([variableDeconv[-1], rc], dim=1)

            # Upsample
            variableUpsample.append(self.moduleUpsample[i](variableDeconv[-1]))
            if i < (self.num_block-3):
                # Apply skip connection from the encoder
                variableCombine.append(variableUpsample[-1] + variableConv[self.num_block-3-i-1])
            else:
                # Apply skip connection from the residual layers iff the activations are given
                if not variableRes is None:
                    variableCombine.append(variableUpsample[-1] + variableRes[self.num_block-i-1])
                else:
                    variableCombine.append(variableUpsample[-1])

        # Apply the kernels to the source images
        variableDot1 = self.separableConvolution(self.modulePad(variableInput1),
                                                 self.moduleVertical1(variableCombine[-1]),
                                                 self.moduleHorizontal1(variableCombine[-1]),
                                                 self.ks)
        variableDot2 = self.separableConvolution(self.modulePad(variableInput2),
                                                 self.moduleVertical2(variableCombine[-1]),
                                                 self.moduleHorizontal2(variableCombine[-1]),
                                                 self.ks)
        return variableDot1, variableDot2


##################################################
#########  BIDIRECTIONAL KERNEL NETWORKS  ########
##################################################


class BaseKernelGenerator(nn.Module):
    """Parent class for the KernelGenerator classes."""

    def predict(self, K, T, state, image_size, diff_in, xt):
        """Forward the given difference frames and content frame through the generator (MC-Net).

        :param K: The number of input time steps
        :param T: The number of output time steps
        :param diff_in: The difference frames
        :param xt: The content frame
        :return: (pred, dyn, cont, res) where:
                 - pred is the predicted frames
                 - dyn is the motion encodings for the output time steps
                 - cont is the content encodings for the output time steps
                 - res is the residual layer outputs for the output time steps
        """

        # Compute the motion encoding at each time step
        for t in range(K-1):
            enc_h, res_m = self.motion_enc(diff_in[t])
            h_dyn, state = self.convLstm_cell(enc_h, state)

        # Keep track of outputs
        pred = []
        dyn = []
        cont = []
        res = []
        for t in range(T):
            # Compute the representation of the next motion frame
            if t > 0:
                enc_h, res_m = self.motion_enc(diff_in[-1])
                h_dyn, state = self.convLstm_cell(enc_h, state)
            # Compute the representation of the next content frame
            h_cont, res_c = self.content_enc(xt)
            # Combine the motion and content encodings
            h_tpl = self.comb_layers(h_dyn, h_cont)
            # Store the motion and content encodings
            dyn.append(h_dyn)
            cont.append(h_cont)
            # Pass intermediate activations through the residual layers
            if self.enable_res:
                res_1 = self.residual1(res_m[0], res_c[0])
                res_2 = self.residual2(res_m[1], res_c[1])
                res_3 = self.residual3(res_m[2], res_c[2])
                res.append([res_1, res_2, res_3])
                # Pass activations through the decoder
                x_hat = self.dec_cnn(h_tpl, res_1, res_2, res_3)
            else:
                # Pass activations through the decoder
                x_hat = self.dec_cnn(h_tpl)

            # Obtain grayscale versions of the predicted frames
            if self.c_dim == 3:
                x_hat_gray = bgr2gray(inverse_transform(x_hat))
                xt_gray = bgr2gray(inverse_transform(xt))
            else:
                x_hat_gray = inverse_transform(x_hat)
                xt_gray = inverse_transform(xt)

            # Compute the next MotionEnc input, which is difference between grayscale frames
            diff_in.append(x_hat_gray - xt_gray)
            # Update last past frame
            xt = x_hat
            # Update outputs
            pred.append(x_hat.view(-1, self.c_dim, image_size[0], image_size[1]))

        return pred, dyn, cont, res


class KernelGenerator(BaseKernelGenerator):
    """A video frame inpainting network that predicts two sets of middle frames and blends them with a KernelNet."""

    def __init__(self, gf_dim, c_dim, feature_size, ks, num_block=5, kf_dim=32, layers=3, enable_res=True,
                 forget_bias=1, activation=F.tanh, bias=True, rc_loc=-1):
        """Constructor

        :param gf_dim: The number of filters to use in the MotionEnc and ContentEnc modules
        :param c_dim: The number of image channels (e.g. 3 for RGB)
        :param feature_size: The kernel size of the ConvLSTM
        :param ks: The size of the 1D kernel to generate with the KernelNet module
        :param num_block: Controls the number of blocks to use in the encoder and decoder chains of the KernelNet module
        :param kf_dim: Controls the number of filters in each encoder and decoder block in the KernelNet module
        :param layers: The number of layers to use in each encoder and decoder block in the KernelNet module
        :param enable_res: Whether to use residual connections when generating intermediate predictions
        :param forget_bias: The bias for the forget gate in the ConvLSTM
        :param activation: The activation function in the ConvLSTM
        :param bias: Whether to use a bias for the convolutional layer of the ConvLSTM
        :param rc_loc: The index of the KernelNet encoder block to inject temporal information into
        """

        super(KernelGenerator, self).__init__()
        self.c_dim = c_dim
        self.enable_res = enable_res

        self.motion_enc = MotionEnc(gf_dim)
        self.convLstm_cell = ConvLstmCell(feature_size, 4 * gf_dim, forget_bias=forget_bias, activation=activation,
                                          bias=bias)
        self.content_enc = ContentEnc(c_dim, gf_dim)
        self.comb_layers = CombLayers(gf_dim)
        if self.enable_res:
            self.residual3 = Residual(gf_dim * 8, gf_dim * 4)
            self.residual2 = Residual(gf_dim * 4, gf_dim * 2)
            self.residual1 = Residual(gf_dim * 2, gf_dim * 1)
        self.dec_cnn = DecCnn(c_dim, gf_dim)

        self.kernelnet = KernelNet(c_dim, ks=ks, num_block=num_block, layers=layers, kf_dim=kf_dim, rc_loc=rc_loc)

    def forward(self, K, T, F, state, image_size, diff_in, diff_in_F, xt, xt_F, comb_type):
        """Forward method

        :param K: The number of preceding frames
        :param T: The number of middle frames
        :param F: The number of following frames
        :param image_size: The resolution of the input video
        :param diff_in: The difference frames in the preceding sequence
        :param diff_in_F: The difference frames in the following sequence
        :param xt: The last preceding frame
        :param xt_F: The first following frame
        :param comb_type: Whether to combine the outputs of the KernelNet module with standard or time-weighted
                          averaging
        """

        # Generate the forward and backward predictions
        forward_pred, _, _, _ = self.predict(K, T, state, image_size, diff_in, xt)
        backward_pred, _, _, _ = self.predict(F, T, state, image_size, diff_in_F, xt_F)
        # Correct the order of the backward frames
        backward_pred = backward_pred[::-1]

        # Store the final predictions
        combination = []
        # Compute the time information to inject
        w = np.linspace(0, 1, num=T+2).tolist()[1:-1]
        for t in range(T):
            # Do time-aware frame interpolation on the forward and backward predictions
            variableDot1, variableDot2 = self.kernelnet(forward_pred[t], backward_pred[t], ratio=1-w[t])

            # Merge the modified forward and backward predictions
            if comb_type == 'avg':
                combination.append(0.5 * variableDot1 + 0.5 * variableDot2)
            elif comb_type == 'w_avg':
                combination.append(variableDot1.mul(1-w[t]) + variableDot2.mul(w[t]))
            else:
                raise ValueError('comb_type [%s] not recognized.' % comb_type)

        return forward_pred, backward_pred, combination


class KernelShallowGenerator(BaseKernelGenerator):
    """A video frame inpainting network that predicts two sets of middle frames and then blends them with a
    KernelNetShallow module.

    In contrast to KernelGenerator, KernelShallowGenerator computes the kernels that get applied to the intermediate
    predictions using the intermediate activations from the generator (MC-Net).
    """

    def __init__(self, gf_dim, c_dim, feature_size, ks, num_block=5, kf_dim=32, layers=3, enable_res=True,
                 forget_bias=1, activation=F.tanh, bias=True, rc_loc=-1):
        """Constructor

        :param gf_dim: The number of filters to use in the MotionEnc and ContentEnc modules
        :param c_dim: The number of image channels (e.g. 3 for RGB)
        :param feature_size: The kernel size of the ConvLSTM
        :param ks: The size of the 1D kernel to generate with the KernelNet module
        :param num_block: Controls the number of blocks to use in the encoder and decoder chains of the KernelNet module
        :param layers: The number of layers to use in each encoder and decoder block in the KernelNet module
        :param kf_dim: Controls the number of filters in each encoder and decoder block in the KernelNet module
        :param enable_res: Whether to use residual connections when generating intermediate predictions
        :param forget_bias: The bias for the forget gate in the ConvLSTM
        :param activation: The activation function in the ConvLSTM
        :param bias: Whether to use a bias for the convolutional layer of the ConvLSTM
        :param rc_loc: The index of the KernelNet encoder block to inject temporal information into
        """

        super(KernelShallowGenerator, self).__init__()
        self.c_dim = c_dim
        self.enable_res = enable_res

        self.motion_enc = MotionEnc(gf_dim)
        self.convLstm_cell = ConvLstmCell(feature_size, 4 * gf_dim, forget_bias=forget_bias, activation=activation,
                                          bias=bias)
        self.content_enc = ContentEnc(c_dim, gf_dim)
        self.comb_layers = CombLayers(gf_dim)
        if self.enable_res:
            self.residual3 = Residual(gf_dim * 8, gf_dim * 4)
            self.residual2 = Residual(gf_dim * 4, gf_dim * 2)
            self.residual1 = Residual(gf_dim * 2, gf_dim * 1)
            self.merge_residual3 = Residual(gf_dim * 8, kf_dim * 4)
            self.merge_residual2 = Residual(gf_dim * 4, kf_dim * 2)
            self.merge_residual1 = Residual(gf_dim * 2, kf_dim * 1)
        self.dec_cnn = DecCnn(c_dim, gf_dim)

        self.kernelnet = KernelNetShallow(gf_dim, ks=ks, num_block=num_block, layers=layers, kf_dim=kf_dim,
                                          rc_loc=rc_loc)

    def forward(self, K, T, F, state, image_size, diff_in, diff_in_F, xt, xt_F, comb_type):
        """Forward method

        :param K: The number of preceding frames
        :param T: The number of middle frames
        :param F: The number of following frames
        :param image_size: The resolution of the input video
        :param diff_in: The difference frames in the preceding sequence
        :param diff_in_F: The difference frames in the following sequence
        :param xt: The last preceding frame
        :param xt_F: The first following frame
        :param comb_type: Whether to combine the outputs of the KernelNet module with standard or time-weighted
                          averaging
        """

        # Generate the forward and backward predictions
        forward_pred, forward_dyn, forward_cont, forward_res = self.predict(K, T, state, image_size, diff_in, xt)
        backward_pred, backward_dyn, backward_cont, backward_res = self.predict(F, T, state, image_size, diff_in_F,
                                                                                xt_F)
        # Correct the order of the backward frames
        backward_pred = backward_pred[::-1]
        backward_dyn = backward_dyn[::-1]
        backward_cont = backward_cont[::-1]
        backward_res = backward_res[::-1]

        # Store the final predictions
        combination = []
        # Compute the time information to inject
        w = np.linspace(0, 1, num=T+2).tolist()[1:-1]
        for t in range(T):
            merged_res = None
            if self.enable_res:
                merged_res = []
                merged_res.append(self.merge_residual1(forward_res[t][0], backward_res[t][0]))
                merged_res.append(self.merge_residual2(forward_res[t][1], backward_res[t][1]))
                merged_res.append(self.merge_residual3(forward_res[t][2], backward_res[t][2]))
                # Do time-aware frame interpolation on the forward and backward predictions
                variableDot1, variableDot2 = self.kernelnet(forward_pred[t], backward_pred[t], forward_dyn[t],
                                                            backward_dyn[t], forward_cont[t], backward_cont[t],
                                                            merged_res, ratio=1-w[t])
            else:
                # Do time-aware frame interpolation on the forward and backward predictions
                variableDot1, variableDot2 = self.kernelnet(xt, xt_F, forward_dyn[t], backward_dyn[t], forward_cont[t],
                                                            backward_cont[t], merged_res, ratio=1-w[t])

            # Merge the modified forward and backward predictions
            if comb_type == 'avg':
                combination.append(0.5 * variableDot1 + 0.5 * variableDot2)
            elif comb_type == 'w_avg':
                combination.append(variableDot1.mul(1-w[t]) + variableDot2.mul(w[t]))
            else:
                raise ValueError('comb_type [%s] not recognized.' % comb_type)

        return forward_pred, backward_pred, combination
