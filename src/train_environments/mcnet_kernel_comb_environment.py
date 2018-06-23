from models import generators
from .bidirectional_prediction_environment import BidirectionalPredictionEnvironment


class McnetKernelCombEnvironment(BidirectionalPredictionEnvironment):
    def name(self):
        """Return a string that identifies this type of environment."""
        return 'McnetKernelCombEnvironment'

    def create_generator(self, shallow, gf_dim, c_dim, ks, num_block, layers, kf_dim, enable_res, rc_loc):
        """Initialize the video frame inpainting or prediction model (the generator).

        :param shallow: If True, use KernelShallowGenerator for video inpainting; otherwise, use KernelGenerator
        :param gf_dim: The number of filters to use in the MotionEnc and ContentEnc modules
        :param c_dim: The number of image channels (e.g. 3 for RGB)
        :param ks: The size of the 1D kernel to generate with the KernelNet module
        :param num_block: Controls the number of blocks to use in the encoder and decoder chains of the KernelNet module
        :param layers: The number of layers to use in each encoder and decoder block in the KernelNet module
        :param kf_dim: Controls the number of filters in each encoder and decoder block in the KernelNet module
        :param enable_res: Whether to use residual connections when generating intermediate predictions
        :param rc_loc: The index of the KernelNet encoder block to inject temporal information into
        """
        if shallow:
            generator = generators.KernelShallowGenerator(gf_dim, c_dim, 3, ks=ks, num_block=num_block, kf_dim=kf_dim,
                                                          layers=layers, enable_res=enable_res, rc_loc=rc_loc)
        else:
            generator = generators.KernelGenerator(gf_dim, c_dim, 3, ks=ks, num_block=num_block, kf_dim=kf_dim,
                                                   layers=layers, enable_res=enable_res, rc_loc=rc_loc)
        return generator

    def forward(self):
        """Forward the current inputs through the video frame inpainting or prediction model."""
        self.pred_forward, self.pred_backward, self.pred = self.generator(self.K, self.T, self.F, self.state,
                                                                          self.image_size, self.diff_in, self.diff_in_F,
                                                                          self.targets[self.K - 1],
                                                                          self.targets[-self.F], self.comb_type)
