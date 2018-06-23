import numpy as np

from models.generators import Generator
from .bidirectional_prediction_environment import BidirectionalPredictionEnvironment


class McnetInfillEnvironment(BidirectionalPredictionEnvironment):
    def name(self):
        """Return a string that identifies this type of environment."""
        return 'McnetInfillEnvironment'

    def create_generator(self, gf_dim, c_dim):
        """Initialize the video frame inpainting or prediction model (the generator).

        :param gf_dim: The number of filters to use in the MotionEnc and ContentEnc modules
        :param c_dim: The number of image channels (e.g. 3 for RGB)
        """
        return Generator(gf_dim, c_dim, 3)

    def forward(self):
        """Forward the current inputs through the video frame inpainting or prediction model."""

        # Generate the forward prediction
        target_forward = self.targets[:-self.F]
        self.pred_forward = self.generator(self.K, self.T, self.state, self.image_size, self.diff_in,
                                           target_forward[self.K - 1])
        # Generate the backward prediction
        target_backward = self.targets[self.K:][::-1]
        self.pred_backward = self.generator(self.F, self.T, self.state, self.image_size, self.diff_in_F,
                                            target_backward[self.F - 1])[::-1]

        # Combine the predictions
        self.pred = []
        if self.comb_type == 'avg':
            for t in range(self.T):
                self.pred.append((self.pred_forward[t] + self.pred_backward[t])/2)
        elif self.comb_type == 'w_avg':
            w = np.linspace(0, 1, num=self.T+2).tolist()[1:-1]
            for t in range(self.T):
                self.pred.append(self.pred_forward[t].mul(1-w[t]) + self.pred_backward[t].mul(w[t]))
        else:
            raise ValueError('comb_type [%s] not recognized.' % self.comb_type)
