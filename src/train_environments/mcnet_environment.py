import torch
from torch.autograd import Variable

from models.generators import Generator
from util.util import inverse_transform
from .base_environment import BaseEnvironment


class McnetEnvironment(BaseEnvironment):
    def name(self):
        """Return a string that identifies this type of environment."""
        return 'McnetEnvironment'

    def create_generator(self, gf_dim, c_dim):
        """Initialize the video frame inpainting or prediction model (the generator).

        :param gf_dim: The number of filters to use in the MotionEnc and ContentEnc modules
        :param c_dim: The number of image channels (e.g. 3 for RGB)
        """
        return Generator(gf_dim, c_dim, 3)

    def set_inputs(self, input):
        """Set the current data to use for computing fake videos, losses, etc."""

        self.data = input
        targets = input['targets'] # shape[-1] = K+T
        diff_in = input['diff_in'] # shape[-1] = K-1
        self.diff_in = []
        self.targets = []
        f_volatile = not self.updateG or not self.is_train
        for i in range(self.K - 1):
            self.diff_in.append(Variable(diff_in[:, :, :, :, i].cuda(), volatile=f_volatile))
        for i in range(self.K + self.T):
            self.targets.append(Variable(targets[:, :, :, :, i].cuda(), volatile=f_volatile))

    def forward(self):
        """Forward the current inputs through the video frame inpainting or prediction model."""

        self.pred = self.generator(self.K, self.T, self.state, self.image_size, self.diff_in, self.targets[self.K - 1])

    def create_input_fake(self):
        """Return fake videos (i.e. videos containing inpainted/predicted frames)."""

        input_fake = torch.cat(self.targets[:self.K] + self.pred, dim=1)
        return input_fake

    def backward_G(self):
        """Compute the generator's loss and backprop the loss gradients through the generator."""

        # Map image intensities to [0, 1]
        outputs = inverse_transform(torch.cat(self.pred, dim=0))
        targets = inverse_transform(torch.cat(self.targets[self.K:], dim=0))

        # Compute reconstruction loss
        self.Lp = self.loss_Lp(outputs, targets)
        self.gdl = self.loss_gdl(outputs, targets)
        self.loss_G = self.alpha * (self.Lp + self.gdl)

        # Compute adversarial loss
        if not self.no_adversarial:
            input_fake = self.create_input_fake()
            h_sigmoid, h = self.discriminator(input_fake)
            labels = Variable(torch.ones(h.size()).cuda())
            self.L_GAN = self.loss_d(h_sigmoid, labels)

            if not self.updateD:
                # Store discriminator's current performance on real and fake videos
                labels_ = Variable(torch.zeros(h.size()).cuda())
                self.loss_d_fake = self.loss_d(h_sigmoid, labels_)

                input_real = torch.cat(self.targets, dim=1)
                input_real_ = Variable(input_real.data)
                h_sigmoid_, h_ = self.discriminator(input_real_)
                labels__ = Variable(torch.ones(h_.size()).cuda())
                self.loss_d_real = self.loss_d(h_sigmoid_, labels__)

            self.loss_G += self.beta * self.L_GAN

        self.loss_G.backward()

    def optimize_parameters(self):
        """Perform one generator update step and one discriminator update step (if applicable)."""

        self.forward()

        if self.no_adversarial:
            # Simply update the generator if no discriminator exists
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
        else:
            if self.D_G_switch == 'selective':
                if self.updateD:
                    self.optimizer_D.zero_grad()
                    self.backward_D()
                    self.optimizer_D.step()

                if self.updateG:
                    self.optimizer_G.zero_grad()
                    self.backward_G()
                    self.optimizer_G.step()

                # If the discriminator's cross-entropy loss is too small, stop updating it
                if self.loss_d_fake.data[0] < self.margin or self.loss_d_real.data[0] < self.margin:
                    self.updateD = False

                # If the generator's cross-entropy loss is too high, stop updating it
                if self.loss_d_fake.data[0] > (1. - self.margin) or self.loss_d_real.data[0] > (1.- self.margin):
                    self.updateG = False

                # If both the discriminator and the generator are performing too poorly, update both of them
                if not self.updateD and not self.updateG:
                    self.updateD = True
                    self.updateG = True

            elif self.D_G_switch == 'alternative':
                # Simply update both the discriminator and generator
                self.optimizer_D.zero_grad()
                self.backward_D()
                self.optimizer_D.step()

                self.optimizer_G.zero_grad()
                self.backward_G()
                self.optimizer_G.step()

            else:
                raise NotImplementedError('switch method [%s] is not implemented' % self.D_G_switch)

    def get_current_errors(self):
        """Obtain a dict that specifies the losses associated with training."""

        loss = [('G_loss', self.loss_G.data[0])]
        loss += [('G_Lp', self.Lp.data[0]),
                ('G_gdl', self.gdl.data[0]),
                ('G_Final_Total', self.Lp.data[0] + self.gdl.data[0])]

        if not self.no_adversarial:
            loss_adversarial = [('D_real', self.loss_d_real.data[0]),
                                ('D_fake', self.loss_d_fake.data[0]),
                                ('G_GAN', self.L_GAN.data[0]),
                                ('G_loss_wo_D', self.loss_G.data[0] - self.beta * self.L_GAN.data[0])]
            loss = loss + loss_adversarial
        else:
            loss.append(('G_loss_wo_D', self.loss_G.data[0]))

        return {k: v for k, v in loss}