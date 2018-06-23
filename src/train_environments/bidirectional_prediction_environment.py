import torch
from torch.autograd import Variable

from util.util import inverse_transform
from .base_environment import BaseEnvironment


class BidirectionalPredictionEnvironment(BaseEnvironment):
    """The base environment for models that perform video frame inpainting by blending together a forward and a
    backward prediction."""

    def __init__(self, generator_args, gpu_ids, is_train, checkpoints_dir, name, K, T, image_size, batch_size, c_dim,
                 no_adversarial, alpha, beta, D_G_switch, margin, lr, beta1, sn, df_dim, Ip, disc_t, F, comb_type,
                 comb_loss):
        """Constructor

        :param generator_args: A tuple of arguments used to initialize the generator associated with this environment
        :param gpu_ids: The device numbers of the GPUs to use
        :param is_train: Whether this is a training environment (or False for a testing environment)
        :param checkpoints_dir: The root folder where checkpoints are stored
        :param name: The name of the experiment
        :param K: The number of preceding frames
        :param T: The number of output frames (e.g. middle or future frames)
        :param image_size: The spatial resolution of the video
        :param batch_size: The batch size of the data
        :param c_dim: The number of color channels (e.g. 3 for RGB)
        :param no_adversarial: If True, do not use a discriminator/adversarial loss
        :param alpha: The weight of the image reconstruction-based loss
        :param beta: The weight of the adversarial/discriminator-based loss
        :param D_G_switch: A string indicating when to update the discriminator and generator (see train_options.py)
        :param margin: The margin to use when choosing whether to update the discriminator and generator (used if
                       D_G_switch is 'selective'
        :param lr: The learning rate of the optimizers
        :param beta1: The first beta term used by the Adam optimizer
        :param sn: Whether to use a spectral-norm discriminator
        :param df_dim: Controls the number of features in each layer of the discriminator
        :param Ip: The number of power iterations to use when computing max singular value (used if sn is True)
        :param disc_t: The total number of frames per video that the discriminator will take in
        :param F: The number of following frames
        :param comb_type: The method of blending the intermediate predictions ('avg' for simple averaging and 'w_avg'
                          for time-weighted averaging)
        :param comb_loss: How to penalize the intermediate predictions ('ToTarget' to penalize based on similarity to
                          the target, and 'Closer' to penalize based on similarity to each other)
        """

        super(BidirectionalPredictionEnvironment, self).__init__(generator_args, gpu_ids, is_train, checkpoints_dir, name, K, T,
                                                                 image_size, batch_size, c_dim, no_adversarial, alpha, beta,
                                                                 D_G_switch, margin, lr, beta1, sn, df_dim, Ip, disc_t)

        self.F = F
        self.comb_type = comb_type

        if self.is_train:
            self.inter_loss = False
            self.comb_loss = comb_loss
        self.final_loss = False

    def enable_inter_loss(self):
        """Turn on the penalty term on intermediate predictions (if applicable)."""
        self.inter_loss = True

    def enable_final_loss(self):
        """Turn on the penalty term on final predictions (if applicable)."""
        self.final_loss = True

    def set_inputs(self, input):
        """Set the current data to use for computing fake videos, losses, etc."""

        self.data = input
        targets = input['targets'] # shape[-1] = K+T
        diff_in = input['diff_in'] # shape[-1] = K-1
        seq_len = int(targets.size()[-1])
        diff_in_F = input['diff_in_F']
        self.diff_in = []
        self.diff_in_F = []
        self.targets = []
        f_volatile = not self.updateG or not self.is_train
        for i in range(self.K - 1):
            self.diff_in.append(Variable(diff_in[:, :, :, :, i].contiguous().cuda(async=True), volatile=f_volatile))
        for i in range(self.F - 1):
            self.diff_in_F.append(Variable(diff_in_F[:, :, :, :, i].contiguous().cuda(async=True), volatile=f_volatile))
        for i in range(seq_len):
            self.targets.append(Variable(targets[:, :, :, :, i].contiguous().cuda(async=True), volatile=f_volatile))

    def create_input_fake(self):
        """Return fake videos (i.e. videos containing inpainted/predicted frames)."""
        input_fake = torch.cat(self.targets[:self.K] + self.pred + self.targets[self.K + self.T:], dim=1)
        return input_fake

    def backward_G(self):
        """Compute the generator's loss and backprop the loss gradients through the generator."""

        # Map image intensities to [0, 1]
        targets = inverse_transform(torch.cat(self.targets[self.K:self.K + self.T], dim=0))

        self.loss_G = 0

        # Compute reconstruction losses associated with the final prediction
        if self.final_loss:
            # Map image intensities to [0, 1]
            outputs = inverse_transform(torch.cat(self.pred, dim=0))
            self.Lp = self.loss_Lp(outputs, targets)
            self.gdl = self.loss_gdl(outputs, targets)
            self.loss_G = (self.Lp + self.gdl) * self.alpha

        # Compute reconstruction losses associated with the intermediate predictions
        if self.inter_loss:
            # Map image intensities to [0, 1]
            outputs_forward = inverse_transform(torch.cat(self.pred_forward, dim=0))
            outputs_backward = inverse_transform(torch.cat(self.pred_backward, dim=0))

            if self.comb_loss == 'ToTarget':
                # Compute intermediate prediction loss based on similarity to target
                self.Lp_forward = self.loss_Lp(outputs_forward, targets)
                self.Lp_backward = self.loss_Lp(outputs_backward, targets)
                self.gdl_forward = self.loss_gdl(outputs_forward, targets)
                self.gdl_backward = self.loss_gdl(outputs_backward, targets)
                self.loss_G += self.alpha * (self.Lp_forward + self.Lp_backward + self.gdl_forward + self.gdl_backward)
            elif self.comb_loss == 'Closer':
                # Compute intermediate prediction loss based on similarity to each other
                forward_target = Variable(outputs_forward.data.clone())
                backward_target = Variable(outputs_backward.data.clone())
                self.Lp_forward = self.loss_Lp(outputs_forward, backward_target)
                self.Lp_backward = self.loss_Lp(outputs_backward, forward_target)
                self.gdl_forward = self.loss_gdl(outputs_forward, backward_target)
                self.gdl_backward = self.loss_gdl(outputs_backward, forward_target)
                self.loss_G += self.alpha * (self.Lp_forward + self.Lp_backward + self.gdl_forward + self.gdl_backward)
            else:
                raise ValueError('comb_loss [%s] not recognized.' % self.comb_loss)

        # Compute adversarial loss (on final predictions only)
        if not self.no_adversarial and self.final_loss:
            # Only add discriminator loss in adversarial setting when final results are being penalized
            input_fake = self.create_input_fake()
            h_sigmoid, h = self.discriminator(input_fake)
            labels = Variable(torch.ones(h.size()).cuda(async=True))
            self.L_GAN = self.loss_d(h_sigmoid, labels)

            if not self.updateD:
                # Store discriminator's current performance on real and fake videos
                labels_ = Variable(torch.zeros(h.size()).cuda(async=True))
                self.loss_d_fake = self.loss_d(h_sigmoid, labels_)

                input_real = torch.cat(self.targets, dim=1)
                input_real_ = Variable(input_real.data)
                h_sigmoid_, h_ = self.discriminator(input_real_)
                labels__ = Variable(torch.ones(h_.size()).cuda(async=True))
                self.loss_d_real = self.loss_d(h_sigmoid_, labels__)

            self.loss_G += self.beta * self.L_GAN

        self.loss_G.backward()

    def optimize_parameters(self):
        """Perform one generator update step and one discriminator update step (if applicable)."""

        self.forward()

        if self.no_adversarial or not self.final_loss:
            # Only update generator if there's no discriminator, or if there's a discriminator but final result is not
            # being penalized yet
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
        if self.final_loss:
            # Final prediction losses are only defined when the final result is being penalized
            loss += [('G_Lp', self.Lp.data[0]),
                    ('G_gdl', self.gdl.data[0]),
                    ('G_Final_Total', self.Lp.data[0] + self.gdl.data[0])]

        if not self.no_adversarial and self.final_loss:
            # Discriminator losses are only defined in adversarial setting and when final result is penalized
            loss_adversarial = [('D_real', self.loss_d_real.data[0]),
                                ('D_fake', self.loss_d_fake.data[0]),
                                ('G_GAN', self.L_GAN.data[0]),
                                ('G_loss_wo_D', self.loss_G.data[0] - self.beta * self.L_GAN.data[0])]
            loss = loss + loss_adversarial
        else:
            loss.append(('G_loss_wo_D', self.loss_G.data[0]))

        if self.inter_loss:
            loss_comb = [('G_Lp_forward', self.Lp_forward.data[0]),
                         ('G_gdl_forward', self.gdl_forward.data[0]),
                         ('G_Forward_Total', self.Lp_forward.data[0] + self.gdl_forward.data[0]),
                         ('G_Lp_backward', self.Lp_backward.data[0]),
                         ('G_gdl_backward', self.gdl_backward.data[0]),
                         ('G_Backward_Total', self.Lp_backward.data[0] + self.gdl_backward.data[0])]
            loss = loss + loss_comb

        loss_dict = {k: v for k, v in loss}

        return loss_dict

    def get_current_visuals(self):
        """Obtain a dict of video tensors to visualize in TensorBoard."""

        vis_dict = BaseEnvironment.get_current_visuals(self)

        vis_dict['pred_forward'] = [a.data.cpu() for a in self.pred_forward]
        vis_dict['pred_backward'] = [a.data.cpu() for a in self.pred_backward]

        return vis_dict

    def get_current_state_dict(self, total_updates):
        """Get a dict defining the current state of training (used for snapshotting).

        :param total_updates: The number of training iterations performed so far
        """

        current_state = BaseEnvironment.get_current_state_dict(self, total_updates)
        current_state['inter_loss'] = self.inter_loss
        current_state['final_loss'] = self.final_loss

        return current_state

    def load(self, update_label):
        """Load a snapshot of the environment.

        :param update_label: The name of the snapshot to load
        """

        snapshot = BaseEnvironment.load(self, update_label)
        if snapshot is not None and self.is_train:
            self.final_loss = snapshot['final_loss']
            self.inter_loss = snapshot['inter_loss']