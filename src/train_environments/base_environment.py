import os
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
from torch.autograd import Variable

from util.util import move_to_devices, weights_init, weights_init_std_discriminator

from models.discriminators import SNDiscriminator, Discriminator
from models.generators import GDL


class BaseEnvironment:
    """
    Environments manage the various aspects of training and deploying a video frame inpainting/prediction model, such as
    loading/saving snapshots, defining models, computing loss, etc. BaseEnvironment is the parent class of all such
    environments.
    """

    __metaclass__ = ABCMeta

    def __init__(self, generator_args, gpu_ids, is_train, checkpoints_dir, name, K, T, image_size, batch_size, c_dim,
                 no_adversarial, alpha, beta, D_G_switch, margin, lr, beta1, sn, df_dim, Ip, disc_t):
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
        """


        self.gpu_ids = gpu_ids
        self.is_train = is_train
        self.save_dir = os.path.join(checkpoints_dir, name)

        self.K = K
        self.T = T

        self.image_size = image_size
        self.batch_size = batch_size

        # Set the initial state of the ConvLSTM in the video prediction module
        self.state = Variable(torch.zeros(self.batch_size, 512, self.image_size[0]/8, self.image_size[1]/8).cuda(),
                              requires_grad=False)

        # Initialize flags that determine whether to update the disriminator and generator
        self.updateD = True
        self.updateG = True

        # basic training setting: loss definition
        if self.is_train:
            self.start_update = 0
            self.loss_Lp = torch.nn.MSELoss()
            self.loss_gdl = GDL(c_dim)

            if not no_adversarial:
                self.loss_d = torch.nn.BCELoss()

            self.alpha = alpha
            self.beta = beta

            self.D_G_switch = D_G_switch
            self.margin = margin

        self.targets = []

        # define generator
        self.generator = self.create_generator(*generator_args)
        self.generator = move_to_devices(self.generator, gpu_ids)
        self.generator.apply(weights_init)

        # training setting
        if self.is_train:
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                                lr=lr, betas=(beta1, 0.999))

            # define discriminator
            self.no_adversarial = no_adversarial
            if not no_adversarial:
                if sn:
                    # Use spectral-normalized discriminator
                    discriminator = SNDiscriminator(image_size, c_dim, disc_t, df_dim, Ip)
                    discriminator = move_to_devices(discriminator, gpu_ids)
                    discriminator.apply(weights_init)
                else:
                    # Use normal discriminator
                    discriminator = Discriminator(image_size, c_dim, disc_t, df_dim)
                    discriminator = move_to_devices(discriminator, gpu_ids)
                    discriminator.apply(weights_init_std_discriminator)
                self.discriminator = discriminator
                self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    @abstractmethod
    def name(self):
        """Return a string that identifies this type of environment."""
        pass

    @abstractmethod
    def create_generator(self, *args):
        """Initialize the video frame inpainting or prediction model (the generator)."""
        pass

    @abstractmethod
    def forward(self):
        """Forward the current inputs through the video frame inpainting or prediction model."""
        pass

    @abstractmethod
    def backward_G(self):
        """Compute the generator's loss and backprop the loss gradients through the generator."""
        pass

    @abstractmethod
    def create_input_fake(self):
        """Return fake videos (i.e. videos containing inpainted/predicted frames)."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Perform one generator update step and one discriminator update step (if applicable)."""
        pass

    @abstractmethod
    def get_current_errors(self):
        """Obtain a dict that specifies the losses associated with training."""
        pass

    @abstractmethod
    def set_inputs(self, input):
        """Set the current data to use for computing fake videos, losses, etc."""
        pass

    def enable_inter_loss(self):
        """Turn on the penalty term on intermediate predictions (if applicable)."""
        pass

    def enable_final_loss(self):
        """Turn on the penalty term on final predictions (if applicable)."""
        pass

    def set_start_iter(self, iter):
        """Initialize the iteration count."""
        self.start_iter = iter

    def backward_D(self):
        """Compute the discriminator's loss on real and fake videos, and backprop the loss through the discriminator."""

        # fake
        input_fake = self.create_input_fake()
        input_fake_ = Variable(input_fake.data)
        h_sigmoid, h = self.discriminator(input_fake_)
        labels = Variable(torch.zeros(h.size()).cuda(async=True))
        self.loss_d_fake = self.loss_d(h_sigmoid, labels)

        # real
        input_real = torch.cat(self.targets, dim=1)
        input_real_ = Variable(input_real.data)
        h_sigmoid_, h_ = self.discriminator(input_real_)
        labels_ = Variable(torch.ones(h_.size()).cuda(async=True))
        self.loss_d_real = self.loss_d(h_sigmoid_, labels_)

        self.loss_D = self.loss_d_fake + self.loss_d_real

        self.loss_D.backward()

    def get_current_visuals(self):
        """Obtain a dict of video tensors to visualize in TensorBoard."""

        vis_dict = {
            'seq_batch': self.data['targets'].cpu(), # [1, 1, h, w, t]
            'pred': [a.data.cpu() for a in self.pred] # [1, 1, h, w]
        }

        return vis_dict

    def get_current_state_dict(self, total_updates):
        """Get a dict defining the current state of training (used for snapshotting).

        :param total_updates: The number of training iterations performed so far
        """

        generator_state_dict = self.generator.module.state_dict() if len(self.gpu_ids) > 1 else self.generator.state_dict()
        current_state = {
            'updates': total_updates,
            'generator': generator_state_dict,
            'optimizer_G': self.optimizer_G.state_dict(),
            'updateD': self.updateD,
            'updateG': self.updateG,
        }

        if not self.no_adversarial:
            discriminator_state_dict = self.discriminator.module.state_dict() if len(self.gpu_ids) > 1 else self.discriminator.state_dict()
            current_state['discriminator'] = discriminator_state_dict
            current_state['optimizer_D'] = self.optimizer_D.state_dict()

        return current_state

    def load(self, update_label):
        """Load a snapshot of the environment.

        :param update_label: The name of the snapshot to load
        """

        save_filename = '%s_model.pth.tar' % (update_label)
        save_path = os.path.join(self.save_dir, save_filename)
        snapshot = None

        if os.path.isfile(save_path):
            print('=> loading snapshot from {}'.format(save_path))
            snapshot = torch.load(save_path)
            generator_state_dict = OrderedDict()
            for k, v in snapshot['generator'].items():
                if len(self.gpu_ids) > 1:
                    if not k.startswith('module.'):
                        name = 'module.' + k  # add `module.`
                        generator_state_dict[name] = v
                    else:
                        generator_state_dict[k] = v
                else:
                    generator_state_dict[k] = v
            self.generator.load_state_dict(generator_state_dict)
            if self.is_train:
                self.start_update = snapshot['updates']
                self.optimizer_G.load_state_dict(snapshot['optimizer_G'])
                if not self.no_adversarial:
                    discriminator_state_dict = OrderedDict()
                    for k, v in snapshot['discriminator'].items():
                        if len(self.gpu_ids) > 1:
                            if not k.startswith('module.'):
                                name = 'module.' + k  # add `module.`
                                discriminator_state_dict[name] = v
                            else:
                                discriminator_state_dict[k] = v
                        else:
                            discriminator_state_dict[k] = v
                    self.discriminator.load_state_dict(discriminator_state_dict)
                    self.optimizer_D.load_state_dict(snapshot['optimizer_D'])
                self.updateD = snapshot['updateD']
                self.updateG = snapshot['updateG']

        return snapshot

    def save(self, label, total_updates):
        """Save the current state of the environment.

        :param label: A name for the snapshot to save
        :param total_updates: The number of training iterations performed so far
        """
        current_state = self.get_current_state_dict(total_updates)
        save_filename = '%s_model.pth.tar' % (label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(current_state, save_path)