from util.util import listopt
from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()

        # optimization parameters
        opt_arg_group = self.parser.add_argument_group('Optimization parameters')
        opt_arg_group.add_argument('--lr', type=float, default=0.0001, help='Base learning rate')
        opt_arg_group.add_argument('--beta1', type=float, default=0.5, help='Momentum term of adam')
        opt_arg_group.add_argument('--max_iter', type=int, default=100000,
                                   help='Maximum number of iterations (batches) to train on')

        # loss params
        loss_arg_group = self.parser.add_argument_group('Loss parameters')
        loss_arg_group.add_argument('--alpha', type=float, default=1.0, help='Image loss weight')
        loss_arg_group.add_argument('--beta', type=float, default=0.02, help='GAN loss weight')
        loss_arg_group.add_argument('--comb_loss',  type=str, default='ToTarget', choices=['ToTarget', 'Closer'],
                                    help='Option to penalize intermediate predictions based on their similarity to the'
                                         'ground truth middle frames ("ToTarget") or to each other ("Closer")')
        loss_arg_group.add_argument('--inter_sup_update', type=int, default=0,
                                    help='The iteration number at which intermediate predictions should be penalized')
        loss_arg_group.add_argument('--final_sup_update', type=int, default=0,
                                    help='The iteration number at which the final prediction should be penalized')

        # freqs
        freq_arg_group = self.parser.add_argument_group('Training frequency parameters')
        freq_arg_group.add_argument('--display_freq', type=int, default=400,
                                    help='Frequency at which to log training results to TensorBoard (in iterations)')
        freq_arg_group.add_argument('--print_freq', type=int, default=400,
                                    help='Frequency at which to print training results in the console (in iterations)')
        freq_arg_group.add_argument('--save_latest_freq', type=int, default=1000,
                                    help='Frequency at which to save a snapshot of the training environment')
        freq_arg_group.add_argument('--validate_freq', type=int, default=40000,
                                    help='Frequency at which to perform model validation')

        # resume experiment
        resume_arg_group = self.parser.add_argument_group('Resume training parameters')
        resume_arg_group.add_argument('--continue_train', action='store_true',
                                      help='Flag to continue training using a stored snapshot')

        # adversarial training
        adv_train_arg_group = self.parser.add_argument_group('Adversarial training parameters')
        adv_train_arg_group.add_argument('--df_dim', type=int, default=64,
                                       help='Number of filters in first conv layer of the discriminator')
        adv_train_arg_group.add_argument('--D_G_switch', type=str, default='selective',
                                         choices=['selective', 'alternative'],
                                         help='Option to always alternate updates between D and G ("alternative") or '
                                              'selectively update D and G based on the margin provided by "--margin" '
                                              '("selective")')
        adv_train_arg_group.add_argument('--margin', type=float, default=0.3,
                                         help='the margin used for choosing whether to optimize D or G. Only used if '
                                              'option "--D_G_switch" is set to "selective"')
        adv_train_arg_group.add_argument('--no_adversarial', action='store_true',
                                         help='Flag to not use the adversarial loss')
        adv_train_arg_group.add_argument('--sn', action='store_true',
                                         help='Flag to use spectral normalization in the discriminator')
        adv_train_arg_group.add_argument('--Ip', type=int, default=3,
                                         help='Number of power iterations used to compute max singular value in the '
                                              'spectral-normalized discriminator (only used if "--sn" option is used)')

    def parse(self):
        opt = BaseOptions.parse(self)

        if opt.video_list is None:
            opt.video_list = 'train_data_list_trimmed.txt'

        listopt(opt)

        return opt
