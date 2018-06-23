import argparse

import torch


class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # general experiment
        exp_arg_group = self.parser.add_argument_group('Experiment parameters')
        exp_arg_group.add_argument('--name', type=str, default='experiment_name', help='Name of the experiment')

        # inputs and outputs
        in_out_arg_group = self.parser.add_argument_group('Model input/output parameters')
        in_out_arg_group.add_argument('--K', type=int, default=10, help='Length of the preceding sequence (in frames)')
        in_out_arg_group.add_argument('--T', type=int, default=10, help='Length of the middle sequence (in frames)')
        in_out_arg_group.add_argument('--F', type=int, default=10, help='Length of the following sequence (in frames)')
        in_out_arg_group.add_argument('--batch_size', type=int, default=8, help='Mini-batch size')
        in_out_arg_group.add_argument('--image_size', type=int,  nargs='+', default=[128],
                                      help='Image size (H x W). Can be specified as two numbers (e.g. "160 208") or '
                                           'one (in which case, H = W)')

        # gpu
        gpu_arg_group = self.parser.add_argument_group('GPU parameters')
        gpu_arg_group.add_argument('--gpu_ids', type=int, nargs='+', default=[0],
                                   help='Device IDs of GPUs to use (e.g. "0" or "0 1 2" or "0 2"')

        # snapshot
        snapshot_arg_group = self.parser.add_argument_group('Snapshot parameters')
        snapshot_arg_group.add_argument('--which_update', type=str, default='latest',
                                        help='The name of the model to load from. Set to "latest" to use the latest '
                                             'cached model')

        # basic dimension
        com_dim_arg_group = self.parser.add_argument_group('Common dimensionality parameters')
        com_dim_arg_group.add_argument('--c_dim', type=int, default=3, help='Number of channels in the image input')
        com_dim_arg_group.add_argument('--gf_dim', type=int, default=64,
                                       help='Number of filters in first conv layer of the generator')

        # path to save ckpt tb and result
        dir_arg_group = self.parser.add_argument_group('Directory parameters')
        dir_arg_group.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                                   help='Path to store checkpoint files')
        dir_arg_group.add_argument('--tensorboard_dir', type=str, default='./tb',
                                   help='Path to store TensorBoard log files')
        dir_arg_group.add_argument('--result_dir', type=str, default='./results',
                                   help='Path to store evaluation results')

        # define model type
        model_type_arg_group = self.parser.add_argument_group('Model type parameters')
        model_type_arg_group.add_argument('--model_type', type=str, default='mcnet',
                                          choices=['mcnet', 'simplecomb', 'kernelcomb'],
                                          help='Whether to use MC-Net ("mcnet"), bidirectional MC-Net with simple '
                                               'blending ("simplecomb"), or bidirectional MC-Net with deep kernel '
                                               'blending ("kernelcomb") as the generator')
        model_type_arg_group.add_argument('--comb_type', type=str, default='avg', choices=['avg', 'w_avg'],
                                          help='Option for how to blend intermediate predictions. Only used when '
                                               '"--model" is "simplecomb" or "kernelcomb"')

        # kernel_based arguments
        kernel_arg_group = self.parser.add_argument_group('Deep kernel blending parameters')
        kernel_arg_group.add_argument('--ks', type=int, default=51, help='Size of the 1D adaptive kernel')
        kernel_arg_group.add_argument('--enable_res', action='store_true',
                                      help='Whether to include the residual network in the model')
        kernel_arg_group.add_argument('--num_block', type=int, default=5,
                                      help='The number of decoder blocks in the kernel generator subnetwork')
        kernel_arg_group.add_argument('--layers', type=int, default=3,
                                      help='Number of convolution layers in each encoder/decoder block of the kernel '
                                           'generator subnetwork')
        kernel_arg_group.add_argument('--kf_dim', type=int, default=32,
                                      help='The number of filters in the first block of the kernel generator'
                                           ' subnetwork')
        kernel_arg_group.add_argument('--shallow', action='store_true',
                                      help='Flag to use the intermediate activations from the prediction subnetwork in '
                                           'the kernel generator subnetwork')
        kernel_arg_group.add_argument('--rc_loc', type=int, default=-1,
                                      help='The index of the decoder block where the scaled time step should be'
                                           ' concatenated')

        # data loading
        data_load_arg_group = self.parser.add_argument_group('Data loading parameters')
        data_load_arg_group.add_argument('--serial_batches', action='store_true',
                                         help='Flag for loading videos sequentially. If False, videos will be loaded '
                                              'randomly')
        data_load_arg_group.add_argument('--data', required=True, type=str, choices=['KTH', 'UCF', 'HMDB51', 'S1M'],
                                         help='name of training dataset')
        data_load_arg_group.add_argument('--backwards', default=True, type=bool,
                                         help='Flag to augment the dataset with videos played backwards')
        data_load_arg_group.add_argument('--pick_mode', default='Random', type=str,
                                         choices=['Random', 'First', 'Slide'],
                                         help='How to select video clips from the dataset. "Random" selects by '
                                              'randomly seeking within the current video; "First" selects the first '
                                              'full clip in the video; and "Slide" selects all clips via a sliding '
                                              'window')
        data_load_arg_group.add_argument('--flip', default=True, type=bool,
                                         help='Flag to augment the dataset by flipping video frames horizontally')
        data_load_arg_group.add_argument('--dataroot', required=True, help='Path to full video files')
        data_load_arg_group.add_argument('--textroot', required=True,
                                         help='Path to text files that define videos for training, validation and '
                                              'testing')
        data_load_arg_group.add_argument('--video_list', type=str, default=None,
                                         help='The name of the text file containing the video list. If not defined, '
                                              'default names for training/val/test are used')
        data_load_arg_group.add_argument('--nThreads', type=int, default=2, help='Number of threads used to load data')
        data_load_arg_group.add_argument('--skip', type=int, default=1,
                                         help='Number of real frames to skip when sampling clips from a real video')

    def parse(self):
        opt = self.parser.parse_args()

        if len(opt.image_size) == 1:
            a = opt.image_size[0]
            opt.image_size.append(a)

        # Check that at least one GPU is available and specified
        assert(torch.cuda.is_available())
        assert(len(opt.gpu_ids) > 0)

        return opt