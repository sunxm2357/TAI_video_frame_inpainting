from util.util import listopt
from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()

        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--output_both_directions', action='store_true', help='whether to store intermediate predictions')

    def parse(self):
        opt = BaseOptions.parse(self)

        opt.serial_batches = True
        opt.flip = False
        opt.backwards = False
        opt.video_list = 'test_data_list.txt'
        opt.test_name = opt.name + '_' + str(opt.K) + '_' + str(opt.T) + '_' + opt.which_epoch

        listopt(opt)

        return opt
