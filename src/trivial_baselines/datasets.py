import cv2
import os

import imageio
import numpy as np


def create_dataset(opt):
    if opt.data == 'KTH':
        dataset = KthDataset(opt)
    elif opt.data == 'UCF':
        dataset = UcfDataset(opt)
    elif opt.data == 'HMDB51':
        dataset = HMDBDataset(opt)
    else:
        raise ValueError('Dataset [%s] not recognized.' % opt.data)
    print('dataset [%s] was created' % (dataset.name()))
    return dataset


class KthDataset(object):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.textroot = opt.textroot
        f = open(os.path.join(self.textroot, opt.video_list), 'r')
        self.files = f.readlines()
        self.K = opt.K
        self.T = opt.T
        self.F = opt.F
        self.pick_mode = opt.pick_mode
        self.image_size = opt.image_size
        self.seq_len = self.K + self.T + self.F

    def __len__(self):
        return len(self.files)

    def name(self):
        return 'KthDataset'

    def read_seq(self, vid, stidx, tokens):
        targets = []
        for t in range(self.seq_len):
            while True:
                try:
                    img = cv2.cvtColor(cv2.resize(vid.get_data(stidx + t), (self.image_size[1], self.image_size[0])),
                                       cv2.COLOR_RGB2GRAY)
                    break
                except Exception:
                    print('in cv2', self.vid_path, stidx+t)
                    print('imageio failed loading frames, retrying')
            assert (np.max(img) > 1, 'the range of image should be [0,255]')
            if len(img.shape) == 2: img = np.expand_dims(img, axis=2)

            targets.append(img.copy())

        target = np.stack(targets, axis=-1)

        return {'targets': target, 'video_name': '%s_%s_%s' % (tokens[0], tokens[1], tokens[2]),
                  'start-end':  '%d-%d' % (stidx, stidx + self.seq_len - 1)}

    def __getitem__(self, index):
        tokens = self.files[index].split()
        # with open(self.log_name, 'a') as log_file:
        #     log_file.write(tokens[0])
        # print(tokens[0])
        vid_path = os.path.join(self.root, tokens[0]+'_uncomp.avi')
        self.vid_path = vid_path
        while True:
            try:
                vid = imageio.get_reader(vid_path,'ffmpeg')
                break
            except Exception:
                print(vid_path)
                print('imageio failed loading frames, retrying')
        low = int(tokens[1])
        high = min([int(tokens[2]), vid.get_length()]) - self.seq_len
        assert(high >= low, 'the video is not qualified')
        if self.pick_mode == 'Random':
            if low == high:
                stidx = low
            else:
                stidx = np.random.randint(low=low, high=high)
        elif self.pick_mode == 'First':
            stidx = low
        elif self.pick_mode == 'Slide':
            stidx = low
        else:
            raise NotImplementedError('pick_mode method [%s] is not implemented' % self.pick_mode)

        if not self.pick_mode == 'Slide':
            input_data = self.read_seq(vid, stidx, tokens)
        else:
            input_data = []
            action = vid_path.split('_')[1]
            if action in ['running', 'jogging']:
                n_skip = 3
            else:
                n_skip = self.T
            for j in range(low, high, n_skip):
                input_data.append(self.read_seq(vid, j, tokens))

        return input_data


class UcfDataset(object):
    def __init__(self, opt):
        self.opt= opt
        self.root = opt.dataroot
        self.textroot = opt.textroot
        f = open(os.path.join(self.textroot, opt.video_list), 'r')
        self.files = f.readlines()
        self.K = opt.K
        self.T = opt.T
        self.F = opt.F
        self.pick_mode = opt.pick_mode
        self.image_size = opt.image_size
        self.seq_len = self.K + self.T + self.F

    def __len__(self):
        return len(self.files)

    def name(self):
        return 'UcfDataset'

    def read_seq(self, vid, stidx, vid_name):
        targets = []
        for t in range(self.seq_len):
            while True:
                try:
                    img = cv2.resize(vid.get_data(stidx + t), (self.image_size[1], self.image_size[0]))[:, :, ::-1]
                    break
                except Exception:
                    print('in cv2', self.vid_path, stidx+t)
                    print('imageio failed loading frames, retrying')
            assert (np.max(img) > 1, 'the range of image should be [0,255]')
            if len(img.shape) == 2: img = np.expand_dims(img, axis=2)
            targets.append(img.copy())

        target = np.stack(targets, axis=-1)

        return {'targets': target, 'video_name': '%s' % (vid_name),
                  'start-end':  '%d-%d' % (stidx, stidx + self.seq_len - 1)}

    def __getitem__(self, index):
        self.files[index] = self.files[index].replace('/HandStandPushups/',
                                            '/HandstandPushups/')
        vid_name = self.files[index].split()[0]
        vid_path = os.path.join(self.root, vid_name)
        self.vid_path = vid_path
        while True:
            try:
                vid = imageio.get_reader(vid_path,'ffmpeg')
                break
            except Exception:
                print(vid_path)
                print('imageio failed loading frames, retrying')

        low = 1
        high = vid.get_length() - self.seq_len
        assert(high >= low, 'the video is not qualified')
        if self.pick_mode == 'Random':
            if low == high:
                stidx = low
            else:
                stidx = np.random.randint(low=low, high=high)
        elif self.pick_mode == 'First':
            stidx = low
        elif self.pick_mode == 'Slide':
            stidx = low
        else:
            raise NotImplementedError('pick_mode method [%s] is not implemented' % self.pick_mode)

        if not self.pick_mode == 'Slide':
            input_data = self.read_seq(vid, stidx, vid_name)
        else:
            input_data = []
            n_skip = self.T
            for j in range(low, high, n_skip):
                input_data.append(self.read_seq(vid, j, vid_name))

        return input_data


class HMDBDataset(object):
    def __init__(self, opt):
        self.opt= opt
        self.root = opt.dataroot
        self.textroot = opt.textroot
        f = open(os.path.join(self.textroot, opt.video_list), 'r')
        self.files = f.readlines()
        self.K = opt.K
        self.T = opt.T
        self.F = opt.F
        self.pick_mode = opt.pick_mode
        self.image_size = opt.image_size
        self.seq_len = self.K + self.T + self.F

    def __len__(self):
        return len(self.files)

    def name(self):
        return 'HMDBDataset'

    def read_seq(self, vid, stidx, vid_name):
        targets = []
        for t in range(self.seq_len):
            while True:
                try:
                    img = cv2.resize(vid.get_data(stidx + t), (self.image_size[1], self.image_size[0]))[:, :, ::-1]
                    break
                except Exception:
                    raise ValueError('in cv2', self.vid_path, stidx+t)
                    print('imageio failed loading frames, retrying')
            assert (np.max(img) > 1, 'the range of image should be [0,255]')
            if len(img.shape) == 2: img = np.expand_dims(img, axis=2)
            targets.append(img.copy())

        target = np.stack(targets, axis=-1)

        return {'targets': target, 'video_name': '%s' % (vid_name),
                'start-end':  '%d-%d' % (stidx, stidx + self.seq_len - 1)}

    def __getitem__(self, index):
        vid_name = self.files[index]
        if vid_name.endswith('\n'):
            vid_name = vid_name[:-1]
        vid_path = os.path.join(self.root, vid_name)
        self.vid_path = vid_path
        while True:
            try:
                vid = imageio.get_reader(vid_path, 'ffmpeg')
                break
            except Exception:
                raise ValueError('imageio failed in loading vidoe %s, retrying' % vid_path)
                print('imageio failed in loading vidoe %s, retrying' % vid_path)

        low = 1
        high = vid.get_length() - self.seq_len
        assert(high >= low, 'the video is not qualified')
        if self.pick_mode == 'Random':
            if low == high:
                stidx = low
            else:
                stidx = np.random.randint(low=low, high=high)
        elif self.pick_mode == 'First':
            stidx = low
        elif self.pick_mode == 'Slide':
            stidx = low
        else:
            raise NotImplementedError('pick_mode method [%s] is not implemented' % self.pick_mode)

        if not self.pick_mode == 'Slide':
            input_data = self.read_seq(vid, stidx, vid_name)
        else:
            input_data = []
            n_skip = self.T
            for j in range(low, high, n_skip):
                input_data.append(self.read_seq(vid, j, vid_name))

        return input_data
