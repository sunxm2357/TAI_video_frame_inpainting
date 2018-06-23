import os
import random
import traceback
from warnings import warn

import cv2
import imageio
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms.functional import to_tensor

from util.util import fore_transform


class BaseDataset(data.Dataset):
    def name(self):
        return 'BaseDataset'

    def __init__(self, c_dim, dataroot, textroot, video_list, K, T, backwards, flip, pick_mode, image_size,
                 include_following, skip, F):
        """Constructor

        :param c_dim: The number of color channels each output video should have
        :param dataroot: The root folder containing all videos in the dataset
        :param textroot: The root folder containing all video list text files for the dataset
        :param video_list: The name of the video list text file
        :param K: The number of preceding frames
        :param T: The number of future or middle frames
        :param backwards: Flag to allow data augmentation by randomly reversing videos temporally
        :param flip: Flag to allow data augmentation by randomly flipping videos horizontally
        :param pick_mode: How to grab clips from each video in the dataset. "Slide" gives all clips obtained by a
                          sliding window over the video, "First" gives the first clip in the video, and "Random"
                          randomly chooses a clip in the video.
        :param image_size: The spatial resolution of the video (W x H)
        :param include_following: Whether to generate following frame info (specifically, following frame differences)
        :param skip: How many frames to skip when grabbing each frame from a video (i.e. controls temporal resolution)
        :param F: The number of following frames
        """

        super(BaseDataset, self).__init__()

        # Check for valid pick mode
        if pick_mode not in ['Slide', 'First', 'Random']:
            raise NotImplementedError('pick_mode option "%s" is not supported' % pick_mode)

        # Initialize basic properties
        self.c_dim = c_dim
        self.dataroot = dataroot
        self.K = K
        self.T = T
        self.backwards = backwards
        self.flip = flip
        self.pick_mode = pick_mode
        self.image_size = image_size
        self.include_following = include_following
        self.skip = skip

        # Read the list of files
        with open(os.path.join(textroot, video_list), 'r') as f:
            self.files = [line.rstrip() for line in f.readlines()]

        # Disable data augmentation if using sliding window or picking first clip only
        if pick_mode in ['Slide', 'First']:
            self.backwards = False
            self.flip = False

        self.seq_len = K + T
        if include_following:
            self.F = F
            self.seq_len += F

        self.vid_path = None

    def __getitem__(self, index):
        """Obtain data associated with a video clip from the dataset."""

        # Try to use the given video to extract clips. If it's too short, get a random video instead
        while True:
            # Extract video name
            vid_name = self.files[index]
            vid_path = os.path.join(self.dataroot, vid_name)
            # Open the video
            vid = self.open_video(vid_path)

            if vid is not None:
                # Get the minimum and maximum index for the first frame
                low = 1
                high = vid.get_length() - self.seq_len * self.skip
                if high >= low:
                    # We found a valid video clip, so break loop
                    break

            # Video could not be opened or the length was too short, so try another video
            index = np.random.randint(0, len(self.files))

        # Get the index for the first frame
        stidx = self.start_point(low, high)

        # Read in a sequence or sequences for 'Slide'
        if self.pick_mode == 'Slide':
            input_data = []
            n_skip = self.T
            for j in range(low, high, n_skip):
                try:
                    seq = self.read_seq(vid, j, vid_name, vid_path)
                except IndexError:
                    warn('Skipping invalid sequence starting at frame %d in %s' % (j, vid_name))
                input_data.append(seq)
        else:
            input_data = self.read_seq(vid, stidx, vid_name, vid_path)

        return input_data

    def __len__(self):
        """Return the number of videos in the dataset."""
        return len(self.files)

    def open_video(self, vid_path):
        """Obtain a file reader for the video at the given path.

        Wraps the line to obtain the reader in a while loop. This is necessary because it fails randomly even for
        readable videos.

        :param vid_path: The path to the video file
        """
        num_attempts = 0
        while num_attempts < 5:
            try:
                vid = imageio.get_reader(vid_path, 'ffmpeg')
                return vid
            except IOError:
                traceback.print_exc()
                warn('imageio failed in loading video %s, retrying' % vid_path)
                num_attempts += 1

        warn('Failed to load video %s after multiple attempts, returning' % vid_path)
        return None

    def start_point(self, low, high):
        """Select the starting frame index within a given range.

        The selected start point is random only if the pick_mode is "Random". Otherwise, it is just the lowest index.

        :param low: The lowest possible index (inclusive)
        :param high: The highest possible index (exclusive)
        """
        if self.pick_mode == 'Random':
            if low == high:
                stidx = low
            else:
                stidx = np.random.randint(low=low, high=high)
        else:
            stidx = low

        return stidx

    def read_seq(self, vid, stidx, vid_name, vid_path):
        """Obtain a video clip along with corresponding difference frames and auxiliary information.

        Returns a dict with the following key-value pairs:
        - vid_name: A string identifying the video that the clip was extracted from
        - start-end: The start and end indexes of the video frames that were extracted (inclusive)
        - targets: The full video clip [C x H x W x T FloatTensor]
        - diff_in: The difference frames of the preceding frames in the full video clip [C x H x W x T_P]
        - diff_in_F: The difference frames of the following frames in the full video clip [C x H x W x T_F]

        :param vid: An imageio Reader
        :param stidx: The index of the first video frame to extract clips from
        :param vid_name: A string identifying the video that the clip was extracted from
        :param vid_path: The path to the given video file
        """

        targets = []
        gray_imgs = []

        # generate [0, 1] random variable to determine flip and backward
        flip_flag = self.flip and (random.random() > 0.5)
        back_flag = self.backwards and (random.random() > 0.5)

        # read and process in each frame
        for t in range(self.seq_len):
            # read in one frame from the video
            while True:
                try:
                    vid_frame = vid.get_data(stidx + t * self.skip)
                    break
                except IndexError as e:
                    raise e
                except:
                    traceback.print_exc()
                    print('in cv2', vid_path, stidx + t * self.skip)
                    print('imageio failed loading frames, retrying')

            # resize frame
            img = cv2.resize(vid_frame, (self.image_size[1], self.image_size[0]))[:, :, ::-1]
            # calculate the gray image
            gray_img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)

            # expand one dimension for channel if img is gray scale
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            if gray_img.ndim == 2:
                gray_img = np.expand_dims(gray_img, axis=2)

            # flip the input frame horizontally
            if flip_flag:
                img = img[:, ::-1, :]
                gray_img = gray_img[:, ::-1, :]

            # if # channel is 1, use the gray scale image as input
            if self.c_dim == 1:
                targets.append(to_tensor(gray_img.copy()))
            else:
                targets.append(to_tensor(img.copy()))
            gray_imgs.append(gray_img)

        # Reverse the temporal ordering of frames
        if back_flag:
            targets = targets[::-1]
            gray_imgs = gray_imgs[::-1]

        # stack frames and map [0, 255] to [-1, 1]
        target = fore_transform(torch.stack(targets, dim=-1))

        # calculate the difference img for the preceding frames
        diff_ins = []
        for t in range(1, self.K):
            prev = gray_imgs[t - 1] / 255.
            next = gray_imgs[t] / 255.
            diff_ins.append(torch.from_numpy(np.transpose(next - prev, axes=(2, 0, 1)).copy()).float())
        diff_in = torch.stack(diff_ins, dim=-1)

        ret = {
            'targets': target,
            'diff_in': diff_in,
            'video_name': vid_name,
            'start-end': '%d-%d' % (stidx, stidx + self.seq_len * self.skip - 1)
        }

        # calculate the difference img for the following frames
        if self.include_following:
            diff_ins_F = []
            for t in range(self.seq_len - 2, self.K + self.T - 1, -1):
                prev = gray_imgs[t + 1] / 255.
                next = gray_imgs[t] / 255.
                diff_ins_F.append(torch.from_numpy(np.transpose(next - prev, axes=(2, 0, 1)).copy()).float())
            diff_in_F = torch.stack(diff_ins_F, dim=-1)
            ret['diff_in_F'] = diff_in_F

        return ret