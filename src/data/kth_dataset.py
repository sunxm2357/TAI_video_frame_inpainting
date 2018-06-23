import os
from warnings import warn

import numpy as np

from .base_dataset import BaseDataset


class KthDataset(BaseDataset):
    def name(self):
        return 'KthDataset'

    def __getitem__(self, index):
        """Obtain data associated with a video clip from the dataset."""

        # Try to use the given video to extract clips. If it's too short, get a random video instead
        while True:
            # Extract line for current video. Format: <file name prefix> <first frame index> <last frame index>
            line = self.files[index]

            # Get video name prefix, clip start and clip end
            file_name_prefix, first_frame_index, last_frame_index = line.split()
            # Open video
            vid_path = os.path.join(self.dataroot, file_name_prefix + '_uncomp.avi')
            vid = self.open_video(vid_path)

            if vid is not None:
                # Get valid frame index range for current video
                low = int(first_frame_index)
                high = min([int(last_frame_index), vid.get_length()]) - self.seq_len * self.skip
                if high >= low:
                    # We found a valid video clip, so break loop
                    break

            # Video could not be opened or the length was too short, so try another video
            index = np.random.randint(0, len(self.files))

        # Set a new name for each clip
        vid_name = '%s_%s_%s' % (file_name_prefix, first_frame_index, last_frame_index)

        # get the index for the first frame
        stidx = self.start_point(low, high)

        # read in a sequence or sequences for 'Slide'
        if self.pick_mode == 'Slide':
            # for jogging and running case, define large jump
            input_data = []
            action = vid_path.split('_')[1]
            if action in ['running', 'jogging']:
                n_skip = 3
            else:
                n_skip = self.T
            for j in range(low, high, n_skip):
                try:
                    seq = self.read_seq(vid, j + 5 - self.K, vid_name, vid_path)
                except IndexError:
                    warn('Skipping invalid sequence starting at frame %d in %s' % (j + 5 - self.K, vid_name))
                input_data.append(seq)
        else:
            input_data = self.read_seq(vid, stidx, vid_name, vid_path)

        return input_data
