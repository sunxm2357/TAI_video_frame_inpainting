import os
from warnings import warn

import numpy as np

from .base_dataset import BaseDataset


class S1MDataset(BaseDataset):
    def name(self):
        return 'S1MDataset'

    def __getitem__(self, index):
        """Obtain data associated with a video clip from the dataset."""

        # Try to use the given video to extract clips. If it's too short, get a random video instead
        while True:
            # Extract line for current video. Format: <file name> <last readable frame index>
            line = self.files[index]

            # Get the video path and last readable frame index
            vid_name, last_readable_index = line.split()
            # Open video
            vid_path = os.path.join(self.dataroot, vid_name)
            vid = self.open_video(vid_path)

            if vid is not None:
                # Get valid frame index range for current video
                low = 1
                high = min(vid.get_length(), int(last_readable_index)) - self.seq_len * self.skip
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
