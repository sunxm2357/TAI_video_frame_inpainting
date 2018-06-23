import torch.utils.data

from .kth_dataset import KthDataset
from .ucf_dataset import UcfDataset
from .s1m_dataset import S1MDataset
from .hmdb_dataset import HMDBDataset


def create_dataset(data, c_dim, dataroot, textroot, video_list, K, T, backwards, flip, pick_mode, image_size,
                   include_following, skip, F):
    if data == 'KTH':
        dataset = KthDataset(c_dim, dataroot, textroot, video_list, K, T, backwards, flip, pick_mode, image_size,
                             include_following, skip, F)
    elif data == 'UCF':
        dataset = UcfDataset(c_dim, dataroot, textroot, video_list, K, T, backwards, flip, pick_mode, image_size,
                             include_following, skip, F)
    elif data == 'S1M':
        dataset = S1MDataset(c_dim, dataroot, textroot, video_list, K, T, backwards, flip, pick_mode, image_size,
                             include_following, skip, F)
    elif data == 'HMDB51':
        dataset = HMDBDataset(c_dim, dataroot, textroot, video_list, K, T, backwards, flip, pick_mode, image_size,
                              include_following, skip, F)
    else:
        raise ValueError('Dataset [%s] not recognized.' % data)

    print('dataset [%s] was created' % (dataset.name()))
    return dataset

class CustomDataLoader:
    def name(self):
        return 'CustomDataLoader'

    def __init__(self, data, c_dim, dataroot, textroot, video_list, K, T, backwards, flip, pick_mode, image_size,
                 include_following, skip, F, batch_size, serial_batches, nThreads):
        self.dataset = create_dataset(data, c_dim, dataroot, textroot, video_list, K, T, backwards, flip, pick_mode,
                                      image_size, include_following, skip, F)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=not serial_batches,
                                                      num_workers=nThreads, drop_last=True)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
