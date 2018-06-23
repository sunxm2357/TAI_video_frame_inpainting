"""
Script to run evaluation on trivial baselines for video frame inpainting (e.g. copying a preceding or following frame
repeatedly).
"""

import os

import numpy as np
import skimage.measure as measure
from skimage.measure import compare_ssim as ssim

from trivial_baselines.datasets import create_dataset
from trivial_baselines.options import TestOptions
from util.util import draw_err_plot, refresh_donelist


def metrics(seq_batch, opt, true_data, pred_data,  psnr_err, ssim_err, multichannel=True):
    pred_data = np.concatenate((seq_batch[:, :, :, :opt.K], pred_data, seq_batch[:, :, :, opt.K + opt.T:]),  axis=3)
    true_data = np.concatenate((seq_batch[:, :, :, :opt.K], true_data, seq_batch[:, :, :, opt.K + opt.T:]), axis=3)
    seq_len = opt.K + opt.T + opt.F

    cpsnr = np.zeros((seq_len,))
    cssim = np.zeros((seq_len,))

    for t in range(seq_len):
        pred = pred_data[:, :, :, t].astype('uint8')
        target = true_data[:, :, :, t].astype('uint8')
        if opt.c_dim == 1:
            pred = np.squeeze(pred, axis=-1)
            target = np.squeeze(target, axis=-1)
        cpsnr[t] = measure.compare_psnr(pred, target)
        cssim[t] = ssim(target, pred, multichannel=multichannel)
    psnr_err = np.concatenate((psnr_err, cpsnr[None, opt.K:opt.K + opt.T]), axis=0)
    ssim_err = np.concatenate((ssim_err, cssim[None, opt.K:opt.K + opt.T]), axis=0)
    return psnr_err, ssim_err


def main():
    save_freq = 50
    opt = TestOptions().parse()
    if opt.data == 'KTH':
        lims_ssim = [1, opt.T, 0.6, 1]
        lims_psnr = [1, opt.T, 20, 34]
    elif opt.data in ['UCF', 'HMDB51']:
        lims_ssim = [1, opt.T, 0.3, 1]
        lims_psnr = [1, opt.T, 10, 35]
    else:
        raise ValueError('Dataset [%s] not recognized.' % opt.data)
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('# testing videos = %d' % dataset_size)

    psnr_err = np.zeros((0, opt.T))
    ssim_err = np.zeros((0, opt.T))
    multichannel = not (opt.c_dim == 1)

    for i in range(dataset_size):
        print('dealing %d/%d'%(i, dataset_size))
        datas = dataset[i]
        if opt.pick_mode == 'First': datas = [datas]
        for data in datas:
            seq_batch = data['targets'] #[h,w,c,idx]#
            last_preceding = seq_batch[:, :, :, opt.K-1]
            first_following = seq_batch[:, :, :, opt.K + opt.T]
            pred = []
            for t in range(opt.T):
                if opt.comb_type == 'repeat_P':
                    pred.append(last_preceding.copy())
                elif opt.comb_type == 'repeat_F':
                    pred.append(first_following.copy())
                elif opt.comb_type == 'avg':
                    frame = 0.5 * first_following.copy() + 0.5 * last_preceding.copy()
                    frame = frame.clip(0, 255)
                    pred.append(frame)
                elif opt.comb_type == 'w_avg':
                    w = float(t+1) / (opt.T + 1)
                    # print(w)
                    frame = w * first_following.copy() + (1-w) * last_preceding.copy()
                    frame = frame.clip(0, 255)
                    pred.append(frame)
                else:
                    raise ValueError('combination method [%s] not recognized.' % opt.comb_type)
            pred_data = np.stack(pred, axis=-1)
            true_data = seq_batch[:, :, :, opt.K: opt.K + opt.T]
            psnr_err, ssim_err = metrics(seq_batch, opt, true_data, pred_data, psnr_err, ssim_err, multichannel=multichannel)
        if i % (save_freq) == 0:
            print('psnr:', psnr_err.mean(axis=0))
            print('ssim:', ssim_err.mean(axis=0))
            save_path = os.path.join(opt.quant_dir, 'results.npz')
            np.savez(save_path, psnr=psnr_err, ssim=ssim_err)
            psnr_plot = os.path.join(opt.quant_dir, 'psnr_final.png')
            draw_err_plot(psnr_err, 'Peak Signal to Noise Ratio', lims_psnr, path=psnr_plot)
            ssim_plot = os.path.join(opt.quant_dir, 'ssim_final.png')
            draw_err_plot(ssim_err, 'Structural Similarity', lims_ssim, path=ssim_plot)

    refresh_donelist(opt.data, opt.comb_type, 'trivial', None, opt.K, opt.T, opt.test_name, True)
    print('Done.')


if __name__ == '__main__':
    main()

