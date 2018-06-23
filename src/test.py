import cv2
import os
from os import system

import numpy as np
import skimage.measure as measure
from skimage.measure import compare_ssim as ssim

from data.data_loader import CustomDataLoader
from train_environments.create_environment import create_environment
from options.test_options import TestOptions
from util.util import inverse_transform, makedir, draw_frame, draw_err_plot, refresh_donelist, listopt, to_numpy


def create_video(frames, savedir, img_prefix, video_name, fps, acc=6):
    """Save the given frames as a GIF."""

    print('creating video: %s'% os.path.join(savedir, video_name))
    # Save all frames to disk
    name = img_prefix + '_%0{}d.png'.format(str(acc))
    for c, frame in enumerate(frames):
        cv2.imwrite(os.path.join(savedir, name % c), frame)

    # Remove existing gif file
    cmd1 = 'rm ' + os.path.join(savedir, video_name)
    # Encode frames into gif
    cmd2 = ('ffmpeg -framerate %d -i ' % (fps) + os.path.join(savedir, name) + ' -pix_fmt yuv420p ' + os.path.join(savedir, video_name))
    # Remove all frames
    cmd3 = 'rm ' + os.path.join(savedir, img_prefix + '*.png')

    # Run commands
    system(cmd1)
    system(cmd2)
    system(cmd3)


def eval(seq_batch, K, T, img_dir, pred_data, video_name, start_end, psnr_err, ssim_err, type_):
    """Evaluate the quality of the predicted frames, and store visualizations.

    :param seq_batch: A batch of ground-truth videos (B x K+T+K x H x W x C np.ndarray with values in [-1, 1])
    :param K: The number of preceding frames (and following frames)
    :param T: The number of middle frames
    :param img_dir: The root folder where qualitative results for this experiment will be stored
    :param pred_data: The predicted future or middle frames (B x T x H x W x C np.ndarray with values in [-1, 1])
    :param video_name: A list of video sources corresponding to each video in seq_batch
    :param start_end: A list of strings indicating the indexes of the ground-truth frames in the video
    :param psnr_err: The previously computed PSNR values (N x T np.ndarray)
    :param ssim_err: The previously computed SSIM values (N x T np.ndarray)
    :param type_: Which kind of prediction is being evaluated/visualized (i.e. "forward", "backward", or "final") (str)
    """

    # prepare gt and preds
    batch_size = seq_batch.shape[0]
    c_dim = seq_batch.shape[-1]
    multichannel = (c_dim != 1)

    true_data = seq_batch.copy()
    tmp = seq_batch.copy()
    tmp[:, K: K + T] = pred_data
    pred_data = tmp

    seq_len = seq_batch.shape[1]

    pred_data = pred_data.clip(-1, 1)
    true_data = true_data.clip(-1, 1)

    # qualitative results
    for b in range(batch_size):
        # Create a folder to store the visualizations for the current video
        savedir = os.path.join(img_dir, video_name[b].split('.')[0] + '_' + start_end[b])
        makedir(savedir)

        # create prediction gif
        preds = []
        for t in range(seq_len):
            pred = (inverse_transform(pred_data[b, t]) * 255).astype('uint8')
            pred = draw_frame(pred, t < K or t >= (K + T))
            preds.append(pred)
        create_video(preds, savedir, 'pred_{}'.format(type_), 'pred_{}.gif'.format(type_), 3, acc=4)

        # create gt gif
        if type_ == 'final':
            gts = []
            for t in range(seq_len):
                target = (inverse_transform(true_data[b, t]) * 255).astype('uint8')
                target = draw_frame(target, t < K or t >= (K + T))
                gts.append(target)
            create_video(gts, savedir, 'gt', 'gt.gif', 3, acc=4)

    # squeeze dimension for gray-scale
    if c_dim == 1:
        pred_data = np.squeeze(pred_data, axis=-1)
        true_data = np.squeeze(true_data, axis=-1)

    # quantitative eval
    for b in range(batch_size):
        cpsnr = np.zeros((T,))
        cssim = np.zeros((T,))
        for c, t in enumerate(range(K, K + T)):
            pred = (inverse_transform(pred_data[b, t]) * 255).astype('uint8')
            target = (inverse_transform(true_data[b, t]) * 255).astype('uint8')
            cpsnr[c] = measure.compare_psnr(pred, target)
            cssim[c] = ssim(target, pred, multichannel=multichannel)

        psnr_err = np.concatenate((psnr_err, cpsnr[None, :]), axis=0)
        ssim_err = np.concatenate((ssim_err, cssim[None, :]), axis=0)

    return psnr_err, ssim_err


def save_quant_result(quant_dir, output_both_directions, psnr_err, ssim_err, lims_psnr, lims_ssim, pred_types):
    """Store quantitative results to disk and also plot them.

    :param quant_dir: The root directory where this experiment's quantitative results will be stored
    :param output_both_directions: Whether to store the errors of intermediate predictions
    :param psnr_err: A dict containing the PSNR errors of each type of prediction (each value is a N x T np.ndarray)
    :param ssim_err: A dict containing the SSIM errors of each type of prediction (each value is a N x T np.ndarray)
    :param lims_psnr: The axis limits to use when printing the PSNR plots
    :param lims_ssim: The axis limits to use when printing the SSIM plots
    :param pred_types: A list of strings indicating which kind of prediction errors to plot
    """

    # save npz
    save_path = os.path.join(quant_dir, 'results.npz')
    if output_both_directions:
        np.savez(save_path, psnr=psnr_err['final'], ssim=ssim_err['final'], psnr_forward=psnr_err['forward'],
                 ssim_forward=ssim_err['forward'], psnr_backward=psnr_err['backward'],
                 ssim_backward=ssim_err['backward'])
    else:
        np.savez(save_path, psnr=psnr_err['final'], ssim=ssim_err['final'])

    # save plots
    for pred_type in pred_types:
        psnr_plot = os.path.join(quant_dir, 'psnr_%s.png' % (pred_type))
        draw_err_plot(psnr_err[pred_type], 'Peak Signal to Noise Ratio', lims_psnr, path=psnr_plot)
        ssim_plot = os.path.join(quant_dir, 'ssim_%s.png' % (pred_type))
        draw_err_plot(ssim_err[pred_type], 'Structural Similarity', lims_ssim, path=ssim_plot)


def main():
    save_freq = 500

    # parse test arguments and create directories
    opt = TestOptions().parse()
    file_name = os.path.join(opt.checkpoints_dir, opt.name, 'test_opt.txt')
    with open(file_name, 'wt') as opt_file:
        listopt(opt, opt_file)
    opt.quant_dir = os.path.join(opt.result_dir, 'quantitative', opt.data, opt.name + '_' + str(opt.K) + '_' + str(
                                opt.T) + '_' + opt.which_update)
    makedir(opt.quant_dir)
    opt.img_dir = os.path.join(opt.result_dir, 'images', opt.data, opt.name + '_' + str(opt.K) + '_' + str(
                                opt.T) + '_' + opt.which_update)
    makedir(opt.img_dir)

    # preparation for quant eval
    if opt.data == 'KTH':
        lims_ssim = [1, opt.T, 0.6, 1]
        lims_psnr = [1, opt.T, 20, 34]
    elif opt.data in ['UCF', 'HMDB51', 'S1M']:
        lims_ssim = [1, opt.T, 0.3, 1]
        lims_psnr = [1, opt.T, 10, 35]
    else:
        raise ValueError('Dataset [%s] not recognized.' % opt.data)
    psnr_err = {'final': np.zeros((0, opt.T)), 'forward': np.zeros((0, opt.T)), 'backward': np.zeros((0, opt.T))}
    ssim_err = {'final': np.zeros((0, opt.T)), 'forward': np.zeros((0, opt.T)), 'backward': np.zeros((0, opt.T))}

    # preparation for qual eval
    if opt.output_both_directions:
        pred_types = ['final', 'forward', 'backward']
    else:
        pred_types = ['final']

    # create dataloader
    include_following = (opt.model_type != 'mcnet')
    data_loader = CustomDataLoader(opt.data, opt.c_dim, opt.dataroot, opt.textroot, opt.video_list, opt.K, opt.T,
                                   opt.backwards, opt.flip, opt.pick_mode, opt.image_size, include_following,
                                   opt.skip, opt.F, opt.batch_size, opt.serial_batches, opt.nThreads)
    print(data_loader.name())
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('# testing videos = %d' % dataset_size)

    # create model
    env = create_environment(opt.model_type, opt.gf_dim, opt.c_dim, opt.gpu_ids, False, opt.checkpoints_dir,
                             opt.name, opt.K, opt.T, opt.F, opt.image_size, opt.batch_size, opt.which_update, opt.comb_type,
                             opt.shallow, opt.ks, opt.num_block, opt.layers, opt.kf_dim, opt.enable_res, opt.rc_loc)

    # evaluate each video
    for i, datas in enumerate(dataset):
        # unify the data format for different pick up modes
        if opt.pick_mode == 'First':
            datas = [datas]
        for data in datas:
            # prepare the ground truth in the form of [batch, f, h, w, c]
            seq_batch = to_numpy(data['targets'], (0, 4, 2, 3, 1))

            # extract the test case's information
            video_name = data['video_name']
            start_end = data['start-end']

            # compute the inpainting results
            env.set_inputs(data)
            env.forward()

            # list prediction results
            if opt.model_type != 'mcnet':
                preds = {'final': env.pred, 'forward': env.pred_forward, 'backward': env.pred_backward}
            else:
                preds = {'final': env.pred}

            # quant and qual analysis of each result
            for pred_type in pred_types:
                pred = [to_numpy(a.data, (0, 2, 3, 1)) for a in preds[pred_type]]
                pred_data = np.stack(pred, axis=1)
                psnr_err[pred_type], ssim_err[pred_type] = eval(seq_batch, opt.K, opt.T, opt.img_dir, pred_data,
                                                                video_name, start_end, psnr_err[pred_type],
                                                                ssim_err[pred_type], pred_type)

        if i % save_freq == 0:
            save_quant_result(opt.quant_dir, opt.output_both_directions, psnr_err, ssim_err, lims_psnr, lims_ssim,
                              pred_types)

    save_quant_result(opt.quant_dir, opt.output_both_directions, psnr_err, ssim_err, lims_psnr, lims_ssim, pred_types)

    refresh_donelist(opt.data, opt.comb_type, opt.model_type, opt.rc_loc, opt.K, opt.T, opt.test_name)
    print('Done.')


if __name__ == '__main__':
    main()
