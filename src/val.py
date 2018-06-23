import numpy as np
import skimage.measure as measure
import torch
from skimage.measure import compare_ssim as ssim

from data.data_loader import CustomDataLoader
from train_environments.create_environment import create_environment
from util.util import inverse_transform, draw_err_plot, visual_grid, to_numpy


def val(c_dim, data, T, dataroot, textroot, video_list, K, backwards, flip, pick_mode, image_size, gpu_ids, model_type, skip,
        F, batch_size, serial_batches, nThreads, gf_dim, is_train, checkpoints_dir, name, no_adversarial, alpha, beta,
        D_G_switch, margin, lr, beta1, sn, df_dim, Ip, comb_type, comb_loss, shallow, ks, num_block, layers, kf_dim,
        enable_res, rc_loc, continue_train, which_update):
    # *************************** quant
    # preparation for quant eval
    multichannel = (c_dim != 1)
    # Set axis limits for SSIM and PSNR plots
    if data == 'KTH':
        lims_ssim = [1, T, 0, 1]
        lims_psnr = [1, T, 0, 34]
    elif data in ['UCF', 'HMDB51', 'S1M']:
        lims_ssim = [1, T, 0, 1]
        lims_psnr = [1, T, 0, 35]
    else:
        raise ValueError('Dataset [%s] not recognized.' % data)
    # Initialize PSNR and SSIM tables
    psnr_err = np.zeros((0, T))
    ssim_err = np.zeros((0, T))

    # create dataloader
    include_following = (model_type != 'mcnet')
    data_loader = CustomDataLoader(data, c_dim, dataroot, textroot, video_list, K, T, backwards, flip, pick_mode,
                                   image_size, include_following, skip, F, batch_size, serial_batches, nThreads)
    print(data_loader.name())
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('# val videos = %d' % dataset_size)

    # create model
    env = create_environment(model_type, gf_dim, c_dim, gpu_ids, is_train, checkpoints_dir, name, K, T, F, image_size,
                             batch_size, which_update, comb_type, shallow, ks, num_block, layers, kf_dim, enable_res,
                             rc_loc, no_adversarial, alpha, beta, D_G_switch, margin, lr, beta1, sn, df_dim, Ip,
                             continue_train, comb_loss)

    for datas in dataset:
        if pick_mode == 'First':
            datas = [datas]
        for d in datas:
            # prepare the ground truth in the form of [batch, f, h, w, c]
            seq_batch = to_numpy(d['targets'], (0, 4, 2, 3, 1))

            # compute the inpainting results
            env.set_inputs(d)
            env.forward()

            # Concatenate and clip color intensities in predicted and GT frames
            pred = [to_numpy(a.data, (0, 2, 3, 1)) for a in env.pred]

            pred_data = np.stack(pred, axis=1)
            true_data = seq_batch[:, K:K + T].copy()

            pred_data = pred_data.clip(-1, 1)
            true_data = true_data.clip(-1, 1)

            if c_dim == 1:
                pred_data = np.squeeze(pred_data, axis=-1)
                true_data = np.squeeze(true_data, axis=-1)

            # Compute SSIM and PSNR curves for each video in current batch
            for b in range(batch_size):
                cpsnr = np.zeros((T,))
                cssim = np.zeros((T,))
                # Compute SSIM and PSNR for each frame in current video
                for t in range(T):
                    pred = (inverse_transform(pred_data[b, t]) * 255).astype('uint8')
                    target = (inverse_transform(true_data[b, t]) * 255).astype('uint8')
                    cpsnr[t] = measure.compare_psnr(pred, target)
                    cssim[t] = ssim(target, pred, multichannel=multichannel)
                psnr_err = np.concatenate((psnr_err, cpsnr[None, :]), axis=0)
                ssim_err = np.concatenate((ssim_err, cssim[None, :]), axis=0)

    # Generate plot images
    psnr_plot = draw_err_plot(psnr_err, 'Peak Signal to Noise Ratio', lims_psnr)
    ssim_plot = draw_err_plot(ssim_err, 'Structural Similarity', lims_ssim)

    vis_video_list = 'vis_data_list.txt'
    vis_pick_mode = 'First'

    # Create data loader for qualitative results
    vis_data_loader = CustomDataLoader(data, c_dim, dataroot, textroot, vis_video_list, K, T, backwards, flip,
                                       vis_pick_mode, image_size, include_following, skip, F, batch_size,
                                       serial_batches, nThreads)
    print(vis_data_loader.name())
    vis_dataset = vis_data_loader.load_data()
    vis_dataset_size = len(vis_data_loader)
    print('# visualization videos = %d' % vis_dataset_size)

    # Generate the fill-in predictions and store the visual results
    vis = []
    for d in vis_dataset:
        env.set_inputs(d)
        env.forward()

        visuals = env.get_current_visuals()
        vis.append(visual_grid(visuals, K, T))

    grid = torch.cat(vis, dim=1)
    print('Validation done.')
    return psnr_plot, ssim_plot, grid
