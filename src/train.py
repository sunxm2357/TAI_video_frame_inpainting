import os
import time

from tensorboardX import SummaryWriter

from data.data_loader import CustomDataLoader
from train_environments.create_environment import create_environment
from options.train_options import TrainOptions
from util.util import makedir, listopt, print_current_errors, visual_grid
from val import val


def main():
    opt = TrainOptions().parse()

    # Determine validation step options that might differ from training
    if opt.data == 'KTH':
        val_pick_mode = 'Slide'
        val_gpu_ids = [opt.gpu_ids[0]]
        val_batch_size = 1
    elif opt.data in ['UCF', 'HMDB51', 'S1M']:
        val_pick_mode = 'First'
        val_gpu_ids = opt.gpu_ids
        val_batch_size = opt.batch_size / 2
    else:
        raise ValueError('Dataset [%s] not recognized.' % opt.data)

    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    makedir(expr_dir)
    tb_dir = os.path.join(opt.tensorboard_dir, opt.name)
    makedir(tb_dir)

    file_name = os.path.join(expr_dir, 'train_opt.txt')
    with open(file_name, 'wt') as opt_file:
        listopt(opt, opt_file)

    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    print('after reading options')
    include_following = (opt.model_type != 'mcnet')
    data_loader = CustomDataLoader(opt.data, opt.c_dim, opt.dataroot, opt.textroot, opt.video_list, opt.K, opt.T,
                                   opt.backwards, opt.flip, opt.pick_mode, opt.image_size, include_following,
                                   opt.skip, opt.F, opt.batch_size, opt.serial_batches, opt.nThreads)
    print(data_loader.name())
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('# training videos = %d' % dataset_size)

    env = create_environment(opt.model_type, opt.gf_dim, opt.c_dim, opt.gpu_ids, True, opt.checkpoints_dir,
                             opt.name, opt.K, opt.T, opt.F, opt.image_size, opt.batch_size, opt.which_update, opt.comb_type,
                             opt.shallow, opt.ks, opt.num_block, opt.layers, opt.kf_dim, opt.enable_res, opt.rc_loc,
                             opt.no_adversarial, opt.alpha, opt.beta, opt.D_G_switch, opt.margin, opt.lr, opt.beta1, opt.sn,
                             opt.df_dim, opt.Ip, opt.continue_train, opt.comb_loss)

    total_updates = env.start_update
    writer = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

    while True:
        for data in dataset:
            iter_start_time = time.time()

            # Enable losses on intermediate and final predictions partway through training
            if total_updates >= opt.inter_sup_update:
                env.enable_inter_loss()
            if total_updates >= opt.final_sup_update:
                env.enable_final_loss()

            # Update model
            total_updates += 1
            env.set_inputs(data)
            env.optimize_parameters()

            if total_updates % opt.print_freq == 0:
                errors = env.get_current_errors()
                t = (time.time()-iter_start_time)/opt.batch_size
                writer.add_scalar('iter_time', t, total_updates)
                for key in errors.keys():
                    writer.add_scalar('loss/%s' % (key), errors[key], total_updates)
                print_current_errors(log_name, total_updates, errors, t)

            if total_updates % opt.display_freq == 0:
                visuals = env.get_current_visuals()
                grid = visual_grid(visuals, opt.K, opt.T)
                writer.add_image('current_batch', grid, total_updates)

            if total_updates % opt.save_latest_freq == 0:
                print('saving the latest model (update %d)' % total_updates)
                env.save('latest', total_updates)
                env.save(total_updates, total_updates)

            if total_updates % opt.validate_freq == 0:
                psnr_plot, ssim_plot, grid = val(opt.c_dim, opt.data, opt.T * 2, opt.dataroot, opt.textroot,
                                                 'val_data_list.txt', opt.K, opt.backwards, opt.flip, val_pick_mode,
                                                 opt.image_size, val_gpu_ids, opt.model_type, opt.skip, opt.F,
                                                 val_batch_size, True, opt.nThreads, opt.gf_dim, False,
                                                 opt.checkpoints_dir, opt.name, opt.no_adversarial, opt.alpha, opt.beta,
                                                 opt.D_G_switch, opt.margin, opt.lr, opt.beta1, opt.sn, opt.df_dim,
                                                 opt.Ip, opt.comb_type, opt.comb_loss, opt.shallow, opt.ks,
                                                 opt.num_block, opt.layers, opt.kf_dim, opt.enable_res, opt.rc_loc,
                                                 opt.continue_train, 'latest')
                writer.add_image('psnr', psnr_plot, total_updates)
                writer.add_image('ssim', ssim_plot, total_updates)
                writer.add_image('samples', grid, total_updates)

            if total_updates >= opt.max_iter:
                env.save('latest', total_updates)
                break

        if total_updates >= opt.max_iter:
            break


if __name__ == '__main__':
    main()

