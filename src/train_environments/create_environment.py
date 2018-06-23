from .mcnet_infill_environment import McnetInfillEnvironment
from .mcnet_kernel_comb_environment import McnetKernelCombEnvironment
from .mcnet_environment import McnetEnvironment


def create_environment(model_type, gf_dim, c_dim, gpu_ids, is_train, checkpoints_dir, name, K, T, F, image_size,
                       batch_size, which_update, comb_type, shallow, ks, num_block, layers, kf_dim, enable_res, rc_loc,
                       no_adversarial=None, alpha=None, beta=None, D_G_switch=None, margin=None, lr=None, beta1=None,
                       sn=None, df_dim=None, Ip=None, continue_train=None, comb_loss=None):
    print(model_type)
    if model_type == 'mcnet':
        generator_args = gf_dim, c_dim
        model = McnetEnvironment(generator_args, gpu_ids, is_train, checkpoints_dir, name, K, T, image_size, batch_size,
                                 c_dim, no_adversarial, alpha, beta, D_G_switch, margin, lr, beta1, sn, df_dim, Ip,
                                 K + T)
    elif model_type == 'simplecomb':
        generator_args = gf_dim, c_dim
        model = McnetInfillEnvironment(generator_args, gpu_ids, is_train, checkpoints_dir, name, K, T, image_size,
                                       batch_size, c_dim, no_adversarial, alpha, beta, D_G_switch, margin, lr, beta1,
                                       sn, df_dim, Ip, K + F + T, F, comb_type, comb_loss)
    elif model_type == 'kernelcomb':
        generator_args = shallow, gf_dim, c_dim, ks, num_block, layers, kf_dim, enable_res, rc_loc
        model = McnetKernelCombEnvironment(generator_args, gpu_ids, is_train, checkpoints_dir, name, K, T, image_size,
                                           batch_size, c_dim, no_adversarial, alpha, beta, D_G_switch, margin, lr,
                                           beta1, sn, df_dim, Ip, K + F + T, F, comb_type, comb_loss)
    else:
        raise ValueError('Model type [%s] not recognized.' % model_type)

    if not is_train or continue_train:
        model.load(which_update)
    print('model [%s] is created' % (model.name()))

    return model