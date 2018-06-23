import io
import os

import matplotlib
from torch.nn import init

matplotlib.use('Agg')
import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt
import torchvision.utils as vutils


def print_current_errors(log_name, update, errors, t):
    message = 'update: %d, time: %.3f ' % (update, t)
    for k, v in errors.items():
        if k.startswith('Update'):
            message += '%s: %s ' % (k, str(v))
        else:
            message += '%s: %.3f ' % (k, v)

    print(message)
    with open(log_name, 'a') as log_file:
        log_file.write('%s \n' % message)


def inverse_transform(images):
    return (images+1.)/2


def fore_transform(images):
    return images * 2 - 1


def bgr2gray(image):
    # rgb -> grayscale 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray_ = 0.1140 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.2989 * image[:, 2, :, :]
    gray = torch.unsqueeze(gray_, 1)
    return gray


def makedir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def draw_frame(img, is_input):
    """Draws a red or green frame on the boundary of the given image.

    :param img: A np.ndarray image
    :param is_input: Whether the given image represents an input frame. If True, draws a green frame, otherwise draws a
                     red frame.
    """

    if img.shape[2] == 1:
        img = np.repeat(img, [3], axis=2)

    if is_input:
        # Draw a 2px green frame
        img[:2,:,0]  = img[:2,:,2] = 0
        img[:,:2,0]  = img[:,:2,2] = 0
        img[-2:,:,0] = img[-2:,:,2] = 0
        img[:,-2:,0] = img[:,-2:,2] = 0
        img[:2,:,1]  = 255
        img[:,:2,1]  = 255
        img[-2:,:,1] = 255
        img[:,-2:,1] = 255
    else:
        # Draw a 2px red frame
        img[:2,:,0]  = img[:2,:,1] = 0
        img[:,:2,0]  = img[:,:2,1] = 0
        img[-2:,:,0] = img[-2:,:,1] = 0
        img[:,-2:,0] = img[:,-2:,1] = 0
        img[:2,:,2]  = 255
        img[:,:2,2]  = 255
        img[-2:,:,2] = 255
        img[:,-2:,2] = 255
    return img


def draw_frame_tensor(img, K, T):
    """ Draws a red or green frame on the boundary of the given image tensor
    :param img: a torch.tensor image with all frames of a test case
    :param K: the number of input frames
    :param T: the number of output frames
    :return: a torch,tensor image
    """
    img[:, 0, :2, :] = img[:, 2, :2, :] = 0
    img[:, 0, :, :2] = img[:, 2, :, :2] = 0
    img[:, 0, -2:, :] = img[:, 2, -2:, :] = 0
    img[:, 0, :, -2:] = img[:, 2, :, -2:] = 0
    img[:, 1, :2, :] = 1
    img[:, 1, :, :2] = 1
    img[:, 1, -2:, :] = 1
    img[:, 1, :, -2:] = 1
    img[K:K+T, 0, :2, :] = img[K:K+T, 1, :2, :] = 0
    img[K:K+T, 0, :, :2] = img[K:K+T, 1, :, :2] = 0
    img[K:K+T, 0, -2:, :] = img[K:K+T, 1, -2:, :] = 0
    img[K:K+T, 0, :, -2:] = img[K:K+T, 1, :, -2:] = 0
    img[K:K+T, 2, :2, :] = 1
    img[K:K+T, 2, :, :2] = 1
    img[K:K+T, 2, -2:, :] = 1
    img[K:K+T, 2, :, -2:] = 1
    return img


def draw_err_plot(err,  err_name, lims, path=None):
    """Draws an average PSNR or SSIM error plot and either saves it to disk or returns the image as a np.ndarray.

    :param err: The error values to plot as a N x T np.ndarray
    :param err_name: The title of the plot
    :param lims: The axis limits of the plot
    :param path: The path to write the plot image to. If None, return the plot image as an np.ndarray
    """
    avg_err = np.mean(err, axis=0)
    T = err.shape[1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(1, T+1)
    ax.plot(x, avg_err, marker='d')
    ax.set_xlabel('time steps')
    ax.set_ylabel(err_name)
    ax.grid()
    ax.set_xticks(x)
    ax.axis(lims)
    if path is None:
        plot_buf = gen_plot(fig)
        im = np.array(Image.open(plot_buf), dtype=np.uint8)
        plt.close(fig)
        return im
    else:
        plt.savefig(path)


def gen_plot(fig):
    """
    Create a pyplot plot and save to buffer.
    https://stackoverflow.com/a/38676842
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf


def visual_grid(visuals, K, T):
    """Generate the qualitative results for validation

    :param visuals: A dictionary containing the data to visualize. It should contain the following key-value pairs:
                    - seq_batch: a np.ndarray containing all ground truth frames
                    - pred: a list of np.ndarrays containing the final prediction of frame inpainting
                    - pred_forward (optional): a list of np.ndarrays containing the forward prediction of frame
                      inpainting
                    - pred_backward (optional): a list of np.ndarrays containing the backward prediction of frame
                      inpainting
    :param K: int, the number of input frames
    :param T: int, the number of output frames
    :return: a torch.tensor with the visualization of qualitative results for validation
    """

    seq_batch = visuals['seq_batch']
    pred = visuals['pred']
    pred_forward = visuals.get('pred_forward', None)
    pred_backward = visuals.get('pred_backward', None)

    # Make sure pred_forward and pred_backward are both None or both not None
    assert((pred_forward is None and pred_backward is None) or (pred_forward is not None and pred_backward is not None))
    # Determine whether the given data is for inpainting or prediction results
    infill = (pred_forward is not None and pred_backward is not None)

    # stack predicted frames
    pred_data = torch.stack(pred, dim=-1)
    # according to pred_forward and pred_backward, set vis_f and vis_b and stack intermediate results
    if infill:
        pred_forward_data = torch.stack(pred_forward, dim=-1)
        pred_backward_data = torch.stack(pred_backward, dim=-1)
    # get the ground truth for the predicted part
    true_data = seq_batch[:, :, :, :, K:K + T].clone()
    # get the ground truth, final prediction, intermediate predictions of the whole sequence,
    # i.e. preceding sequence + middle sequence + the following sequence
    if infill:
        pred_forward_data = torch.cat([seq_batch[:, :, :, :, :K], pred_forward_data, seq_batch[:, :, :, :, K + T:]],
                                      dim=-1)
        pred_backward_data = torch.cat([seq_batch[:, :, :, :, :K], pred_backward_data, seq_batch[:, :, :, :, K + T:]],
                                       dim=-1)
        pred_data = torch.cat([seq_batch[:, :, :, :, :K], pred_data, seq_batch[:, :, :, :, K+T:]], dim=-1)
        true_data = torch.cat([seq_batch[:, :, :, :, :K], true_data, seq_batch[:, :, :, :, K+T:]], dim=-1)
    else:
        pred_data = torch.cat([seq_batch[:, :, :, :, :K], pred_data], dim=-1)
        true_data = torch.cat([seq_batch[:, :, :, :, :K], true_data], dim=-1)
    # get the batch size, channel dimension of data and the length of the whole sequence
    batch_size = int(pred_data.size()[0])
    c_dim = int(pred_data.size()[1])
    seq_len = int(pred_data.size()[-1])

    vis = []
    for i in range(batch_size):
        # transpose the tensor and map the range from [-1, 1] to [0, 1]
        if infill:
            pred_forward_data_sample = inverse_transform(pred_forward_data[i, :, :, :, :]).permute(3, 0, 1, 2)
            pred_backward_data_sample = inverse_transform(pred_backward_data[i, :, :, :, :]).permute(3, 0, 1, 2)
        pred_data_sample = inverse_transform(pred_data[i, :, :, :, :]).permute(3, 0, 1, 2)
        target_sample = inverse_transform(true_data[i, :, :, :, :]).permute(3, 0, 1, 2)
        # if the original data is gray-scale, expand its color dimension to 3 for adding on colorful box
        if c_dim == 1:
            if infill:
                pred_forward_data_sample = torch.cat([pred_forward_data_sample] * 3, dim=1)
                pred_backward_data_sample = torch.cat([pred_backward_data_sample] * 3, dim=1)
            pred_data_sample = torch.cat([pred_data_sample] * 3, dim=1)
            target_sample = torch.cat([target_sample] * 3, dim=1)
        # draw boxes with different color according to the frame index
        if infill:
            pred_forward_data_sample = draw_frame_tensor(pred_forward_data_sample, K, T)
            pred_backward_data_sample = draw_frame_tensor(pred_backward_data_sample, K, T)
        pred_data_sample = draw_frame_tensor(pred_data_sample, K, T)
        target_sample = draw_frame_tensor(target_sample, K, T)

        # align the output with the order of [forward prediction, backward prediction, final prediction and gt]
        # for each test case
        draw_sample = []
        if infill:
            draw_sample.append(pred_forward_data_sample)
            draw_sample.append(pred_backward_data_sample)
        draw_sample += [pred_data_sample, target_sample]
        output = torch.cat(draw_sample, dim=0)
        # generate image grid using function in torchvision, each result for a whole sequence is visualized in one line
        vis.append(vutils.make_grid(output, nrow=seq_len))
    # combine all cases in the same mini-batch
    grid = torch.cat(vis, dim=1)
    # clip value to (0, 1)
    grid = torch.from_numpy(np.clip(np.flip(grid.numpy(), 0).copy(), 0, 1))
    return grid


def get_semantic_name(model, rc_loc, comb_type):
    """Generate the semantic name of the architecture determined by the given arguments."""
    if model == 'kernelcomb' and rc_loc != -1:
        semantic_model_name = 'TAI'
    elif model == 'kernelcomb' and rc_loc == -1 and comb_type == 'w_avg':
        semantic_model_name = 'TWI'
    elif model == 'simplecomb' and comb_type == 'w_avg':
        semantic_model_name = 'bi-TW'
    elif model == 'simplecomb' and comb_type == 'avg':
        semantic_model_name = 'bi-SA'
    elif model == 'mcnet':
        semantic_model_name = 'MC-Net'
    else:
        print('{model, rc_loc, comb_type} = {%s, %d, %s} does not have a name' % (model, rc_loc, comb_type))
        semantic_model_name = 'Unknown'

    return semantic_model_name


def get_semantic_name_trivial(comb_type):
    """Generate the semantic name of the trivial baseline determined by the given arguments."""
    if comb_type == 'repeat_P' or comb_type == 'repeat_F':
        semantic_model_name = comb_type
    elif comb_type == 'w_avg':
        semantic_model_name = 'TW_P_F'
    elif comb_type == 'avg':
        semantic_model_name = 'SA_P_F'
    else:
        print('comb_type = {%s} does not have a name for trivial baseline' % (comb_type))
        semantic_model_name = 'Unknown'

    return semantic_model_name


def refresh_donelist(data, comb_type, model, rc_loc, K, T, test_name, is_trivial=False):
    """ Record finished experiments in a list

    :param data: str, the dataset to train
    :param comb_type: str, the comb type of the final prediction
    :param model: str, determine whether to blend via a interpolation network
    :param rc_loc: int, where to insert the scaled time step
    :param K: int, the number of input frames
    :param T: int, the number of output frames
    :param test_name: str, combined with exp_name, K and T
    :param is_trivial: bool, a flag to determine whether it is a trivial exp,
    """
    makedir('records/')
    exps_path = 'records/finished_exp.npy'
    data_name = {'KTH': 'KTH Actions', 'UCF': 'UCF-101', 'HMDB51': 'HMDB-51'}
    data = data_name[data]
    if is_trivial:
        semantic_model_name = get_semantic_name_trivial(comb_type)
    else:
        semantic_model_name = get_semantic_name(model, rc_loc, comb_type)
    i_o_num = '%d_%d'%(K, T)
    if os.path.exists(exps_path):
        exps_dict = np.load(exps_path).item()
    else:
        exps_dict={}

    if not data in exps_dict.keys():
        exps_dict[data] = {}

    if not semantic_model_name in exps_dict[data].keys():
        exps_dict[data][semantic_model_name] = {}

    if not i_o_num in exps_dict[data][semantic_model_name].keys():
        exps_dict[data][semantic_model_name][i_o_num] = []

    exps_dict[data][semantic_model_name][i_o_num].append(test_name)

    np.save(exps_path, exps_dict)


def listopt(opt, f=None):
    """Pretty-print a given namespace either to console or to a file.

    :param opt: A namespace
    :param f: The file descriptor to write to. If None, write to console
    """
    args = vars(opt)

    if f is not None:
        f.write('------------ Options -------------\n')
    else:
        print('------------ Options -------------')

    for k, v in sorted(args.items()):
        if f is not None:
            f.write('%s: %s\n' % (str(k), str(v)))
        else:
            print('%s: %s' % (str(k), str(v)))

    if f is not None:
        f.write('-------------- End ----------------\n')
    else:
        print('-------------- End ----------------')


def to_numpy(tensor, transpose=None):
    """Converts the given Tensor to a np.ndarray.

    :param tensor: The Tensor to convert
    :param transpose: An iterable specifying the new dimensional ordering
    """

    arr = tensor.cpu().numpy()
    if transpose is not None:
        arr = np.transpose(arr, transpose)

    return arr


def move_to_devices(model, gpu_ids):
    """Moves the model to the specified GPUs. If gpu_ids has one element, this simply moves the model to the GPU."""
    assert(len(gpu_ids) > 0)
    model = torch.nn.DataParallel(model.cuda()) if len(gpu_ids) >= 2 else model.cuda()
    return model


def weights_init_std_discriminator(m):
    """Weight initialization used specifically for the normal discriminator. The ranges used here are smaller than those
    used in the normal initialization scheme because large values caused the discriminator to have high cross-entropy
    error but small gradients.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname != 'ConvLstmCell':
        init.uniform(m.weight.data, 0.0, 0.0001)
        init.constant(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.0001)
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname != 'ConvLstmCell':
        init.xavier_normal(m.weight.data, gain=1)
        init.constant(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)