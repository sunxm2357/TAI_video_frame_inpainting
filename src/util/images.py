"""
Script to generate the qualitative results in our video frame inpainting paper (https://arxiv.org/abs/1803.07218).
"""

import argparse
import cv2
import os

import numpy as np
from PIL import Image
from skimage.transform import resize

from util import makedir

nick_name = {
    'KTH Actions': 'KTH',
    'UCF-101': 'UCF',
    'HMDB-51': 'HMDB51'
}
videos = {
    'KTH Actions': 'videolist/KTH/paper_visual.txt',
    'UCF-101': 'videolist/UCF/paper_visual.txt',
    'HMDB-51': 'videolist/HMDB/paper_visual.txt'
}
input_output = {
    'KTH Actions': '5_10',
    'UCF-101': '4_5',
    'HMDB-51': '4_5'
}
models = ['MC-Net', 'bi-SA', 'bi-TW', 'TAI']
start = {
    'KTH Actions': 1,
    'HMDB-51': 0,
    'UCF-101': 0
}
skip = {
    'KTH Actions': 2,
    'HMDB-51': 2,
    'UCF-101': 2
}


def draw_boxing(frame, h1, w1, h2, w2):
    h = frame.shape[0]
    w = frame.shape[1]
    linewidth = 2
    h1 = max([0, h1-linewidth]) + linewidth
    w1 = max([0, w1-linewidth]) + linewidth
    h2 = min([h-1, h2+linewidth]) - linewidth
    w2 = min([w-1, w2+linewidth]) - linewidth
    frame[h1-linewidth:h1, w1-linewidth: w2 + linewidth, 0] = 255
    frame[h1 - linewidth:h1, w1 - linewidth: w2 + linewidth, 1] = 0
    frame[h1 - linewidth:h1, w1 - linewidth: w2 + linewidth, 2] = 0
    frame[h2: h2 + linewidth, w1 - linewidth: w2 + linewidth, 0] = 255
    frame[h2: h2 + linewidth, w1 - linewidth: w2 + linewidth, 1] = 0
    frame[h2: h2 + linewidth, w1 - linewidth: w2 + linewidth, 2] = 0
    frame[h1 - linewidth: h2 + linewidth, w1 - linewidth: w1, 0] = 255
    frame[h1 - linewidth: h2 + linewidth, w1 - linewidth: w1, 1] = 0
    frame[h1 - linewidth: h2 + linewidth, w1 - linewidth: w1, 2] = 0
    frame[h1 - linewidth: h2 + linewidth, w2: w2 + linewidth, 0] = 255
    frame[h1 - linewidth: h2 + linewidth, w2: w2 + linewidth, 1] = 0
    frame[h1 - linewidth: h2 + linewidth, w2: w2 + linewidth, 2] = 0
    return frame


def slide_selected_videos(input_output, lines, models, image_dir, name, skip=1, start=0, baseline=None):
    maxw = 0
    F = K = int(input_output.split('_')[0])
    T = int(input_output.split('_')[1])
    display_idx = range(start, K+F+T, skip)
    for c, line in enumerate(lines):
        if line.endswith('\n'):
            line = line[:-1]
        clip = line.split()[0]

        if len(line.split()) > 1:
            enable_zoom = True
            frame_idx = int(line.split()[1])
            frame_idx = (frame_idx-start)/skip
            h1, w1 = int(line.split()[2]), int(line.split()[3])
            h2, w2 = int(line.split()[4]), int(line.split()[5])
        else:
            enable_zoom = False
        print('dealing with clip %s' % clip)

        gt_flag = True
        for j, model in enumerate(models):
            count = 0
            frames = []
            print('dealing with model %s' % model)
            model_dir = os.path.join(image_dir, model)
            if not os.path.isdir(model_dir):
                raise ValueError('model %s is not found' % model)

            if model == baseline:
                prefix = clip.split('-')[0]
                suffix = clip.split('-')[1]
                clip_true = prefix + '-' + str(int(suffix)-F)
            else:
                clip_true = clip

            gif_dir = os.path.join(model_dir, clip_true)
            gt = Image.open(os.path.join(gif_dir, 'gt.gif'))
            img = np.array(gt.convert('RGB').getdata()).reshape(gt.size[1], gt.size[0], 3)[:,:,::-1]
            if count in display_idx:
                frames.append(img)
            count += 1
            try:
                while 1:
                    gt.seek(gt.tell() + 1)
                    img = np.array(gt.convert('RGB').getdata()).reshape(gt.size[1], gt.size[0], 3)[:, :, ::-1]
                    if count in display_idx:
                        frames.append(img)
                    count += 1
            except EOFError:
                pass  # end of sequence

            if gt_flag:
                if enable_zoom:
                    gt_clip = np.copy(frames[frame_idx][h1:h2, w1:w2])
                    h_ = gt_clip.shape[0]
                    w_ = gt_clip.shape[1]
                    height = img.shape[0]
                    w_prime = int(float(height) / h_ * w_)
                    gt_clip = resize(gt_clip/255., (height, w_prime)) * 255
                    frames[frame_idx] = draw_boxing(frames[frame_idx], h1, w1, h2, w2)
                    gt_img = np.concatenate(frames, axis=1)
                    width = len(display_idx) * img.shape[1]
                    if model == baseline:
                        current_width = gt_img.shape[1]
                        gt_img = np.concatenate([gt_img, np.zeros((img.shape[0], width-gt_img.shape[1], 3))], axis=1)
                    if j == 0:
                        seq_img = np.concatenate([gt_img, 255 * np.ones([gt_img.shape[0], 20, gt_img.shape[2]]), gt_clip], axis=1)
                    else:
                        seq_img[:, current_width:width, :] = gt_img[:, current_width:width, :]
                else:
                    gt_img = np.concatenate(frames, axis=1)
                    width = len(display_idx) * img.shape[1]
                    if model == baseline:
                        current_width = gt_img.shape[1]
                        gt_img = np.concatenate([gt_img, np.zeros((img.shape[0], width-gt_img.shape[1], 3))], axis=1)
                    if j == 0:
                        seq_img = gt_img
                    else:
                        seq_img[:, current_width:width, :] = gt_img[:, current_width:width, :]

                if model != baseline:
                    gt_flag = False

        for j, model in enumerate(models):
            print('dealing with model %s' % model)
            model_dir = os.path.join(image_dir, model)
            if not os.path.isdir(model_dir):
                raise ValueError('model %s is not found' % model)

            if model == baseline:
                prefix = clip.split('-')[0]
                suffix = clip.split('-')[1]
                clip_true = prefix + '-' + str(int(suffix) - F)
            else:
                clip_true = clip

            gif_dir = os.path.join(model_dir, clip_true)
            pred_gifs = ['pred_final.gif']
            preds = []
            for i, pred_gif in enumerate(pred_gifs):
                count = 0
                frames_preds = []
                preds.append(Image.open(os.path.join(gif_dir, pred_gif)))
                img = np.array(preds[i].convert('RGB').getdata()).reshape(preds[i].size[1], preds[i].size[0], 3)[:, :, ::-1]
                if count in display_idx:
                    frames_preds.append(img)
                count += 1
                try:
                    while 1:
                        idx = preds[i].tell() + 1
                        preds[i].seek(idx)
                        img = np.array(preds[i].convert('RGB').getdata()).reshape(preds[i].size[1], preds[i].size[0],
                                                                                  3)[:,:, ::-1]
                        if count in display_idx:
                            frames_preds.append(img)
                        count += 1
                except EOFError:
                    pass  # end of sequence
                if enable_zoom:
                    pred_clip = np.copy(frames_preds[frame_idx][h1:h2, w1:w2])
                    h_ = pred_clip.shape[0]
                    w_ = pred_clip.shape[1]
                    height = img.shape[0]
                    w_prime = int(float(height) / h_ * w_)
                    pred_clip = resize(pred_clip/255., (height, w_prime)) * 255
                    frames_preds[frame_idx] = draw_boxing(frames_preds[frame_idx], h1, w1, h2, w2)

                    pred_img = np.concatenate(frames_preds, axis=1)
                    width = len(display_idx) * img.shape[1]
                    current_width = pred_img.shape[1]
                    if model == baseline:
                        pred_img = np.concatenate([pred_img, np.zeros((img.shape[0], width - pred_img.shape[1], 3))], axis=1)
                    pred_img = np.concatenate([pred_img, 255 * np.ones([pred_img.shape[0], 20, pred_img.shape[2]]), pred_clip], axis=1)
                    pred_img[:, current_width:width, :] = gt_img[:, current_width:width, :]
                else:
                    width = len(display_idx) * img.shape[1]
                    pred_img = np.concatenate(frames_preds, axis=1)
                    current_width = pred_img.shape[1]
                    if model == baseline:
                        pred_img = np.concatenate([pred_img, np.zeros((img.shape[0], width-pred_img.shape[1], 3))], axis=1)
                    pred_img[:, current_width:width, :] = gt_img[:, current_width:width, :]

                seq_img = np.concatenate([seq_img, 255*np.ones((5, seq_img.shape[1], seq_img.shape[2])), pred_img], axis=0)

                if seq_img.shape[1] > maxw:
                    maxw = seq_img.shape[1]

        name_n = name + '_%d.png' % c
        cv2.imwrite(name_n, seq_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='[KTH Actions|UCF-101|HMDB-51]')
    parser.add_argument('--results_dir', type=str, default='results/')
    parser.add_argument('--output_dir', type=str, default='paper_imgs')
    opt = parser.parse_args()
    exp_dict = dict(np.load('records/finished_exp.npy').item())
    f = open(videos[opt.dataset], 'r')
    cases = f.readlines()
    test_names = []
    baseline = None
    makedir(opt.output_dir)
    for model in models:
        try:
            test_name = exp_dict[opt.dataset][model][input_output[opt.dataset]][-1]
            test_names.append(test_name)
            if model == 'MC-Net':
                baseline = test_name
        except:
            print('exp with {%s, %s, %s} does not exist' % (opt.dataset, model, input_output[opt.dataset]))
            continue

    img_dir = os.path.join(opt.results_dir, 'images', nick_name[opt.dataset])
    name_prefix = os.path.join(opt.output_dir, '%s_vis'%(opt.dataset))
    slide_selected_videos(input_output[opt.dataset], cases, test_names, img_dir, name_prefix, skip=skip[opt.dataset], start=start[opt.dataset], baseline=baseline)


if __name__ == '__main__':
    main()
