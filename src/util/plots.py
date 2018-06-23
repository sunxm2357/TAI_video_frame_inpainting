"""
Script to generate the plots in our video frame inpainting paper (https://arxiv.org/abs/1803.07218).
"""

import os

import matplotlib
matplotlib.use('Agg')
import numpy as np

import matplotlib.pyplot as plt
import argparse
from util import makedir

color_dict = {
    'MC-Net': 'k',
    'bi-SA': 'r',
    'bi-TW': 'g',
    'TAI': 'b',
    'TWI': 'C5',
    'repeat_P': 'k',
    'repeat_F': 'C7',
    'SA_P_F': 'r',
    'TW_P_F': 'g'
}

ls_dict = {
    'MC-Net': '-',
    'bi-SA': '-',
    'bi-TW': '-',
    'TAI': '-',
    'TWI': '-',
    'repeat_P': '--',
    'repeat_F': '--',
    'SA_P_F': '--',
    'TW_P_F': '--'
}

nick_name = {
    'KTH Actions': 'KTH',
    'UCF-101': 'UCF',
    'HMDB-51': 'HMDB51'
}

ylims = {
    'psnr': {
        'KTH Actions': {
            'fig4': [25, 38],
            'fig8': [28, 38],
            'fig24': [27, 38]
        },
        'UCF-101': {
            'fig9': [24, 32]
        },
        'HMDB-51': {
            'fig9': [24, 32]
        }
    },
    'ssim': {
        'KTH Actions': {
            'fig4': [ 0.8, 0.98],
            'fig7(a)': [0.88, 0.98],
            'fig8':[0.88, 1],
            'fig24': [0.83, 0.98]
        },
        'UCF-101': {
            'fig9': [0.8, 0.92]
        },
        'HMDB-51': {
            'fig9': [0.75, 0.9]
        }
    }
}


def name2keys(which_plot):
    if which_plot == 'fig4':
        datasets = ['KTH Actions']
        metrics = ['psnr', 'ssim']
        models = ['repeat_P', 'repeat_F', 'SA_P_F',  'TW_P_F', 'MC-Net', 'bi-SA', 'bi-TW', 'TAI']
        i_os = ['5_10']

    elif which_plot == 'fig7(a)':
        datasets = ['KTH Actions']
        metrics = ['ssim']
        models = ['bi-TW', 'TWI', 'TAI']
        i_os = ['5_10']

    elif which_plot == 'fig8':
        datasets = ['KTH Actions']
        metrics = ['psnr', 'ssim']
        models = ['TAI']
        i_os = ['2_10', '3_10', '4_10', '5_10']

    elif which_plot == 'fig9':
        datasets = ['UCF-101', 'HMDB-51']
        metrics = ['psnr', 'ssim']
        models = ['repeat_P', 'repeat_F', 'SA_P_F',  'TW_P_F', 'MC-Net', 'bi-SA', 'bi-TW', 'TAI']
        i_os = ['4_5']

    elif which_plot == 'fig24':
        datasets = ['KTH Actions']
        metrics = ['psnr', 'ssim']
        models = ['MC-Net', 'bi-SA', 'bi-TW', 'TAI']
        i_os = ['5_6', '5_7', '5_8', '5_9']

    else:
        raise ValueError('%s is not a plot in the paper'%(which_plot))

    return datasets, metrics, models, i_os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--which_plot', type=str, required=True, choices=['fig4', 'fig7(a)', 'fig8', 'fig9', 'fig24'])
    parser.add_argument('--results_dir', type=str, default='results/')
    parser.add_argument('--output_dir', type=str, default='paper_plots')
    opt = parser.parse_args()
    datasets, metrics, models, i_os = name2keys(opt.which_plot)
    exp_dict = dict(np.load('records/finished_exp.npy').item())

    makedir(opt.output_dir)
    if opt.which_plot != 'fig8':
        for metric in metrics:
            if metric == 'ssim':
                ylabel = 'SSIM'
            else:
                ylabel = 'PSNR'

            for dataset in datasets:
                for i_o in i_os:
                    fig, ax = plt.subplots(1, 1)
                    for model in models:
                        try:
                            test_name = exp_dict[dataset][model][i_o][-1]
                        except:
                            print('exp with {%s, %s, %s} does not exist'%(dataset, model, i_o))
                            continue
                        quant_exp_dir = os.path.join(opt.results_dir,'quantitative', nick_name[dataset], test_name)
                        metrics_dict = dict(np.load(os.path.join(quant_exp_dir, 'results.npz')))
                        quant_result = metrics_dict[metric]
                        mask = np.isinf(quant_result)
                        quant_result = np.ma.array(quant_result, mask=mask)
                        avg_err = quant_result.mean(axis=0)
                        T = quant_result.shape[1]
                        x = np.arange(1, T + 1)
                        ax.plot(x, avg_err.data, linestyle=ls_dict[model], color=color_dict[model], linewidth=2)

                    axes = plt.gca()
                    axes.set_ylim(ylims[metric][dataset][opt.which_plot])
                    x_lim = [1, T]
                    axes.set_xlim(x_lim)

                    ax.set_xlabel('time steps')
                    ax.set_ylabel(ylabel)
                    ax.set_xticks(x)
                    ax.set_title(dataset)
                    plt.legend(models, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, ncol=min(4, len(models)))

                    plt.grid()

                    plot_name = os.path.join(opt.output_dir, '%s_%s_%s_%s.eps'%(opt.which_plot, nick_name[dataset], metric, i_o))
                    fig.set_size_inches(7, 5*0.75)
                    plt.savefig(plot_name, format='eps', bbox_inches='tight')
    else:
        for metric in metrics:
            if metric == 'ssim':
                ylabel = 'SSIM'
            else:
                ylabel = 'PSNR'

            for dataset in datasets:
                for model in models:
                    fig, ax = plt.subplots(1, 1)
                    legends = []
                    for i_o in i_os:
                        try:
                            test_name = exp_dict[dataset][model][i_o][-1]
                        except:
                            print('exp with {%s, %s, %s} does not exist'%(dataset, model, i_o))
                            continue
                        in_num = int(i_o.split('_')[0])
                        legends.append('%s(input %d frames)'%(model, in_num))
                        quant_exp_dir = os.path.join(opt.results_dir, 'quantitative', nick_name[dataset], test_name)
                        metrics_dict = dict(np.load(os.path.join(quant_exp_dir, 'results.npz')))
                        quant_result = metrics_dict[metric]
                        mask = np.isinf(quant_result)
                        quant_result = np.ma.array(quant_result, mask=mask)
                        avg_err = quant_result.mean(axis=0)
                        T = quant_result.shape[1]
                        x = np.arange(1, T + 1)
                        ax.plot(x, avg_err.data, linewidth=2)

                    axes = plt.gca()
                    axes.set_ylim(ylims[metric][dataset][opt.which_plot])
                    x_lim = [1, T]
                    axes.set_xlim(x_lim)

                    ax.set_xlabel('time steps')
                    ax.set_ylabel(ylabel)
                    ax.set_xticks(x)
                    ax.set_title(dataset)
                    plt.legend(models, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, ncol=min(4, len(models)))
                    plt.grid()

                    plot_name = os.path.join(opt.output_dir, '%s_%s_%s.eps'%(opt.which_plot, nick_name[dataset], metric))
                    fig.set_size_inches(7, 5*0.75)
                    plt.savefig(plot_name, format='eps', bbox_inches='tight')


if __name__ == '__main__':
    main()
