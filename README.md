# A Temporally-Aware Interpolation Network for Video Frame Inpainting
Ximeng Sun, Ryan Szeto and Jason J. Corso, University of Michigan



## Using this code
If you use this work in your research, please cite the following paper:

```
@article{sun2018temporally,
  title={A Temporally-Aware Interpolation Network for Video Frame Inpainting},
  author={Sun, Ximeng and Szeto, Ryan and Corso, Jason J},
  journal={arXiv preprint arXiv:1803.07218},
  year={2018}
} 
```

## Introduction

## Setup the environment
In this section, we provide an instruction to help you set up the experiment environment. This code was tested for Python 2.7 on Ubuntu 16.04, and is based on PyTorch 0.3.1.

First, please make sure you have installed  `cuda(8.0.61)`, `cudnn(8.0-v7.0.5)` `opencv(2.4.13)`, and `ffmpeg(2.2.3)`.  Note that different versions might not work.

Then, to avoid conflicting with your current environment, we suggest to use `virtualenv`. For using `virtualenv`, please refer https://virtualenv.pypa.io/en/stable/. Below is an example of initializing and activating a virtual environment:

```bash
virualenv .env
source .env/bin/activate
```

After activating a virtual environment, use the following command to install all dependencies you need to run this model.

```bash
pip install -r requirements.txt
```

### Compiling the separable convolution module

The separable convolution module uses CUDA code that must be compiled before it can be used. To compile, start by activating the virtual environment described above. Then, set the `CUDA_HOME` environment variable as the path to your CUDA installation. For example, if your CUDA installation is at `/usr/local/cuda-8.0`, run this command:

```bash
export CUDA_HOME=/usr/local/cuda-8.0
```

Then, identify a virtual architecture that corresponds to your GPU from [this site](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list) (e.g. `compute_52` for Maxwell GPUs). Finally, to compile the CUDA code, run the install script with the virtual architecture as an argument, e.g.:

```bash
bash bashes/misc/install.bash compute_52
```


## Download checkpoints, datasets and sample results

### Download checkpoints

The checkpoints for the models used in our paper are available online. To set them up, download and extract them to the project root directory:

```bash
wget http://web.eecs.umich.edu/~szetor/media/tai-checkpoints.zip
unzip tai-checkpoints.zip
rm tai-checkpoints.zip
```

### Download datasets 

In our paper, we use the KTH Actions, UCF-101, and HMDB-51 datasets to evaluate our method. Use the commands below to download these datasets. The sample commands below download the datasets into a new `datasets` directory, but you can replace this argument with another location if needed. **Note: Each command may take about 1.5 hours on a 1 Gbit/s connection, but they can be run in parallel.**

```bash
bash bashes/download/download_KTH.bash datasets
bash bashes/download/download_UCF.bash datasets
bash bashes/download/download_HMDB.bash datasets
```

## Evaluate models and reproduce results in paper

### Evaluate single model

In this section, we describe how to generate inpainted frames and quantitative comparisons to the ground-truth frames. Before you start, make sure you have extracted the checkpoints as described in the "Download checkpoints" section.

The scripts to run each model described in our paper are spread throughout the `bashes` folder, but generally are organized as `bashes/evaluation/<train_dataset>_<model_name>/test_<test_dataset>.bash`. Each script takes the following arguments:

1. The number of preceding/following frames
2. The number of middle frames to generate
3. The path to the test dataset

For example, the command below uses our KTH-trained TAI model to generate predictions on the KTH test set. It takes in five preceding frames and five following frames to generate ten middle frames:

```bash
bash bashes/evaluation/kth_TAI/test_KTH.bash 5 10 datasets/KTH
```

Each script takes about four hours to finish. When the evaluation is finished, the quantitative results will be under `results/quantitative/<test_dataset>/<exp_test_name>` and the qualitative results will be under `results/images/<test_dataset>/<exp_test_name>`.

### Common arguments for test
In each test bash file, we define a set of arguments which are passed to `test.py`. To help understand what common arguments are, we take `bashes/evaluation/kth_TAI/test_KTH.bash` for example.

```bashes
python src/test.py \
    --name=kth_TAI_exp1 \
    --K=${1} \
    --F=${1} \
    --T=${2} \
    --data=KTH --dataroot=${3} --textroot=videolist/KTH/ --c_dim=1 --pick_mode=Slide \
    --model=kernelcomb --comb_type=avg \
    --image_size 128 \
    --batch_size=1 \
    --num_block=5 \
    --layers=3 \
    --kf_dim=32 \
    --rc_loc=4 \
    --enable_res \
    --output_both_directions \
    --shallow
```

This is what each argument in the above example specifies:

* name: the name of experiments
* K: the number of preceding frames
* F: the number of following frames
* T: the number of middle frames
* data: the name of dataset
* dataroot: the path of the dataset
* textroot: the path of files defining train/test/val split
* c_dim: the number of color channels (1 for grayscale)
* pick_mode: how to extract test cases from the specified dataset. Use "Slide" to generate test clips via a sliding window on every test video, or "First" to evaluate on the first K + F + T frames of each test video.
* model/comb_type/rc_loc/shallow: Together, these specify how to make intermediate predictions and stitch them together. In this case, kernelcomb/avg/4/True yields our TAI network.
* image_size: the height and width to scale all frames to
* batch_size: number of samples in a mini-batch
* num_block: the number of VGG blocks in the encoder/decoder of the blend network
* layers: the number of convolution layers in a VGG block
* kf_dim: the number of kernels in the first VGG block of the blend network
* enable_res: enable the residual network in the video prediction part
* output_both_directions: output the intermediate predictions

For all arguments used in test, you can refer `src/options/base_options.py` and `src/options/test_options.py`.

### Reproduce all experiments in paper

We provide a bash file to automatically evaluate all the models in our paper. Its arguments are the paths to the KTH, UCF-101, and HMDB-51 datasets. **WARNING: THIS SCRIPT TAKES OVER 80 HOURS TO COMPLETE. WE RECOMMEND SCHEDULING THE JOBS CONTAINED IN `all_evals.bash` ON A COMPUTE CLUSTER.**

```bash
bash bashes/evaluation/all_evals.bash datasets/KTH datasets/UCF-101 datasets/HMDB-51
```

### Reproduce quantitative and qualitative results in paper

After running all evaluations, use `bashes/evaluation/results_reproduction.bash` to generate the figures shown in our paper.

```bash
bash bashes/evaluation/results_reproduction.bash
```

After running this script, the quantitative plots will be located under `paper_plots` and the qualitative plots will be located under `paper_imgs`. 

Note that if you run this script multiple times for the same model on the same dataset with the same numbers of preceding/following and middle frames, it will overwrite any existing figures under `paper_plots` and `paper_imgs`. Alternatively, you can run individual lines in `bashes/results_reproduction.bash` to generate any specific plot you want.

## Train our model and baselines

### Train single model

To train each model, you can use the training file under `bashes/evaluation/<exp name>/train.bash` by providing the location of the dataset. For example, if we want to train the TAI network on KTH Actions, run

```bash
bash bashes/evaluation/kth_TAI/train.bash datasets/KTH
```
If the training needs to be resumed, you can use the training file under `bashes/evaluation/<exp name>/train.bash` by passing an extra argument. For example, if we want to resume the training of TAI network on KTH Actions, run

```bash
bash bashes/evaluation/kth_TAI/train.bash datasets/KTH resume
```

### Common arguments for training
In each training bash file, we define a set of arguments which are passed to `train.py`. To help understand what common arguments are, we take `bashes/evaluation/kth_TAI/train.bash` for example.

```bashes
python src/train.py  \
    --name=kth_TAI_exp1 \
    --K=5 \
    --F=5 \
    --T=5 \
    --max_iter=200000 \
    --data=KTH --dataroot=${1} --textroot=videolist/KTH/ --c_dim=1 \
    --model=kernelcomb --comb_loss=ToTarget --comb_type=avg \
    --image_size 128 \
    --final_sup_update=1 \
    --batch_size=4 \
    --num_block=5 \
    --layers=3 \
    --kf_dim=32 \
    --D_G_switch=alternative \
    --beta=0.002 \
    --rc_loc=4 \
    --Ip=3 \
    --sn \
    --enable_res \
    --shallow \
```

This is what each argument in the above example specifies:

* name: the name of experiments
* K: the number of preceding frames
* F: the number of following frames
* T: the number of middle frames
* max_iter: the maximum number of iterations to train for
* data: the name of dataset
* dataroot: the path of the dataset
* textroot: the path of files defining train/test/val split
* c_dim: the number of color channels (1 for grayscale)
* model/comb_type/rc_loc/shallow: Together, these specify how to make intermediate predictions and stitch them together. In this case, kernelcomb/avg/4/True yields our TAI network.
* comb_loss: how to penalize the intermediate predictions. "ToTarget" specifies that they should be penalized based on their similarity to the ground truth target, and "Closer" specifies that they should be penalized based on their similarity to each other.
* image_size: the height and width to scale all frames to
* final_sup_update: when to introduce supervision on the final prediction
* batch_size: number of samples in a mini-batch
* num_block: the number of VGG blocks in the encoder/decoder of the blend network
* layers: the number of convolution layers in a VGG block
* kf_dim: the number of kernels in the first VGG block of the blend network
* D_G_switch: whether to update the generator and discriminator in an alternating fashion
* beta: the weight of the GAN loss when training the generator
* Ip: the number of power iterations in every update
* sn: whether to use spectral normalization in the discriminator
* enable_res: enable the residual network in the video prediction part

For all arguments used in training, you can refer `src/options/base_options.py` and `src/options/train_options.py`.

### Visualize the training process
Losses, visualizations for the current batch and the validation are saved in the tensorboard file under `tb/` for each experiment. In order to access these intermediate results, you can activate a tensorboard instance. Take `kth_TAI_exp1` experiment for example, run

```bashes
tensorboard --logdir kth_TAI:./tb/kth_TAI_exp1/
``` 


