# GRASS in Pytorch
This is an Pytorch implementation of the paper "[GRASS: Generative Recursive Autoencoders for Shape Structures](http://kevinkaixu.net/projects/grass.html)". The paper is about learning a generative model for 3D shape structures by structural encoding and decoding with Recursive Neural Networks. This code was originally written by [Chenyang Zhu](http://www.sfu.ca/~cza68/) from Simon Fraser University and is being improved and maintained here in this repository.

Note that the current version implements only the Varational Auto-Encoder (VAE) part of the generative model. The implementation of the Generarive Adverserial Nets (GAN) part is still on-going. But this model can already generate novel 3D shape structures from sampled random noises. With GAN part, it is expected to generate more diverse structures.

## Usage
**Dependancy**
grass_pytorch should be run with Python 3.x. A porting to Python 2.x is provided in the folder of [python2](https://github.com/kevin-kaixu/grass_pytorch/tree/master/python2) (may not be up to date).
grass_pytorch depends on torchfold which is a pytorch tool developed by [Illia Polosukhin](https://github.com/ilblackdragon). It is used for dynamic batching the computations in a dynamic computation graph. The computations across all nodes of all trees are batched based on their module names and dispatched to GPU for parallelization. Download and install [pytorch-tools](https://github.com/nearai/pytorch-tools):
```
git clone https://github.com/nearai/pytorch-tools.git
python setup.py install
```

**Training**
```
python train.py
```
Arguments:
```
'--epochs' (number of epochs; default=300)
'--batch_size' (batch size; default=123 (the size of the provided training dataset is a multiple of 123))
'--show_log_every' (show training log for every X frames; default=3)
'--save_log' (save training log files)
'--save_log_every' (save training log for every X frames; default=3)
'--save_snapshot' (save snapshots of trained model)
'--save_snapshot_every' (save training log for every X frames; default=5)
'--no_plot' (don't show plots of losses)
'--no_cuda' (don't use cuda)
'--gpu' (device id of GPU to run cuda)
'--data_path' (dataset path, default='data')
'--save_path' (trained model path, default='models')
```

**Testing**
```
python test.py
```
This will sample a random noise vector of the same size as the root code. This random noise will be decoded into a tree structure of boxes and displayed using the utility functions in [draw3dobb.py](https://github.com/kevin-kaixu/grass_pytorch/blob/master/draw3dOBB.py) provided in this project.

## Citation
If you use this code, please cite the following paper.
```
@article {li_sig17,
	title = {GRASS: Generative Recursive Autoencoders for Shape Structures},
	author = {Jun Li and Kai Xu and Siddhartha Chaudhuri and Ersin Yumer and Hao Zhang and Leonidas Guibas},
	journal = {ACM Transactions on Graphics (Proc. of SIGGRAPH 2017)},
	volume = {36},
	number = {4},
	pages = {Article No. 52},
	year = {2017}
}
```

## Acknowledgement
This code uses the 'torchfold' in pytorch-tools developed by [Illia Polosukhin](https://github.com/ilblackdragon).
