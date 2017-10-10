# GRASS in Pytorch
Implementation of GRASS with pytorch

This is an Pytorch implementation of the paper "GRASS: Generative Recursive Autoencoders for Shape Structures". The code was originally written by [Chenyang Zhu](http://www.sfu.ca/~cza68/) from Simon Fraser University and is being improved and maintained here in this repository.

Please refer the details to the [paper](http://kevinkaixu.net/projects/grass.html).

## Usage
**Dependancy**
grass_pytorch depends on torchfold which is a pytorch tool developed by [Illia Polosukhin](https://github.com/ilblackdragon). It is used for dynamic batching the computations in a dynamic computation graph. Download and install [pytorch-tools](https://github.com/nearai/pytorch-tools):
```
python setup.py install
```

**Training**
```
python train.py
```

**Testing**
```
python test.py
```

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
This code utilizes the pytorch tool 'torchfold' developed by...
