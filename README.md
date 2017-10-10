# GRASS in Pytorch
This is an Pytorch implementation of the paper "[GRASS: Generative Recursive Autoencoders for Shape Structures](http://kevinkaixu.net/projects/grass.html)". The paper is about learning a generative model for 3D shape structures by structural encoding and decoding with Recursive Neural Networks. This code was originally written by [Chenyang Zhu](http://www.sfu.ca/~cza68/) from Simon Fraser University and is being improved and maintained here in this repository.

Note that the current version implements only the Varational Auto-Encoder (VAE) part of the generative model. The implementation of the Generarive Adverserial Nets (GAN) part is still on-going. But this model can already generate novel 3D shape structures from sampled random noises. With GAN part, it is expected to generate more diverse structures.

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
