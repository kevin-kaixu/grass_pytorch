import torch
from torch import nn
from torch.autograd import Variable
import model
import torchfold
from draw3dOBB import showGenshape

encoder = torch.load('VAEencoder.pkl')
decoder = torch.load('VAEdecoder.pkl')

for i in range(10):
    test = Variable(torch.rand(1,80).mul(2).add(-torch.ones(1,80))).cuda()
    boxes = model.decode_structure(decoder, test)
    showGenshape(torch.cat(boxes,0).data.cpu().numpy())