from __future__ import absolute_import
import torch
from torch import nn
from torch.autograd import Variable
import grassmodel
from draw3dobb import showGenshape

#encoder = torch.load(u'./models/vae_encoder_model.pkl')
decoder = torch.load(u'./models/vae_decoder_model.pkl')


for i in xrange(10):
    test = Variable(torch.randn(1, 80)).cuda()
    boxes = grassmodel.decode_structure(decoder, test)
    showGenshape(torch.cat(boxes, 0).data.cpu().numpy())