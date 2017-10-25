import torch
from torch import nn
from torch.autograd import Variable
import grassmodel
from draw3dobb import showGenshape


decoder = torch.load('./models/vae_decoder_model.pkl')


for i in range(10):
    root_code = Variable(torch.randn(1,80)).cuda()
    boxes = grassmodel.decode_structure(decoder, root_code)
    showGenshape(torch.cat(boxes,0).data.cpu().numpy())