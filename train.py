import time

import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data
import torchfold
import util

from grassdata import GRASSDataset
from grassmodel import GRASSEncoder
from grassmodel import GRASSDecoder
import grassmodel


def class_collate(batch):
    return batch

config = util.get_args()
encoder = GRASSEncoder(config)
decoder = GRASSDecoder(config)
encoder.cuda()
decoder.cuda()


print("Loading data ...... ", end='', flush=True)
grass_data = GRASSDataset('./data')
train_iter = torch.utils.data.DataLoader(grass_data, batch_size=123, shuffle=True, collate_fn=class_collate)
print("DONE")

encoder_opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
decoder_opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)

print("Start training ...... ")

start = time.time()
best_dev_acc = -1
header = '  Time Epoch Iteration Progress    (%Epoch)   ReconLoss   KLDivLoss   TotalLoss'
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>10.6f},{:>8.6f},{:>8.6f}'.split(','))
#os.makedirs(args.save_path, exist_ok=True)
print(header)

for epoch in range(500):
    
#    if epoch % 100 == 0 and epoch != 0 :
#        torch.save(encoder, 'VAEencoder.pkl')
#       torch.save(decoder, 'VAEdecoder.pkl')

    n_total = train_loss = 0

    for batch_idx, batch in enumerate(train_iter):
        # Initialize torchfold for *encoding*
        fold = torchfold.Fold(cuda=True, variable=False)
        enc_fold_nodes = []     # list of fold nodes for encoding
        # Collect computation nodes recursively from encoding process
        for example in batch:
            enc_fold_nodes.append(grassmodel.encode_structure_fold(fold, example))
        # Apply the computations on the encoder model
        enc_fold_nodes = fold.apply(encoder, [enc_fold_nodes])
        # Split into fold nodes per example
        enc_fold_nodes = torch.split(enc_fold_nodes[0], 1, 0)
        # Initialize torchfold for *decoding*
        fold = torchfold.Fold(cuda=True, variable=True)
        dec_fold_nodes = []
        kld_fold_nodes = []
        for example, fnode in zip(batch, enc_fold_nodes):
            root_code, kl_div = torch.chunk(fnode, 2, 1)
            dec_fold_nodes.append(grassmodel.decode_structure_fold(fold, root_code, example))
            kld_fold_nodes.append(kl_div)
        # Apply the computations on the decoder model
        total_loss = fold.apply(decoder, [dec_fold_nodes, kld_fold_nodes])
            # the first 80 dims of total_loss is for reconstruction and the second 80 for KL divergence
        recon_loss = total_loss[0].sum() / len(batch)               # avg. reconstruction loss per example
        kldiv_loss = total_loss[1].sum().mul(-0.05) / len(batch)    # avg. KL divergence loss per example
        total_loss = recon_loss + kldiv_loss

        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        total_loss.backward()
        encoder_opt.step()
        decoder_opt.step()

        #if batch_idx % 1 == 0 :
            #print("reconstruction_error: %s; KLD_error: %s" % (str(err_re.data[0]), str(err_kld.data[0])))
        print(log_template.format(time.time()-start, epoch, batch_idx, 1+batch_idx, len(train_iter),
                100.*(1+batch_idx)/len(train_iter), recon_loss.data[0], kldiv_loss.data[0], total_loss.data[0]))

torch.save(encoder, 'VAEencoder.pkl')
torch.save(decoder, 'VAEdecoder.pkl')