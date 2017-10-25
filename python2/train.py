from __future__ import division
from __future__ import absolute_import
import time
import os
from time import gmtime, strftime
from datetime import datetime
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data
from torchfoldext import FoldExt
import util
from dynamicplot import DynamicPlot

from grassdata import GRASSDataset
from grassmodel import GRASSEncoder
from grassmodel import GRASSDecoder
import grassmodel
from io import open
from itertools import izip


config = util.get_args()

config.cuda = not config.no_cuda
if config.gpu<0 and config.cuda:
    config.gpu = 0
torch.cuda.set_device(config.gpu)
if config.cuda and torch.cuda.is_available():
    print u"Using CUDA on GPU ", config.gpu
else:
    print u"Not using CUDA."

encoder = GRASSEncoder(config)
decoder = GRASSDecoder(config)
if config.cuda:
    encoder.cuda()
    decoder.cuda()


print u"Loading data ...... "
grass_data = GRASSDataset(config.data_path)
def my_collate(batch):
    return batch
train_iter = torch.utils.data.DataLoader(grass_data, batch_size=config.batch_size, shuffle=True, collate_fn=my_collate)
print u"DONE"

encoder_opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
decoder_opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)

print u"Start training ...... "

start = time.time()

if config.save_snapshot:
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    snapshot_folder = os.path.join(config.save_path, 'snapshots_'+strftime("%Y-%m-%d_%H-%M-%S",gmtime()))
    if not os.path.exists(snapshot_folder):
        os.makedirs(snapshot_folder)

if config.save_log:
    fd_log = open(u'training_log.log', mode=u'a')
    fd_log.write(u'\n\nTraining log at '+datetime.now().strftime(u'%Y-%m-%d %H:%M:%S'))
    fd_log.write(u'\n#epoch: {}'.format(config.epochs))
    fd_log.write(u'\nbatch_size: {}'.format(config.batch_size))
    fd_log.write(u'\ncuda: {}'.format(config.cuda))
    fd_log.flush()

header = u'     Time    Epoch     Iteration    Progress(%)  ReconLoss  KLDivLoss  TotalLoss'
log_template = u' '.join(u'{:>9s},{:>5.0f}/{:<5.0f},{:>5.0f}/{:<5.0f},{:>9.1f}%,{:>11.2f},{:>10.2f},{:>10.2f}'.split(u','))

total_iter = config.epochs * len(train_iter)

if not config.no_plot:
    plot_x = [x for x in xrange(total_iter)]
    plot_total_loss = [None for x in xrange(total_iter)]
    plot_recon_loss = [None for x in xrange(total_iter)]
    plot_kldiv_loss = [None for x in xrange(total_iter)]
    dyn_plot = DynamicPlot(title=u'Training loss over epochs (GRASS)', xdata=plot_x, ydata={u'Total_loss':plot_total_loss, u'Reconstruction_loss':plot_recon_loss, u'KL_divergence_loss':plot_kldiv_loss})
    iter_id = 0
    max_loss = 0
    
for epoch in xrange(config.epochs):
    print header
    for batch_idx, batch in enumerate(train_iter):
        # Initialize torchfold for *encoding*
        enc_fold = FoldExt(cuda=config.cuda)
        enc_fold_nodes = []     # list of fold nodes for encoding
        # Collect computation nodes recursively from encoding process
        for example in batch:
            enc_fold_nodes.append(grassmodel.encode_structure_fold(enc_fold, example))
        # Apply the computations on the encoder model
        enc_fold_nodes = enc_fold.apply(encoder, [enc_fold_nodes])
        # Split into a list of fold nodes per example
        enc_fold_nodes = torch.split(enc_fold_nodes[0], 1, 0)
        # Initialize torchfold for *decoding*
        dec_fold = FoldExt(cuda=config.cuda)
        # Collect computation nodes recursively from decoding process
        dec_fold_nodes = []
        kld_fold_nodes = []
        for example, fnode in izip(batch, enc_fold_nodes):
            root_code, kl_div = torch.chunk(fnode, 2, 1)
            dec_fold_nodes.append(grassmodel.decode_structure_fold(dec_fold, root_code, example))
            kld_fold_nodes.append(kl_div)
        # Apply the computations on the decoder model
        total_loss = dec_fold.apply(decoder, [dec_fold_nodes, kld_fold_nodes])
            # the first dim of total_loss is for reconstruction and the second for KL divergence
        recon_loss = total_loss[0].sum() / len(batch)               # avg. reconstruction loss per example
        kldiv_loss = total_loss[1].sum().mul(-0.05) / len(batch)    # avg. KL divergence loss per example
        total_loss = recon_loss + kldiv_loss
        # Do parameter optimization
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        total_loss.backward()
        encoder_opt.step()
        decoder_opt.step()
        # Report statistics
        if batch_idx % config.show_log_every == 0:
            print log_template.format(strftime(u"%H:%M:%S",time.gmtime(time.time()-start)),
                epoch, config.epochs, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx+len(train_iter)*epoch) / (len(train_iter)*config.epochs),
                recon_loss.data[0], kldiv_loss.data[0], total_loss.data[0])
        # Plot losses
        if not config.no_plot:
            plot_total_loss[iter_id] = total_loss.data[0]
            plot_recon_loss[iter_id] = recon_loss.data[0]
            plot_kldiv_loss[iter_id] = kldiv_loss.data[0]
            max_loss = max(max_loss, total_loss.data[0], recon_loss.data[0], kldiv_loss.data[0])
            dyn_plot.setxlim(0., (iter_id+1)*1.05)
            dyn_plot.setylim(0., max_loss*1.05)
            dyn_plot.update_plots(ydata={u'Total_loss':plot_total_loss, u'Reconstruction_loss':plot_recon_loss, u'KL_divergence_loss':plot_kldiv_loss})
            iter_id += 1

    # Save snapshots of the models being trained
    if config.save_snapshot and (epoch+1) % config.save_snapshot_every == 0 :
        print u"Saving snapshots of the models ...... "
        torch.save(encoder, snapshot_folder+u'//vae_encoder_model_epoch_{}_loss_{:.2f}.pkl'.format(epoch+1, total_loss.data[0]))
        torch.save(decoder, snapshot_folder+u'//vae_decoder_model_epoch_{}_loss_{:.2f}.pkl'.format(epoch+1, total_loss.data[0]))
        print u"DONE"
    # Save training log
    if config.save_log and (epoch+1) % config.save_log_every == 0 :
        fd_log = open(u'training_log.log', mode=u'a')
        fd_log.write(u'\nepoch:{} recon_loss:{:.2f} kld_loss:{:.2f} total_loss:{:.2f}'.format(epoch+1, recon_loss.data[0], kldiv_loss.data[0], total_loss.data[0]))
        fd_log.close()

# Save the final models
print u"Saving final models ...... "
torch.save(encoder, config.save_path+u'//vae_encoder_model.pkl')
torch.save(decoder, config.save_path+u'//vae_decoder_model.pkl')
print u"DONE"