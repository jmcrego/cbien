# -*- coding: utf-8 -*-
import torch
import logging
import yaml
import sys
import os
import io
import math
import random
import itertools
import pyonmttok
import glob
import numpy as np
import torch.nn as nn
from os import path
from collections import Counter
from dataset import Dataset
from vocab import Vocab
from tokenizer import OpenNMTTokenizer

min_sigmoid = 1e-06
max_sigmoid = 1.0 - 1e-06

def save_model(pattern, model, n_steps, keep_last_n):
    file = pattern + '.model.{:09d}.pth'.format(n_steps)
    state = {
        'pooling': model.pooling,
        'embedding_size': model.ds,
        'vocab_size': model.vs,
        'idx_pad': model.idx_pad,
        'n_steps': n_steps,
        'model': model.state_dict()
    }
    torch.save(state, file)
    logging.info('saved model checkpoint {}'.format(file))
    files = sorted(glob.glob(pattern + '.model.?????????.pth')) 
    while keep_last_n > 0 and len(files) > keep_last_n:
        f = files.pop(0)
        os.remove(f) ### first is the oldest
        logging.debug('removed checkpoint {}'.format(f))

def load_model(pattern, vocab):
    model = None

    n_steps = 0
    file = pattern + '.model.pth' ### I try first the best model
    if not path.isfile(file):
        files = sorted(glob.glob(pattern + '.model.?????????.pth')) ### I check if there is one model
        if len(files) == 0:
            return model, n_steps
        file = files[-1] ### last is the newest

    checkpoint = torch.load(file)
    pooling = checkpoint['pooling']
    embedding_size = checkpoint['embedding_size']
    n_steps = checkpoint['n_steps']
    vocab_size = checkpoint['vocab_size']
    if vocab_size != len(vocab):
        logging.error('incompatible vocabulary size {} != {}'.format(vocab_size, len(vocab)))
        sys.exit()
    idx_pad = checkpoint['idx_pad']
    if idx_pad != vocab.idx_pad:
        logging.error('incompatible idx_pad {} != {}'.format(idx_pad, vocab.idx_pad))
        sys.exit()
    model = Word2Vec(vocab_size, embedding_size, pooling, idx_pad)
    model.load_state_dict(checkpoint['model'])
    logging.info('loaded checkpoint {} [voc:{},emb:{}] pooling={} n_steps={}'.format(file,len(vocab),embedding_size,pooling,n_steps))
    return model, n_steps

def save_model_best(pattern, model, n_steps, min_loss):
    file = pattern + '.model.pth'
    state = {
        'pooling': model.pooling,
        'embedding_size': model.ds,
        'vocab_size': model.vs,
        'idx_pad': model.idx_pad,
        'n_steps': n_steps,
        'min_loss': min_loss,
        'model': model.state_dict()
    }
    torch.save(state, file)
    logging.info('saved model checkpoint {}'.format(file))


def load_model_best(pattern, vocab):
    model = None
    n_steps = 0
    file = pattern + '.model.pth' 

    if path.isfile(file):
        checkpoint = torch.load(file, map_location=torch.device('cpu'))
        pooling = checkpoint['pooling']
        embedding_size = checkpoint['embedding_size']
        vocab_size = checkpoint['vocab_size']
        if vocab_size != len(vocab):
            logging.error('incompatible vocabulary size {} != {}'.format(vocab_size, len(vocab)))
            sys.exit()
        idx_pad = checkpoint['idx_pad']
        if idx_pad != vocab.idx_pad:
            logging.error('incompatible idx_pad {} != {}'.format(idx_pad, vocab.idx_pad))
            sys.exit()
        n_steps = checkpoint['n_steps']
        min_loss = checkpoint['min_loss']
        model = Word2Vec(vocab_size, embedding_size, pooling, idx_pad)
        model.load_state_dict(checkpoint['model'])
        logging.info('loaded best checkpoint {} [voc:{},emb:{}] pooling={} n_steps={} min_loss={:.6f}'.format(file,len(vocab),embedding_size,pooling,n_steps,min_loss))
    return model, n_steps

def save_optim(pattern, optimizer):
    file = pattern + '.optim.pth'
    state = {
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, file)
    logging.info('saved optim in {}'.format(file))

def load_build_optim(pattern, model, lr, beta1, beta2, eps):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=0.01, amsgrad=False)
    file = pattern + '.optim.pth'
    if os.path.exists(file): 
        checkpoint = torch.load(file)
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info('loaded optimizer from {}.optim.pth'.format(pattern))
    else:
        logging.info('build optimizer from scratch')
    return optimizer

def sequence_mask(lengths):
    lengths = np.array(lengths)
    bs = len(lengths)
    l = lengths.max()
    msk = np.cumsum(np.ones([bs,l],dtype=int), axis=1).T #[l,bs] (transpose to allow combine with lenghts)
    mask = (msk <= lengths) ### i use lenghts-1 because the last unpadded word is <eos> and i want it masked too
    return mask.T #[bs,l]

####################################################################
### Word2Vec #######################################################
####################################################################
class Word2Vec(nn.Module):
    def __init__(self, vs, ds, pooling, idx_pad):
        super(Word2Vec, self).__init__()
        self.vs = vs
        self.ds = ds
        self.pooling = pooling
        self.idx_pad = idx_pad
        self.iEmb = nn.Embedding(self.vs, self.ds, padding_idx=self.idx_pad)
        self.oEmb = nn.Embedding(self.vs, self.ds, padding_idx=self.idx_pad)
        #nn.init.xavier_uniform_(self.iEmb.weight)
        #nn.init.xavier_uniform_(self.oEmb.weight)
        nn.init.uniform_(self.iEmb.weight, -0.1, 0.1)
        nn.init.uniform_(self.oEmb.weight, -0.1, 0.1)

    def WordEmbed(self, wrd, layer):
        wrd = torch.as_tensor(wrd)
        if wrd.type() != 'torch.LongTensor':
            logging.error('bad wrd type {}'.format(wrd.type()))
            sys.exit()
        if self.iEmb.weight.is_cuda:
            wrd = wrd.cuda()

        if layer == 'iEmb':
            emb = self.iEmb(wrd) #[bs,ds]
        elif layer == 'oEmb':
            emb = self.oEmb(wrd) #[bs,ds]
        else:
            logging.error('bad layer {}'.format(layer))
            sys.exit()
 
#        if torch.isnan(emb).any() or torch.isinf(emb).any():
#            logging.error('NaN/Inf detected in {} layer emb.shape={}\nwrds {}'.format(layer,emb.shape,wrd))
#            sys.exit()
        return emb

    def NgramsEmbed(self, ngrams, msk):
        ngrams_emb = self.WordEmbed(ngrams,'iEmb') #[bs,n,ds]
        if self.pooling == 'avg':
            ngrams_emb = (ngrams_emb*msk.unsqueeze(-1)).sum(1) / torch.sum(msk, dim=1).unsqueeze(-1) #[bs,n,ds]x[bs,n,1]=>[bs,ds] / [bs,1] = [bs,ds] 
        elif self.pooling == 'sum':
            ngrams_emb = (ngrams_emb*msk.unsqueeze(-1)).sum(1) #[bs,n,ds]x[bs,n,1]=>[bs,ds]
        elif self.pooling == 'max':
            ngrams_emb, _ = torch.max(ngrams_emb*msk + (1.0-msk)*-999.9, dim=1) #-999.9 should be -Inf but it produces a nan when multiplied by 0.0            
        else:
            logging.error('bad -pooling option {}'.format(self.pooling))
            sys.exit()
        return ngrams_emb

    def forward(self, batch_idx, batch_neg, batch_ctx, batch_msk):
        idx = torch.as_tensor(batch_idx) #batch of center (list:bs)
        BS = idx.size()[0]
        neg = torch.as_tensor(batch_neg) #batch of context (list:bs of list:nn)
        NN = neg.size()[1]
        assert BS == neg.size()[0] 
        ctx = torch.as_tensor(batch_ctx) #batch of negative (list:bs of list:nc)
        NC = ctx.size()[1]
        assert BS == ctx.size()[0] 
        msk = torch.as_tensor(batch_msk) #batch of contex masks (list:bs of list:nc)
        assert BS == msk.size()[0] 
        assert NC == msk.size()[1] 
#        logging.info('BS={}, NN={}, NC={}'.format(BS,NN,NC))

        if msk.type() != 'torch.BoolTensor':
            logging.error('bad msk type {}'.format(msk.type()))
            sys.exit()
        if self.iEmb.weight.is_cuda:
            msk = msk.cuda()

        ###
        ### Context words are embedded using iEmb
        ###
        ctx_emb = self.NgramsEmbed(ctx, msk) #[bs,ds]
        DS = ctx_emb.size()[1]
        assert BS == ctx_emb.size()[0]
#        logging.info('DS={}'.format(DS))
        if torch.isnan(ctx_emb).any() or torch.isinf(ctx_emb).any():
            logging.error('NaN/Inf detected in ctx_emb')
            sys.exit()
        ###
        ### Center words are embedded using oEmb
        ###
        wrd_emb = self.WordEmbed(idx,'oEmb') #[bs,ds]
        assert BS == wrd_emb.size()[0]
        assert DS == wrd_emb.size()[1]
        if torch.isnan(wrd_emb).any() or torch.isinf(wrd_emb).any():
            logging.error('NaN/Inf detected in wrd_emb')
            sys.exit()
        ###
        ### Negative words are embedded using oEmb
        ###
        neg_emb = self.WordEmbed(neg,'oEmb').neg() #[bs,nn,ds]
        assert BS == neg_emb.size()[0]
        assert NN == neg_emb.size()[1]
        assert DS == neg_emb.size()[2]
        if torch.isnan(neg_emb).any() or torch.isinf(neg_emb).any():
            logging.error('NaN/Inf detected in neg_emb')
            sys.exit()
        ###
        ### Computing positive words loss
        ###
        #i use clamp to prevent NaN/Inf appear when computing the log of 1.0/0.0
        ctx_emb_2 = ctx_emb.unsqueeze(1)  #[bs,1,ds]
        #logging.info('ctx_emb_2.size={}'.format(ctx_emb_2.size()))
        assert BS == ctx_emb_2.size()[0]
        assert DS == ctx_emb_2.size()[2]
        wrd_emb_2 = wrd_emb.unsqueeze(-1) #[bs,ds,1]
        #logging.info('wrd_emb_2.size={}'.format(wrd_emb_2.size()))
        assert BS == wrd_emb_2.size()[0]
        assert DS == wrd_emb_2.size()[1]
        bmm = torch.bmm(ctx_emb_2, wrd_emb_2) #[bs,1,1]
        #logging.info('bmm.size={}'.format(bmm.size()))
        assert BS == bmm.size()[0]
        bmm_2 = bmm.squeeze(2).squeeze(1) #[bs] (do not squeeze axis=0 when bs=1)
        #logging.info('bmm_2.size={}'.format(bmm_2.size()))
        assert BS == bmm_2.size()[0]
        err = bmm_2.sigmoid().clamp(min_sigmoid, max_sigmoid).log().neg() #[bs]
#        err = torch.bmm(ctx_emb.unsqueeze(1), wrd_emb.unsqueeze(-1)).squeeze().sigmoid().clamp(min_sigmoid, max_sigmoid).log().neg() #[bs,1,ds] x [bs,ds,1] = [bs,1,1] => [bs]
#        logging.info('err.size={} BS={}'.format(err.size(),BS))
        assert BS == err.size()[0]
        if torch.isnan(err).any() or torch.isinf(err).any():
            logging.error('NaN/Inf detected in positive words err={}'.format(err))
            sys.exit()
        loss = err.mean() # mean errors of examples in this batch
        ###
        ### Computing negative words loss
        ###
        ctx_emb_2 = ctx_emb.unsqueeze(1)   #[bs,1,ds]
        neg_emb_2 = neg_emb.transpose(2,1) #[bs,ds,nn]
        bmm = torch.bmm(ctx_emb_2, neg_emb_2) #[bs,1,nn]
        bmm_2 = bmm.squeeze(1) #[bs,nn] do not squeeze axis=0 when bs=1
        err = bmm_2.sigmoid().clamp(min_sigmoid, max_sigmoid).log().neg() #[bs,nn]
#        err = torch.bmm(ctx_emb.unsqueeze(1), neg_emb.transpose(2,1)).squeeze().sigmoid().clamp(min_sigmoid, max_sigmoid).log().neg() #[bs,1,ds] x [bs,ds,nn] = [bs,1,nn] = > [bs,nn]
        assert BS == err.size()[0]
        assert NN == err.size()[1]
        err = torch.sum(err, dim=1) #[bs] (sum of errors of all negative words) (not averaged)
        if torch.isnan(err).any() or torch.isinf(err).any():
            logging.error('NaN/Inf detected in negative words err={}'.format(err))
            sys.exit()
        loss += err.mean()

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logging.error('NaN/Inf detected in loss')
            sys.exit()
        return loss






