# -*- coding: utf-8 -*-
import logging
import yaml
import sys
import os
import io
import math
import glob
import random
import itertools
import pyonmttok
import pickle 
import numpy as np
import gzip
from collections import defaultdict, Counter
from utils import open_read
from inputter import Inputter

class Examples():

    def __init__(self):
        self.n = 0

    def open_write(self,fout):
        self.f = gzip.open(fout+'.gz', 'wt')

    def write(self, idx, ctx):
        if len(ctx) == 0:
            logging.warning('empty context')
            return
        line = '{} {}\n'.format(idx, ' '.join(map(str,ctx)))
        line.encode("utf-8")
        self.f.write(line)
        self.n += 1

    def close(self):
        self.f.close()

    def __len__(self):
        return self.n


class Dataset():

    def __init__(self, args, vocab, token, isValid=False):
        self.args = args
        self.vocab = vocab
        self.token = token
        self.isValid = isValid

        self.n_src = 0
        self.n_tgt = 0
        self.nunk_src = 0
        self.nunk_tgt = 0
        self.stats_ngrams = defaultdict(int)

    def toks2idxs(self, toks):
        idxs = [] ### idx's of toks
        n_src = 0 ### number of source tokens (other than <pad> <bos> <eos> <sep>)
        n_tgt = 0 ### number of target tokens (other than <pad> <bos> <eos> <sep>)
        is_src = True
        for i in range(len(toks)):
            idxs.append(self.vocab[toks[i]])
            if idxs[-1] == self.vocab.idx_sep:
                is_src = False
            if idxs[-1] != self.vocab.idx_pad and idxs[-1] != self.vocab.idx_bos and idxs[-1] != self.vocab.idx_eos and idxs[-1] != self.vocab.idx_sep:
                if is_src:
                    if idxs[-1] == self.vocab.idx_unk:
                        self.nunk_src += 1
                    self.n_src += 1
                    n_src += 1
                else: #is_tgt
                    if idxs[-1] == self.vocab.idx_unk:
                        self.nunk_tgt += 1
                    self.n_tgt += 1
                    n_tgt += 1
        return idxs, n_src, n_tgt


    def examples(self):
        stats_n_src = 0
        stats_n_tgt = 0
        stats_nunk_src = 0
        stats_nunk_tgt = 0
        e = Examples()
        e.open_write(self.args.name + '.examples.' + self.args.etag)
        nsent = 0
        file_pair = Inputter(self.args.data_src,self.args.data_tgt,self.token,self.vocab.max_ngram, self.vocab.str_sep, self.vocab.str_bos, self.vocab.str_eos, self.vocab.tag_src, self.vocab.tag_tgt)
        for toks in file_pair:
            #toks is either:
            #################
            #[<bos>, the, white, house, <eos>, <sep>, <bos>, la, maison, blanche, <eos>]
            #[<bos>, the, white, house, <eos>, <sep>, <bos>, <eos>]
            #[<bos>, <eos>, <sep>, <bos>, la, maison, blanche, <eos>]
            nsent += 1
            idxs, n_src, n_tgt = self.toks2idxs(toks)
            if self.args.data_src is not None and n_src == 0:
                continue
            if self.args.data_tgt is not None and n_tgt == 0:
                continue

            for c in range(len(idxs)): ### bos, eos, sep are not considered in to_predict set (c is the index in tok,idx of the token to predict)
                if random.random() > self.args.pkeep: #probability to keep an example
                    continue #discard this token (example)
                if idxs[c] < 5: ###0:<pad>, 1:<unk>, 2:<bos>, 3:<eos>, 4:<sep> are not predictable
                    continue
                ctx = self.get_ctx(toks, c) #[idx, idx, ...]
                if len(ctx) == 0:
                    continue
                e.write(idxs[c], ctx)

            if nsent % 10000 == 0:
                if self.args.data_src is not None:
                    logging.info('{} sentences => {} examples {} src tokens {:.2f} %OOV'.format(nsent, len(e),self.n_src,100.0*self.nunk_src/self.n_src))
                if self.args.data_tgt is not None:
                    logging.info('{} sentences => {} examples {} tgt tokens {:.2f} %OOV'.format(nsent, len(e),self.n_tgt,100.0*self.nunk_tgt/self.n_tgt))
                for n,N in sorted(self.stats_ngrams.items(), key=lambda item: item[0], reverse=False): 
                    logging.info('{}-grams: {}'.format(n,N))

        e.close()

        if self.args.data_src is not None:
            logging.info('total {} sentences => {} examples {} src tokens {:.2f} %OOV'.format(nsent, len(e),self.n_src,100.0*self.nunk_src/self.n_src))
        if self.args.data_tgt is not None:
            logging.info('total {} sentences => {} examples {} tgt tokens {:.2f} %OOV'.format(nsent, len(e),self.n_tgt,100.0*self.nunk_tgt/self.n_tgt))
        for n,N in sorted(self.stats_ngrams.items(), key=lambda item: item[0], reverse=False): 
            logging.info('{}-grams: {}'.format(n,N))


    def get_neg(self, idxs):
        neg = []
        while len(neg) < self.args.n_negs:
            idx = random.randint(5, len(self.vocab)-1) #do not consider special tokens (pad, unk, bos, eos, sep)
            if idx in idxs:
                continue
            neg.append(idx)
        return neg

    def get_ctx(self, toks, c): #c==-1 indicates inference
        if self.args.window > 0 and c >= 0:
            beg = max(c - self.args.window, 0)
            end = min(c + self.args.window + 1, len(toks))
        else: #context is all sentence if window==0 or inference (c==-1)
            beg = 0
            end = len(toks)

        ctx = []
        for first in range(beg, end): # find all ngrams within [beg, end)

            for lastplusone in range(first+1,first+self.vocab.max_ngram+1): 
                if lastplusone > len(toks): ### out of bounds
                    break
                if first<=c and lastplusone>c: ### do not consider ngrams containing the center
                    break
                if self.vocab.str_sep in toks[first:lastplusone]: ### do not consier ngrams with <sep> in it
                    break

                idx = self.vocab[' '.join(toks[first:lastplusone])]
                if idx == self.vocab.idx_unk or idx == self.vocab.idx_bos or idx == self.vocab.idx_eos or idx == self.vocab.idx_sep: ### do not consider <unk>, <bos>, <eos>, <sep>
                    break

                ctx.append(idx)
                self.stats_ngrams[lastplusone-first] += 1

        return ctx


    def get_batchs(self,fshard,n,N):
        ### read examples
        logging.info('reading examples in shard {}/{} {}'.format(n,N,fshard))
        examples = []
        if fshard.endswith('.gz'):
            with gzip.open(fshard,'rb') as f:
                for l in f:
                    l = l.decode('utf8')
                    examples.append(l.rstrip().split(' '))
        else:
            with open(file, mode='r', encoding='utf-8') as f:
                for l in f:
                    l = l.decode('utf8')
                    examples.append(l.rstrip().split(' '))

         ### sort examples by len
        logging.info('found {} examples in shard (shuffling...)'.format(len(examples)))
        length = [len(examples[k]) for k in range(len(examples))] #length of sentences in this shard
        index_examples = np.argsort(np.array(length)) ### These are indexs of examples

        logging.info('building batchs sized of {} examples'.format(self.args.batch_size))
        batchs = []
        batch = []
        for index in index_examples:
            idx_ctx = examples[index]
            neg = self.get_neg(idx_ctx)
            batch.append([ int(idx_ctx.pop(0)), list(map(int, neg)), list(map(int, idx_ctx)) ])
            if len(batch) == self.args.batch_size:
                batchs.append(self.add_pad(batch))
                batch = []
        if len(batch):
            batchs.append(self.add_pad(batch)) ### this batch may have few examples
        logging.info('found {} batchs'.format(len(batchs)))
        return batchs


    def add_pad(self, batch):
        batch_idx = []
        batch_neg = []
        batch_ctx = []
        batch_msk = []
        max_ctx_len = max([len(e[2]) for e in batch])
        for e in batch:
            ctx_len = len(e[2])
            addn = max_ctx_len - ctx_len
            batch_idx.append(e[0])
            batch_neg.append(e[1])
            e[2] += [self.vocab.idx_pad]*addn
            batch_ctx.append(e[2])
            msk = [True]*ctx_len
            msk += [False]*addn
            batch_msk.append(msk)
        return [batch_idx, batch_neg, batch_ctx, batch_msk]

    def __iter__(self):
        ######################################################
        ### train ############################################
        ######################################################
        if self.args.mode == 'train':
            if self.isValid:
                fshards = glob.glob(self.args.name + '.valid_?????.gz')
                if len(fshards) == 0:
                    fshards = glob.glob(self.args.name + '.valid_?????')
            else:
                fshards = glob.glob(self.args.name + '.shard_?????.gz')
                #print(fshards)

            random.shuffle(fshards)
            for n,fshard in enumerate(fshards):

                batchs = self.get_batchs(fshard,n+1,len(fshards))
                logging.info('shuffling batchs...')
                random.shuffle(batchs) #shuffle batchs
                logging.info('iterating over {} batchs'.format(len(batchs)))

                for batch in batchs:
                    idx = np.array(batch[0])
                    neg = np.array(batch[1])
                    ctx = np.array(batch[2])
                    msk = np.array(batch[3])
                    yield idx, neg, ctx, msk

        ######################################################
        ### error ############################################
        ######################################################
        else:
            logging.error('bad -mode option {}'.format(self.mode))
            sys.exit()



