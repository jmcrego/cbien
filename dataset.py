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
        line = '{} {}\n'.format( idx, ' '.join(map(str,ctx)))
        line.encode("utf-8")
        self.f.write(line)
        self.n += 1

    def close(self):
        self.f.close()

    def __len__(self):
        return self.n


class Dataset():

    def __init__(self, args, vocab, token):
        self.args = args
        self.vocab = vocab
        self.token = token

    def examples(self):
        self.stats_ngrams = defaultdict(int)
        e = Examples()
        e.open_write(self.args.name + '.examples.' + self.args.etag)
        stats_n_src = 0
        stats_n_tgt = 0
        stats_nunk_src = 0
        stats_nunk_tgt = 0
        nsent = 0
        file_pair = Inputter(self.args.data_src,self.args.data_tgt,self.token,self.vocab.max_ngram, self.vocab.str_sep, self.vocab.str_bos, self.vocab.str_eos, self.vocab.tag_src, self.vocab.tag_tgt)
        for sentence_tok in file_pair:
            nsent += 1

            sentence_idx = []
            to_predict = []
            is_src = True
            n_src = 0
            n_tgt = 0
            nunk_src = 0
            nunk_tgt = 0
            for i in range(len(sentence_tok)):
                sentence_idx.append(self.vocab[sentence_tok[i]])
                if sentence_idx[-1] == self.vocab.idx_sep:
                    is_src = False
                if sentence_idx[-1] > 4: # 0:<pad>, 1:<unk>, 2:<bos>, 3:<eos>, 4:<sep> not considered to be predicted
                    to_predict.append(i)
                    if is_src:
                        n_src += 1
                    else:
                        n_tgt += 1
                elif sentence_idx[-1] == self.vocab.idx_unk:
                    if is_src:
                        nunk_src += 1
                        n_src += 1
                    else:
                        nunk_tgt += 1
                        n_tgt += 1

            if n_src == 0 or n_tgt == 0:
                continue

            stats_n_src += n_src
            stats_n_tgt += n_tgt
            stats_nunk_src += nunk_src
            stats_nunk_tgt += nunk_tgt
            for c in to_predict: ### bos, eos, sep are not considered in to_predict set (c is the index in tok,idx of the token to predict)
                if random.random() > self.args.pkeep_example: #probability to keep an example
                    continue #discard this token (example)
                ctx = self.get_ctx(sentence_tok, c) #[idx, idx, ...]
                if len(ctx) == 0:
                    continue
                e.write(sentence_idx[c], ctx) #, self.args.etag+':nsent='+str(nsent)+':c='+str(c)+':tok='+sentence_tok[c]+':idx='+str(sentence_idx[c]))
            if nsent % 10000 == 0:
                logging.info('{} sentences => {} examples {}/{} tokens {:.2f}/{:.2f} %OOV'.format(nsent, len(e), stats_n_src,stats_n_tgt,100.0*stats_nunk_src/stats_n_src,100.0*stats_nunk_tgt/stats_n_tgt))
        logging.info('read {} sentences => {} examples {}/{} tokens {:.2f}/{:.2f} %OOV'.format(nsent, len(e), stats_n_src,stats_n_tgt,100.0*stats_nunk_src/stats_n_src,100.0*stats_nunk_tgt/stats_n_tgt))
        for n,N in sorted(self.stats_ngrams.items(), key=lambda item: item[0], reverse=False): 
            logging.info('{}-grams: {}'.format(n,N))

        e.close()

    def get_neg(self, idxs):
        neg = []
        while len(neg) < self.args.n_negs:
            idx = random.randint(5, len(self.vocab)-1) #do not consider special tokens (pad, unk, bos, eos, sep)
            if idx in idxs:
                continue
            neg.append(idx)
        return neg

    def get_ctx(self, toks, c):
        if self.args.window > 0:
            beg = max(c - self.args.window, 0)
            end = min(c + self.args.window + 1, len(toks))
        else:
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


    def get_batchs(self,fshard):
        ### read examples
        logging.info('reading examples in shard {}'.format(fshard))
        examples = []
        with gzip.open(fshard,'rb') as f:
            for l in f:
                l = l.decode('utf8')
                idx_ctx = l.rstrip().split('\t')
                idxs = [idx_ctx[0]]
                idxs.extend(idx_ctx[1].split(' '))
                examples.append(idxs)
                if len(examples) == 1:
                    print(examples[0])

         ### sort examples by len
        logging.info('sorting {} examples in shard (by length) to minimize padding'.format(len(examples)))
        length = [len(examples[k]) for k in range(len(examples))] #length of sentences in this shard
        index_examples = np.argsort(np.array(length)) ### These are indexs of examples

        logging.info('building batchs')
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
        logging.info('built {} batchs with up to {} examples each'.format(len(batchs),self.args.batch_size))
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
            fshards = glob.glob(self.args.name + '.shard_aaaaa.gz')
            random.shuffle(fshards)
            for fshard in fshards:

                batchs = self.get_batchs(fshard)
                logging.info('shuffling batchs...')
                random.shuffle(batchs) #shuffle batchs

                for batch in batchs:
                    print(len(batch[0]))
                    print(batch[0])
                    idx = np.array(batch[0])

                    print(len(batch[1]))
#                    print(batch[1][0])
                    neg = np.array(batch[1])

                    print(len(batch[2]))
#                    print(batch[2][0])
                    ctx = np.array(batch[2])

                    print(len(batch[3]))
#                    print(batch[3][0])
                    msk = np.array(batch[3])

                    yield idx, neg, ctx, msk

        ######################################################
        ### error ############################################
        ######################################################
        else:
            logging.error('bad -mode option {}'.format(self.mode))
            sys.exit()



