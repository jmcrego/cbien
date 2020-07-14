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

    def __init__(self, n_negs):
        self.n_negs = n_negs
        self.n = 0

    def open_write(self,fout):
        self.f = gzip.open(fout+'.gz', 'wt')

    def write(self, idx, neg, ctx, etag, nsent):
        if len(neg) != self.n_negs:
            logging.warning('bad number of negative examples {} should be {}'.format(len(neg), self.n_negs))
            return
        if len(ctx) == 0:
            logging.warning('empty context')
            return
        line = '{}\t{}:{}\t{}\t{}\t{}\n'.format(len(ctx), etag, nsent, idx, ' '.join(map(str, neg)), ' '.join(map(str,ctx)))
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
        e = Examples(self.args.n_negs)
        e.open_write(self.args.name + '.examples.' + self.args.etag)
        nsent = 0
        file_pair = Inputter(self.args.data_src,self.args.data_tgt,self.token,self.vocab)
        for sentence_tok, sentence_idx, to_predict in file_pair:
            nsent += 1
#            print('nsent',str(nsent))
#            print('tok',sentence_tok)
#            print('idx',sentence_idx)
#            print('pred',to_predict)
            for c in to_predict: ### bos, eos, sep are not considered in to_predict set (c is the index in tok,idx of the token to predict)
                if random.random() > self.args.pkeep_example: #probability to keep an example
                    continue #discard this token (example)
                neg = self.get_neg(sentence_idx) #[idx, idx, ...]
#                print('c={}'.format(c))
                ctx = self.get_ctx(sentence_tok, c) #[idx, idx, ...]
                if len(ctx) == 0:
                    continue
                e.write(sentence_idx[c], neg, ctx, self.args.etag, nsent)
            if nsent % 10000 == 0:
                logging.info('{} sentences => {} examples'.format(nsent, len(e)))
        logging.info('read {} sentences => {} examples'.format(nsent, len(e)))
        for n,N in sorted(self.stats_ngrams.items(), key=lambda item: item[0], reverse=False): 
            logging.info('{}-grams: {}'.format(n,N))

        e.close()
        #logging.info('saving examples...')
        #fd = open(self.args.name + '.examples.' + self.args.etag , 'wb') 
        #pickle.dump(examples, fd)

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
#                print(idx, ' '.join(toks[first:lastplusone]))
                self.stats_ngrams[lastplusone-first] += 1

        return ctx


    def batchs(self):
        ### read examples
        logging.info('reading examples')

        batchs = []
        batch = []
        with gzip.open(self.args.name + '.examples.gz','rb') as f:        
            for l in f:
                l = l.decode('utf8')
                idx, neg, ctx = l.split('\t')
                batch.append([ int(idx), map(int, neg.split(' ')), map(int, ctx.split(' ')) ])
                if len(batch) == self.args.batch_size:
                    batchs.append(self.add_pad(batch))
                    batch = []
            if len(batch):
                batchs.append(self.add_pad(batch)) ### this batch may have few examples
            logging.info('built {} batchs with up to {} examples each'.format(len(batchs),self.args.batch_size))

#        logging.info('shuffling {} examples...'.format(len(examples)))
#        random.shuffle(examples) #shuffle examples

#        logging.info('batching...')
#        batchs = []
#        length = [len(examples[k][2]) for k in range(len(examples))] #length of ctx examples
#        ind_examples = np.argsort(np.array(length)) ### These are indexs of examples sorted by length of its context

#        batch = []
#        for ind in ind_examples:
#            batch.append(examples[ind])
#            if len(batch) == self.args.batch_size:
#                batchs.append(self.add_pad(batch))
#                batch = []
#        if len(batch):
#            batchs.append(self.add_pad(batch)) ### this batch may have few examples
#        logging.info('built {} batchs with up to {} examples each'.format(len(batchs),self.args.batch_size))

        logging.info('shuffling batches...')
        random.shuffle(batchs) #shuffle batchs

        logging.info('saving batches...')
        i = 0
        while True:
            first = i*self.args.shard_size
            last = min(len(batchs), (i+1)*self.args.shard_size)
            fd = open(self.args.name + '.batchs.shard' + str(i), 'wb')
            pickle.dump(batchs[first:last], fd)
            logging.info('saved {} with {} batchs'.format(self.args.name + '.batchs.shard' + str(i), last-first))
            i += 1
            if last == len(batchs):
                break

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
            with open(self.args.name + '.batchs','rb') as f:
                batchs = pickle.load(f)
                logging.info('read {} batchs'.format(len(batchs)))
                for batch in batchs:
                    batch = np.array(batch)
                    print(batch.shape)
                    print(batch[0])
                    sys.exit()
                    yield batch #batch[k] contains [idx, [n_negs], [ctx], [msk]]

        ######################################################
        ### error ############################################
        ######################################################
        else:
            logging.error('bad -mode option {}'.format(self.mode))
            sys.exit()



