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
from collections import defaultdict, Counter
from utils import open_read
from inputter import Inputter

class Dataset():

    def __init__(self, args, vocab, token):
        self.args = args
        self.vocab = vocab
        self.token = token
#        self.n_negs = args.n_negs
#        self.window = args.window
#        self.pkeep_example = args.pkeep_example
#        self.batch_size = args.batch_size
#        self.idx_pad = vocab.idx_pad
#        self.idx_unk = vocab.idx_unk
#        self.max_ngram = vocab.max_ngram

    def examples(self):
        self.stats_ngrams = defaultdict(int)

        examples = []
        nsent = 0
        file_pair = Inputter(self.args.data_src,self.args.data_tgt,self.token,self.vocab)
        for sentence_tok, sentence_idx, to_predict in file_pair:
            nsent += 1
            for c in to_predict: ### bos, eos, sep are not considered in to_predict set
                if random.random() > self.args.pkeep_example: #probability to keep an example
                    continue #discard this token (example)
                neg = self.get_neg(sentence_idx) #[idx, idx, ...]
                ctx = self.get_ctx(sentence_tok, c) #[idx, idx, ...]
                if len(ctx) == 0:
                    continue
                examples.append([sentence_idx[c], neg, ctx])
            if nsent % 10000 == 0:
                logging.info('{} sentences => {} examples'.format(nsent, len(examples)))
        logging.info('read {} sentences => {} examples'.format(nsent, len(examples)))
        for n,N in sorted(self.stats_ngrams.items(), key=lambda item: item[0], reverse=False): 
            logging.info('{}-grams: {}'.format(n,N))
        logging.info('saving examples...')
        fd = open(self.args.name + '.examples.' + self.args.etag , 'wb') 
        pickle.dump(examples, fd)

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
                idx = self.vocab[' '.join(toks[first:lastplusone])]
                if idx == self.vocab.idx_unk: ### do not consider <unk>
                    break
                ctx.append(idx)
                self.stats_ngrams[lastplusone-first] += 1
        return ctx


    def batchs(self):
        ### read examples
        examples = []
        fexamples = glob.glob(self.args.name + '.examples.EMEA.en-fr') #'.examples.*')
        logging.info('reading examples from {}'.format(fexamples))
        for file in fexamples:
            with open(file,'rb') as f:
                curr_examples = pickle.load(f)
                examples.extend(curr_examples)
                logging.info('read {} examples from {}'.format(len(curr_examples),file))

        logging.info('shuffling examples...')
        random.shuffle(examples) #shuffle examples

        logging.info('batching...')
        batchs = []
        length = [len(examples[k][2]) for k in range(len(examples))] #length of ctx examples
        ind_examples = np.argsort(np.array(length)) ### These are indexs of examples sorted by length of its context
        batch = []
        for ind in ind_examples:
            batch.append(examples[ind])
            if len(batch) == self.args.batch_size:
                batchs.append(self.add_pad(batch))
                batch = []
        if len(batch):
            batchs.append(self.add_pad(batch)) ### this batch may have few examples
        logging.info('built {} batchs with up to {} examples each'.format(len(batchs),self.args.batch_size))

        logging.info('shuffling batches...')
        random.shuffle(batchs) #shuffle batchs

        logging.info('saving batches...')

        #fd = open(self.args.name + '.batchs', 'wb')
        #pickle.dump(batchs, fd)

        i = 0
        while True:
            first = i*self.args.shard_size
            last = min(len(batchs), (i+1)*self.args.shard_size)
            fd = open(self.args.name + '.batchs.' + str(i), 'wb')
            pickle.dump(batchs[first:last], fd)
            logging.info('saved {}'.format(self.args.name + '.batchs.' + str(i)))
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

    def add_pad2(self, batch):
        #batch[k]:
        ###[0] idx (word to predict)
        ###[1] [idx] n_negs (negative words)
        ###[2] [idx] ctx (context ngrams)
        max_ctx_len = max([len(x[2]) for x in batch])
        #logging.info('max_len={} lens: {}'.format(max_len, [len(x) for x in batch_ctx]))
        for k in range(len(batch)):
            ctx_len = len(batch[k][2])
            addn = max_ctx_len - ctx_len
            batch[k][2] += [self.vocab.idx_pad]*addn
            msk = [True]*ctx_len
            msk += [False]*addn
            batch[k].append(msk)
        return batch


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



