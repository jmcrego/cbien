# -*- coding: utf-8 -*-
import logging
import sys
import os
import io
from collections import defaultdict
from utils import open_read

class Inputter():

    def __init__(self, fsrc, ftgt, token, vocab):
        self.fsrc = fsrc
        self.ftgt = ftgt
        self.token = token
        self.vocab = vocab
        self.stats_nskip = 0
        self.stats_nsent = 0
        self.stats_ntokens = 0
        self.stats_nOOV = 0

    def ngrams(self,toks):
        #returns the list of all ngrams in toks that do not contain <sep>
        ngrams = []
        for u in range(len(toks)):
            for n in range(1,self.vocab.max_ngram+1): #if max_ngram=3, n=[1,2,3]
                if u+n > len(toks):
                    break
                if self.vocab.str_sep in toks[u:u+n]:
                    break
                ngram = ' '.join(toks[u:u+n])
                ngrams.append(ngram)
                #print(u,n,ngram)
        return ngrams

    def __iter__(self):
        fs, is_gzip_src = open_read(self.fsrc)
        ft, is_gzip_tgt = open_read(self.ftgt)
        for ls,lt in zip(fs,ft):
            self.stats_nsent += 1
            if is_gzip_src:
                ls = ls.decode('utf8')
            if is_gzip_tgt:
                lt = lt.decode('utf8')
            ls = ls.strip(" \n")
            lt = lt.strip(" \n")

            SRC = self.token.tokenize(ls)
            TGT = self.token.tokenize(lt)
            if len(SRC) == 0 or SRC[0] == '':
                logging.debug('skip src empty sentence: <{}> {}:{}'.format(ls,self.fsrc,self.stats_nsent))
                self.stats_nskip += 1
                continue
            if len(TGT) == 0 or TGT[0] == '':
                logging.debug('skip tgt empty sentence: <{}> {}:{}'.format(lt,self.ftgt,self.stats_nsent))
                self.stats_nskip += 1
                continue

            toks = []
            idxs = []
            to_predict = [] #tokens to be predicted (all but: bos, eos, sep, unk)
            toks.append(self.vocab.str_bos)
            idxs.append(self.vocab.idx_bos)

            src_ok = False
            for tok in SRC:
                toks.append(self.vocab.tag_src+tok)
                idxs.append(self.vocab[toks[-1]])
                if idxs[-1] != self.vocab.idx_unk:
                    to_predict.append(len(toks)-1)
                    src_ok = True

            if not src_ok:
                #logging.debug('skip src sentence: <{}> {}:{}'.format(ls,self.fsrc,self.stats_nsent))
                self.stats_nskip += 1
                continue

            toks.append(self.vocab.str_eos)
            idxs.append(self.vocab.idx_eos)
            toks.append(self.vocab.str_sep)
            idxs.append(self.vocab.idx_sep)
            toks.append(self.vocab.str_bos)
            idxs.append(self.vocab.idx_bos)

            tgt_ok = False
            for tok in TGT:
                toks.append(self.vocab.tag_tgt+tok)
                idxs.append(self.vocab[toks[-1]])
                if idxs[-1] != self.vocab.idx_unk:
                    to_predict.append(len(toks)-1)
                    tgt_ok = True

            if not tgt_ok:
                #logging.debug('skip tgt sentence: <{}> {}:{}'.format(lt,self.ftgt,self.stats_nsent))
                self.stats_nskip += 1
                continue

            toks.append(self.vocab.str_eos)
            idxs.append(self.vocab.idx_eos)

            self.stats_ntokens += len(SRC) + len(TGT)
            self.stats_nOOV += toks.count(self.vocab.idx_unk)

            yield toks, idxs, to_predict

        logging.info('filtered {} out of {} sentences, {} tokens [{:.2f}% OOVs] in {},{}'.format(self.stats_nskip,self.stats_nsent,self.stats_ntokens,100.0*self.stats_nOOV/self.stats_ntokens,self.fsrc,self.ftgt))
        fs.close()
        ft.close()









