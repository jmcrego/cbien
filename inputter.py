# -*- coding: utf-8 -*-
import logging
import sys
import os
import io
from collections import defaultdict
from utils import open_read

class Inputter():

    def __init__(self, fsrc, ftgt, token, max_ngram, str_sep, str_bos, str_eos, tag_src, tag_tgt, do_filter=True):
        self.fsrc = fsrc
        self.ftgt = ftgt
        self.token = token
        self.max_ngram = max_ngram
        self.str_sep = str_sep
        self.str_bos = str_bos
        self.str_eos = str_eos
        self.tag_src = tag_src
        self.tag_tgt = tag_tgt
        self.do_filter = do_filter

        self.stats_nsent = 0
        self.stats_ntokens = 0
        self.stats_nskip = 0

        if self.fsrc is None:
            logging.info('empty src file')
        if self.ftgt is None:
            logging.info('empty tgt file')


    def ngrams(self,toks):
        #returns the list of all ngrams in toks that do not contain <sep>
        ngrams = []
        for u in range(len(toks)):
            for n in range(1,self.max_ngram+1): #if max_ngram=3, n=[1,2,3]
                if u+n > len(toks):
                    break
                if self.str_sep in toks[u:u+n]:
                    break
                ngram = ' '.join(toks[u:u+n])
                ngrams.append(ngram)
        return ngrams

    def __iter__(self):
        if self.fsrc is not None:
            fs, is_gzip_src = open_read(self.fsrc)
        if self.ftgt is not None:
            ft, is_gzip_tgt = open_read(self.ftgt)

        while True:

            SRC = []
            if self.fsrc is not None:
                ls = fs.readline()
                if not ls:
                    break
                if is_gzip_src:
                    ls = ls.decode('utf8')
                ls = ls.strip(" \n")
                SRC = self.token.tokenize(ls)
                if self.do_filter and (len(SRC) == 0 or SRC[0] == ''):
                    logging.debug('skip src empty sentence: <{}> {}:{}'.format(ls,self.fsrc,self.stats_nsent+1))
                    self.stats_nskip += 1
                    continue

            TGT = []
            if self.ftgt is not None:
                lt = ft.readline()
                if not lt:
                    break
                if is_gzip_tgt:
                    lt = lt.decode('utf8')
                lt = lt.strip(" \n")
                TGT = self.token.tokenize(lt)
                if self.do_filter and (len(TGT) == 0 or TGT[0] == ''):
                    logging.debug('skip tgt empty sentence: <{}> {}:{}'.format(lt,self.ftgt,self.stats_nsent+1))
                    self.stats_nskip += 1
                    tgt_ok = False
                    continue

            self.stats_nsent += 1
            self.stats_ntokens += len(SRC) + len(TGT)

            ###
            ### build sentence pair
            ###
            toks = []

            ### SRC
            toks.append(self.str_bos)
            for tok in SRC:
                toks.append(self.tag_src+tok)
            toks.append(self.str_eos)

            ### SEP
            toks.append(self.str_sep)

            ### TGT
            toks.append(self.str_bos)
            for tok in TGT:
                toks.append(self.tag_tgt+tok)
            toks.append(self.str_eos)

            yield toks

        logging.info('filtered {} out of {} sentences (found {} tokens) in {},{}'.format(self.stats_nskip,self.stats_nsent,self.stats_ntokens,self.fsrc,self.ftgt))
        if self.fsrc is not None:
            fs.close()
        if self.ftgt is not None:
            ft.close()









