# -*- coding: utf-8 -*-
import logging
import sys
import os
import io
from collections import defaultdict
from utils import open_read
from inputter import Inputter


class Vocab():

    def __init__(self):
        self.idx_pad = 0 
        self.str_pad = '<pad>'
        self.idx_unk = 1 
        self.str_unk = '<unk>'
        self.idx_bos = 2
        self.str_bos = '<bos>'
        self.idx_eos = 3
        self.str_eos = '<eos>'
        self.idx_sep = 4
        self.str_sep = '<sep>'
        self.tok_to_idx = {} 
        self.idx_to_tok = [] 
        self.max_ngram = None
        self.tag_src = '⠁' #braille number '1'
        self.tag_tgt = '⠃' #braille number '2'

    def read(self, file):
        if not os.path.exists(file):
            logging.error('missing {} file'.format(file))
            sys.exit()

        f, is_gzip = open_read(file)
        for l in f:
            if is_gzip:
                l = l.decode('utf8')
            tok = l.strip(" \n")
            if self.max_ngram is None:
                info = tok.split()
                if len(info)!=3:
                    logging.error('erroneous first line in vocab (it must contain: max_ngram tag_src tag_tgt)')
                    sys.exit()
                self.max_ngram = int(info.pop(0))
                self.tag_src = info.pop(0)
                self.tag_tgt = info.pop(0)
                logging.info('vocab max_ngram: {}'.format(self.max_ngram))
                logging.info('vocab tag_src: {}'.format(self.tag_src))
                logging.info('vocab tag_tgt: {}'.format(self.tag_tgt))
                continue

            if tok not in self.tok_to_idx:
                self.idx_to_tok.append(tok)
                self.tok_to_idx[tok] = len(self.tok_to_idx)

        f.close()
        logging.info('read vocab ({} entries) from {}'.format(len(self.idx_to_tok), file))

    def dump(self, file):
        f = open(file, "w")
        f.write('{} {} {}\n'.format(self.max_ngram,self.tag_src,self.tag_tgt))
        for tok in self.idx_to_tok:
            f.write(tok+'\n')
        f.close()
        logging.info('written vocab ({} entries, max_ngram={} tags_src: {} tag_tgt: {}) into {}'.format(len(self.idx_to_tok), self.max_ngram, self.tag_src, self.tag_tgt, file))

    def build(self,file_src,file_tgt,token,min_freq=5,max_size=0,max_ngram=1):
        self.max_ngram = max_ngram
        self.tok_to_frq = defaultdict(int)

        file_pair = Inputter(file_src,file_tgt,token,self)
        for src_tgt_sentence_tok, src_tgt_sentence_idx, to_predict in file_pair:
            for ngram in file_pair.ngrams(src_tgt_sentence_tok):
                self.tok_to_frq[ngram] += 1

        ### build vocab
        self.tok_to_idx[self.str_pad] = self.idx_pad #0
        self.idx_to_tok.append(self.str_pad)        
        self.tok_to_idx[self.str_unk] = self.idx_unk #1
        self.idx_to_tok.append(self.str_unk)
        self.tok_to_idx[self.str_bos] = self.idx_bos #2
        self.idx_to_tok.append(self.str_bos)
        self.tok_to_idx[self.str_eos] = self.idx_eos #3
        self.idx_to_tok.append(self.str_eos)
        self.tok_to_idx[self.str_sep] = self.idx_sep #4
        self.idx_to_tok.append(self.str_sep)
        stats_src = defaultdict(int)
        stats_tgt = defaultdict(int)
        stats_ngrams = len(self.idx_to_tok)
        for ngram, frq in sorted(self.tok_to_frq.items(), key=lambda item: item[1], reverse=True):
            if len(self.idx_to_tok) == max_size:
                break
            if frq < min_freq:
                break
            self.tok_to_idx[ngram] = len(self.idx_to_tok)
            self.idx_to_tok.append(ngram)
            ### stats
            stats_ngrams += 1
            n = len(ngram.split(' '))
            if ngram.find(self.tag_src):
                stats_src[n] += 1
            else:
                stats_tgt[n] += 1

        logging.info('built vocab ({} entries)'.format(len(self.idx_to_tok)))
        for n,N in sorted(stats_src.items(), key=lambda item: item[0], reverse=False): 
            logging.info('src {}-grams: {}'.format(n,N))
        for n,N in sorted(stats_tgt.items(), key=lambda item: item[0], reverse=False): 
            logging.info('tgt {}-grams: {}'.format(n,N))
        logging.info('total n-grams: {}'.format(stats_ngrams))


    def __len__(self):
        return len(self.idx_to_tok)

    def __iter__(self):
        for tok in self.idx_to_tok:
            yield tok

    def __contains__(self, s): ### implementation of the method used when invoking : entry in vocab
        if type(s) == int: ### testing an index
            return s>=0 and s<len(self)
        ### testing a string
        return s in self.tok_to_idx

    def __getitem__(self, s): ### implementation of the method used when invoking : vocab[entry]
        if type(s) == int: ### input is an index, i want the string
            if s not in self:
                logging.error("key \'{}\' not found in vocab".format(s))
                sys.exit()
            ### s exists in self.idx_to_tok
            return self.idx_to_tok[s]
        ### input is a string, i want the index
        if s not in self: 
            return self.idx_unk
        return self.tok_to_idx[s]

