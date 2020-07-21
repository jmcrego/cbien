# -*- coding: utf-8 -*-
import torch
import logging
import yaml
import sys
import os
#import io
#import math
import random
#import itertools
#import pyonmttok
import glob
import numpy as np
import torch.nn as nn
#from collections import Counter
from dataset import Dataset
from vocab import Vocab
from tokenizer import OpenNMTTokenizer
from model import Word2Vec, load_model, load_build_optim, save_model, save_optim
from utils import create_logger

def do_preprocess(args):
    ###
    ### build name.{token,vocab}
    ###
    if args.tok_conf is None:
        opts = {}
        opts['mode'] = 'space'
        with open(args.name + '.token', 'w') as yamlfile:
            _ = yaml.dump(opts, yamlfile)
    else:
        with open(args.tok_conf) as yamlfile: 
            opts = yaml.load(yamlfile, Loader=yaml.FullLoader)
            with open(args.name + '.token', 'w') as ofile:
                yaml.dump(opts, ofile)
    logging.info('written tokenizer config file')
    ###
    ### build name.vocab
    ###
    token = OpenNMTTokenizer(args.name + '.token')
    vocab = Vocab()
    vocab.build(args.data_src, args.data_tgt, token, min_freq=args.voc_minf, max_size=args.voc_maxs, max_ngram=args.voc_maxn)
    vocab.dump(args.name + '.vocab')

def do_examples(args):
    ###
    ### build name.examples
    ###
    token = OpenNMTTokenizer(args.name + '.token')
    vocab = Vocab()
    vocab.read(args.name + '.vocab')
    dataset = Dataset(args,vocab,token)
    dataset.examples()

def do_train(args):
    token = OpenNMTTokenizer(args.name + '.token')
    vocab = Vocab()
    vocab.read(args.name + '.vocab')
    dataset = Dataset(args,vocab,token)

    model, n_steps = load_model(args.name, vocab)
    if model is None:
        logging.info('start model from scratch')
        model = Word2Vec(len(vocab), args.embedding_size, args.pooling, vocab.idx_pad)
    if args.cuda:
        model.cuda()

    optimizer = load_build_optim(args.name, model, args.learning_rate, args.beta1, args.beta2, args.eps)
    n_epochs = 0
    losses = []
    while True:
        n_epochs += 1
        for batch_idx, batch_neg, batch_ctx, batch_msk in dataset:
            model.train()
            loss = model.forward(batch_idx, batch_neg, batch_ctx, batch_msk)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_steps += 1
            losses.append(loss.data.cpu().detach().numpy())

            if n_steps % args.report_every_n_steps == 0:
                accum_loss = np.mean(losses)
                logging.info('{} n_epoch={} n_steps={} Loss={:.6f}'.format(args.mode, n_epochs,n_steps,accum_loss))
                losses = []

            if n_steps % args.save_every_n_steps == 0:
                save_model(args.name, model, n_steps, args.keep_last_n)
                save_optim(args.name, optimizer)

            if n_steps % args.valid_every_n_steps == 0:
                do_validation(args,vocab,token,n_epochs,n_steps,model)

        if n_epochs >= args.max_epochs:
            logging.info('Stop ({} epochs reached)'.format(n_epochs))
            break
    save_model(args.name, model, n_steps, args.keep_last_n)
    save_optim(args.name, optimizer)

def do_validation(args,vocab,token,n_epochs,n_steps,model):
    losses = []
    logging.info('run VALIDATION')
    dataset = Dataset(args, vocab, token, isValid=True)
    with torch.no_grad():
        model.eval()
        for batch_idx, batch_neg, batch_ctx, batch_msk in dataset:
            loss = model.forward(batch_idx, batch_neg, batch_ctx, batch_msk)
            losses.append(loss.data.cpu().detach().numpy())
    if len(losses):
        accum_loss = np.mean(losses)
        logging.info('VALIDATION n_epoch={} n_steps={} Loss={:.6f}'.format(n_epochs,n_steps,accum_loss))
    else:
        logging.info('no examples found for VALIDATION')

def do_sentence_vectors(args):
    if not os.path.exists(args.name + '.token'):
        logging.error('missing {} file'.format(args.name + '.token'))
        sys.exit()
    if not os.path.exists(args.name + '.vocab'):
        logging.error('missing {} file'.format(args.name + '.vocab'))
        sys.exit()
    if len(glob.glob(args.name + '.model.?????????.pth')) == 0:
        logging.error('no model available: {}'.format(args.name + '.model.?????????.pth'))
        sys.exit()

    token = OpenNMTTokenizer(args.name + '.token')
    vocab = Vocab()
    vocab.read(args.name + '.vocab')
    model, _ = load_model(args.name, vocab)
    if args.cuda:
        model.cuda()

    dataset = Dataset(args, token, vocab)
    dataset.examples_inference()
    with torch.no_grad():
        model.eval()
        for batch in dataset:
            #[batch_snt, batch_msk, batch_ind]
            msk = torch.as_tensor(batch[1]) #[bs,n] (positive words are 1.0 others are 0.0)
            if args.cuda:
                msk = msk.cuda()            
            snts = model.NgramsEmbed(batch[0], msk).cpu().detach().numpy().tolist()
            for i in range(len(snts)):
                sentence = ["{:.6f}".format(w) for w in snts[i]]
                print('{}\t{}'.format(batch[2][i]+1, ' '.join(sentence) ))


################################################################
### args #######################################################
################################################################
class Args():

    def __init__(self, argv):
        self.name = None
        self.data_src = None
        self.data_tgt = None
        self.etag = None
        self.mode = None
        self.seed = 12345
        self.cuda = False
        self.log_file = None
        self.log_level = 'debug'
        self.voc_minf = 5
        self.voc_maxs = 0
        self.voc_maxn = 1
        self.tok_conf = None
        self.train = None
        self.pkeep = 1.0
        self.pooling = 'avg'
        self.batch_size = 2048
        self.max_epochs = 1
        self.embedding_size = 300
        self.window = 0
        self.n_negs = 10
        self.learning_rate = 0.001
        self.eps = 1e-08
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.keep_last_n = 5
        self.save_every_n_steps = 5000
        self.valid_every_n_steps = 10000
        self.report_every_n_steps = 500
        self.k = 5
        self.sim = 'cos'
        self.prog = argv.pop(0)
        self.usage = '''usage: {} -name STRING -mode STRING -data_src FILES -data_tgt FILES [Options]
   -name         STRING : experiment name
   -mode         STRING : preprocess, examples, train, sentence-vectors, word-vectors

 Options:
   -seed            INT : seed value                                (12345)
   -log_file       FILE : log file (use stderr for STDERR)          ([name].log)
   -log_level     LEVEL : debug, info, warning, critical, error     (debug) 
   -cuda                : use CUDA                                  (False)
   -h                   : this help

 -------- When building vocab (mode preprocess) ------------------------------
   -data_src      FILES : source file
   -data_tgt      FILES : target file
   -voc_minf        INT : min frequency to consider a word          (5)
   -voc_maxs        INT : max size of vocabulary (0 for unlimitted) (0)
   -voc_maxn        INT : consider up to this word ngrams           (1)
   -tok_conf       FILE : YAML file with onmt tokenization options  (space)

 -------- When building examples (mode examples) -----------------------------
   -data_src      FILES : source file
   -data_tgt      FILES : target file
   -window          INT : window size (use 0 for whole sentence)    (0)
   -pkeep         FLOAT : probability to keep an example            (1.0)
   -etag         STRING : output examples tag

 -------- When learning (mode train) -----------------------------------------
   -batch_size      INT : batch size used                           (2048)
   -n_negs          INT : number of negative samples                (10)
   -pooling      STRING : max, avg, sum                             (avg)
   -embedding_size  INT : embedding dimension                       (300)
   -max_epochs      INT : stop learning after this number of epochs (1)

   -learning_rate FLOAT : learning rate for Adam optimizer          (0.001)
   -eps           FLOAT : eps for Adam optimizer                    (1e-08)
   -beta1         FLOAT : beta1 for Adam optimizer                  (0.9)
   -beta2         FLOAT : beta2 for Adam optimizer                  (0.999)

   -keep_last       INT : keep last n checkpoints                   (5)
   -save_every      INT : save checkpoint every n learning steps    (5000)
   -valid_every     INT : run validation every n learning steps     (5000)
   -report_every    INT : print report every n learning steps       (500)

 -------- When inference (mode sentence-vectors or word-vectors) -------------
   -k               INT : find k closest words to each file ngram   (5)
   -sim          STRING : cos, pairwise                             (cos)

Shuffle and split examples into shards: 
  gunzip -c [name].examples.*.gz | shuf | split -a 5 -l 2000000 - [name].shard_ --filter='gzip -c > $FILE.gz'
To allow validation:
  after building shards replace one/some [name].shard_????? by [name].valid_?????

*** The script needs:
  + pytorch:   conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
  + pyyaml:    pip install PyYAML
  + pyonmttok: pip install pyonmttok
'''.format(self.prog)

        if len(argv) == 0:
            sys.stderr.write("{}".format(self.usage))
            sys.exit()

        while len(argv):
            tok = argv.pop(0)
            if   (tok=="-name" and len(argv)): self.name = argv.pop(0)
            elif (tok=="-mode" and len(argv)): self.mode = argv.pop(0)
            elif (tok=="-data_src" and len(argv)): self.data_src = argv.pop(0)
            elif (tok=="-data_tgt" and len(argv)): self.data_tgt = argv.pop(0)
            #
            elif (tok=="-voc_minf" and len(argv)): self.voc_minf = int(argv.pop(0))
            elif (tok=="-voc_maxs" and len(argv)): self.voc_maxs = int(argv.pop(0))
            elif (tok=="-voc_maxn" and len(argv)): self.voc_maxn = int(argv.pop(0))
            elif (tok=="-tok_conf" and len(argv)): self.tok_conf = argv.pop(0)
            #
            elif (tok=="-window" and len(argv)): self.window = int(argv.pop(0))
            elif (tok=="-pkeep" and len(argv)): self.pkeep_example = float(argv.pop(0))
            elif (tok=="-etag" and len(argv)): self.etag = argv.pop(0)            
            #
            elif (tok=="-batch_size" and len(argv)): self.batch_size = int(argv.pop(0))
            elif (tok=="-n_negs" and len(argv)): self.n_negs = int(argv.pop(0))
            elif (tok=="-pooling" and len(argv)): self.pooling = argv.pop(0)
            elif (tok=="-embedding_size" and len(argv)): self.embedding_size = int(argv.pop(0))
            elif (tok=="-max_epochs" and len(argv)): self.max_epochs = int(argv.pop(0))
            elif (tok=="-learning_rate" and len(argv)): self.learning_rate = float(argv.pop(0))
            elif (tok=="-eps" and len(argv)): self.eps = float(argv.pop(0))
            elif (tok=="-beta1" and len(argv)): self.beta1 = float(argv.pop(0))
            elif (tok=="-beta2" and len(argv)): self.beta2 = float(argv.pop(0))
            #
            elif (tok=="-keep_last" and len(argv)): self.keep_last_n = int(argv.pop(0))
            elif (tok=="-save_every" and len(argv)): self.save_every_n_steps = int(argv.pop(0))
            elif (tok=="-valid_every" and len(argv)): self.valid_every_n_steps = int(argv.pop(0))
            elif (tok=="-report_every" and len(argv)): self.report_every_n_steps = int(argv.pop(0))
            #
            elif (tok=="-cuda"): self.cuda = True
            elif (tok=="-seed" and len(argv)): self.seed = int(argv.pop(0))
            elif (tok=="-log_file" and len(argv)): self.log_file = argv.pop(0)
            elif (tok=="-log_level" and len(argv)): self.log_level = argv.pop(0)
            #
            elif (tok=="-k" and len(argv)): self.k = int(argv.pop(0))
            elif (tok=="-sim" and len(argv)): self.sim = argv.pop(0)
            elif (tok=="-h"):
                sys.stderr.write("{}".format(self.usage))
                sys.exit()
            else:
                sys.stderr.write('error: unparsed {} option\n'.format(tok))
                sys.stderr.write("{}".format(self.usage))
                sys.exit()

        if self.log_file is None:
            self.log_file = self.name + '.log'

        create_logger(self.log_file, self.log_level)

        if self.name is None:
            logging.error('missing -name option')
            sys.exit()

        if self.mode is None:
            logging.error('missing -mode option')
            sys.exit()

        if (self.mode == 'preprocess'  or self.mode == 'examples') and (self.data_src is None and self.data_tgt is None):
            logging.error('missing -data_src OR -data_tgt option')
            sys.exit()

        if self.mode == 'examples' and self.etag is None:
            logging.error('missing -etag option')
            sys.exit()

        if self.seed > 0:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            logging.debug('random seed set to {}'.format(self.seed))

#        if ',' in self.data_src:
#            self.data_src = self.data_src.split(',')
#        else:
#            self.data_src = sorted(glob.glob(self.data_src))
#        print(self.data_src)

#        if ',' in self.data_tgt:
#            self.data_tgt = self.data_tgt.split(',')
#        else:
#            self.data_tgt = sorted(glob.glob(self.data_tgt))
#        print(self.data_tgt)

####################################################################
### Main ###########################################################
####################################################################
if __name__ == "__main__":

    args = Args(sys.argv) #creates logger and sets random seed

    if args.mode == 'preprocess':
        do_preprocess(args)

    elif args.mode == 'examples':
        do_examples(args)

    elif args.mode == 'train':
        do_train(args)

    elif args.mode == 'sentence-vectors':
        do_sentence_vectors(args)

    else:
        logging.error('bad -mode option {}'.format(args.mode))
        sys.exit()

    logging.info('Done!')


