from config import Config
from pprint import pprint, pformat
from logger import model_logger
log = model_logger.getLogger('main')
import logging
log.setLevel(logging.INFO)

import pickle

import random
from functools import partial
from nltk.tokenize import WordPunctTokenizer
word_punct_tokenizer = WordPunctTokenizer()
word_tokenize = word_punct_tokenizer.tokenize

from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch

from trainer import Trainer, Feeder, Predictor
from datafeed import DataFeed
from utilz import tqdm, ListTable

from collections import namedtuple, defaultdict
Sample = namedtuple('Sample', ['id','comment_text',
                               'toxic','severe_toxic','obscene',
                               'threat','insult','identity_hate'])

ClassifiedDatapoints = namedtuple('ClassifiedDatapoints', ['toxic','severe_toxic','obscene',
                                                           'threat','insult','identity_hate',
                                                           'neutral'])

if Config().flush:
    import csv
    train_dataset = csv.reader(open('dataset/train.csv'))
    test_dataset = csv.reader(open('dataset/test.csv'))

    # ### Unicode to ascii text
    import unicodedata
    train_datapoints = []
    for i in list(train_dataset)[1:]:
        _id, c, t, st, o, t, ins, ih = i
        t, st, o, t, ins, ih = (int(_) for _ in [t, st, o, t, ins, ih])
        c = unicodedata.normalize('NFD', c).encode('ascii','ignore').decode()
        train_datapoints.append(Sample(_id, c.lower(), t, st, o, t, ins, ih))

    test_datapoints = []
    for i in list(test_dataset)[1:]:
        _id, c = i
        c = unicodedata.normalize('NFD', c).encode('ascii','ignore').decode()
        test_datapoints.append(Sample(_id, c.lower(), 0, 0, 0, 0, 0, 0))


    config = Config()
    seq_len_criteria = lambda p: len(word_tokenize(p.comment_text)) < config.seq_len_limit and len(word_tokenize(p.comment_text)) > 0
    train_datapoints = [p for p in tqdm(train_datapoints) if seq_len_criteria(p)]
    print('train: {}, test: {}'.format(len(train_datapoints), len(test_datapoints)))

    classified_datapoints = defaultdict(list)
    for datapoint in train_datapoints:
        classified_datapoints[tuple(datapoint[2:])].append(datapoint)
            
    sort_key = lambda p: len(word_tokenize(p.comment_text))

    sorted_classified_datapoints = {}
    for i in classified_datapoints.keys():
        split_index = int( len(classified_datapoints[i]) * Config().split_ratio )
        sorted_classified_datapoints[i] = (sorted(classified_datapoints [i] [:split_index], key=sort_key, reverse=True),
                                           sorted(classified_datapoints [i] [split_index:], key=sort_key, reverse=True))

    classified_datapoints = sorted_classified_datapoints
    test_datapoints = sorted(test_datapoints, key=lambda p: -len(word_tokenize(p.comment_text)))
    
    # ## Build vocabulary
    # #### buils INPUT_VOCAB
    datapoints = train_datapoints + test_datapoints
    WORD_FREQ = defaultdict(int)
    CHAR_FREQ = defaultdict(int)

    CHAR_VOCAB = [' '] + list(set([c for dp in tqdm(datapoints) for c in dp.comment_text]))
    CHAR_INDEX = {c: i for i, w in enumerate(CHAR_VOCAB)}

    OUTPUT_VOCAB = ['toxic','severe_toxic','obscene', 'threat','insult','identity_hate']
    INPUT_VOCAB  = [word for dp in tqdm(datapoints) for word in word_tokenize(dp.comment_text)]
    INPUT_VOCAB  = INPUT_VOCAB + OUTPUT_VOCAB
    for word in INPUT_VOCAB:
        WORD_FREQ[word] += 1

    WORD_FREQ_PAIRS = sorted(WORD_FREQ.items(), key=lambda x: -x[1])
    INPUT_VOCAB     = [ x[0] for x in WORD_FREQ_PAIRS ]

    print(WORD_FREQ_PAIRS[:100], WORD_FREQ_PAIRS[-100:])
    print('Vocab size: {}'.format(len(INPUT_VOCAB)))
    
    INPUT_VOCAB = ['<<PAD>>', '<<UNK>>'] + INPUT_VOCAB + OUTPUT_VOCAB
    WORD_INDEX  = defaultdict(lambda : INPUT_VOCAB.index('<<UNK>>'))
    INPUT_VOCAB = INPUT_VOCAB[ :Config().input_vocab_size ]
    
    WORD_INDEX.update( {w: i for i, w in enumerate(INPUT_VOCAB)} )

    OUTPUT_WORD_INDEX = {w: i for i, w in enumerate(OUTPUT_VOCAB)}
    OUTPUT_IDS = [OUTPUT_WORD_INDEX[i] for i in OUTPUT_VOCAB]

    PAD = WORD_INDEX[INPUT_VOCAB[0]]

    print('selvakumar is so stupid that he has no sense of purpose', WORD_INDEX['selvakumar is so stupid that he has no sense of purpose'])

    # caching
    pickle.dump([CHAR_VOCAB, CHAR_INDEX,
                 INPUT_VOCAB, OUTPUT_VOCAB,
                 OUTPUT_IDS, PAD,
                 dict(WORD_INDEX), WORD_FREQ_PAIRS,
                 test_datapoints,  train_datapoints, classified_datapoints], open('cache.pkl', 'wb'))
else:
    [CHAR_VOCAB, CHAR_INDEX,
     INPUT_VOCAB, OUTPUT_VOCAB,
     OUTPUT_IDS, PAD,
     WORD_INDEX_DICT, WORD_FREQ_PAIRS,
     test_datapoints, train_datapoints, classified_datapoints] = pickle.load(open('cache.pkl', 'rb'))
    WORD_INDEX = defaultdict(lambda : INPUT_VOCAB.index('<<UNK>>'))
    WORD_INDEX.update(WORD_INDEX_DICT)
    
#train_datapoints, test_datapoints = train_datapoints[:2000], test_datapoints[:2000]
print(sorted(list(WORD_INDEX.items()), key=lambda x: x[1])[:10], WORD_INDEX['<<PAD>>'], INPUT_VOCAB[0], INPUT_VOCAB[ WORD_INDEX['<<PAD>>'] ])


# ## tests INPUTVOCAB and WORD_INDEX mapping
_i = train_datapoints[random.choice(range(len(train_datapoints)))]
print(_i.comment_text)
print("======")
print(' '.join( [INPUT_VOCAB[i] for i in 
                [WORD_INDEX[j] for j in word_tokenize(_i.comment_text)]])  )

# ### Batching utils
import numpy as np
def seq_maxlen(seqs):
    return max([len(seq) for seq in seqs])


print(PAD)
def pad_seq(seqs, maxlen=0, PAD=PAD):
    if type(seqs[0]) == type([]):
        maxlen = maxlen if maxlen else seq_maxlen(seqs)
        def pad_seq_(seq):
            return seq + [PAD]*(maxlen-len(seq))
        seqs = [ pad_seq_(seq) for seq in seqs ]
    return seqs

def batchop(datapoints, *args, **kwargs):
    indices = [d.id for d in datapoints]
    seq  = pad_seq([ [WORD_INDEX[w] for w in word_tokenize(d.comment_text)[:Config().seq_len_limit]]
                     for d in datapoints])
    target = [(d.toxic, d.severe_toxic, d.obscene, d.threat, d.insult, d.identity_hate)
              for d in datapoints]
    seq, target = np.array(seq), np.array(target)
    return indices, (seq, ), (target,)

def char_emb_batchop(datapoints, *args, **kwargs):
    indices = [d.id for d in datapoints]
    seq  = pad_seq([ [WORD_INDEX[w] for w in word_tokenize(d.comment_text)[:Config().seq_len_limit]]
                     for d in datapoints])
    target = [(d.toxic, d.severe_toxic, d.obscene, d.threat, d.insult, d.identity_hate)
              for d in datapoints]
    seq, target = np.array(seq), np.array(target)
    return indices, (seq, ), (target,)


def test_batchop(datapoints, *args, **kwargs):
    indices = [d.id for d in datapoints]
    seq   = pad_seq([ [WORD_INDEX[w] for w in word_tokenize(d.comment_text + '.')[:Config().seq_len_limit]]
                     for d in datapoints])
    seq = np.array(seq)
    return indices, (seq, ), ()

class BiLSTMDecoderModel(nn.Module):
    def __init__(self, Config, input_vocab_size, char_input_vocab_size, output_vocab_size):
        super(BiLSTMDecoderModel, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.char_input_vocab_size = char_input_vocab_size
        
        self.output_vocab_size = output_vocab_size
        self.hidden_dim = Config.hidden_dim
        self.embed_dim = Config.embed_dim
        self.char_embed_dim = Config.char_embed_dim
        
        self.embed = nn.Embedding(self.input_vocab_size, self.embed_dim)

        self.fencode = nn.LSTMCell(self.embed_dim, self.hidden_dim)
        self.bencode = nn.LSTMCell(self.embed_dim, self.hidden_dim)
        
        self.dropout = nn.Dropout(0.2)
        self.project = nn.Linear(2*self.hidden_dim, Config.project_dim)
        self.classify = nn.Linear(Config.project_dim, self.output_vocab_size)

        self.log = model_logger.getLogger('model')
        self.size_log = self.log.getLogger('size')
        self.log.setLevel(logging.DEBUG)
        self.size_log.setLevel(logging.INFO)

        if Config.cuda:
            self.cuda()
            
    def __(self, tensor, name=''):
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            for i in range(len(tensor)):
                self.size_log.debug('{} <- {}[{}]'.format(tensor[i].size(), name, i))

        else:
            self.size_log.debug('{} <- {}'.format(tensor.size(), name))
                
        return tensor
        
    def init_hidden(self, batch_size):
        ret = torch.zeros(batch_size, self.hidden_dim)
        if Config().cuda: ret = ret.cuda()
        return Variable(ret)
    
    def forward(self, seq, classes=OUTPUT_IDS):
        seq = Variable(torch.LongTensor(seq))
        classes = Variable(torch.LongTensor(classes))
        dropout = self.dropout
        if self.training:
            dropout = lambda i: i
        
        if Config().cuda: 
            seq = seq.cuda()
            classes = classes.cuda()
            
        batch_size, seq_size = seq.size()
        seq_emb = self.__(   dropout( F.tanh(self.embed(seq)).transpose(1,0) ), 'seq_emb'   )

        foutputs, boutputs = [], []
        foutput = self.init_hidden(batch_size), self.init_hidden(batch_size)
        boutput = self.init_hidden(batch_size), self.init_hidden(batch_size)
        for i in range(seq_size):
            foutput = self.__(  self.fencode(seq_emb[ i], foutput), 'foutput'   )
            boutput = self.__(  self.bencode(seq_emb[-i], boutput), 'boutput'   )
            foutput = dropout(foutput[0]), dropout(foutput[1])
            boutput = dropout(boutput[0]), dropout(boutput[1])
            foutputs.append(foutput[0])
            boutputs.append(boutput[0])

        boutputs = list(reversed(boutputs))
        foutputs, boutputs = torch.stack(foutputs), torch.stack(boutputs)
        seq_repr = self.__(  torch.cat([foutputs, boutputs], dim=-1), 'seq_repr'  )

        output = seq_repr[-1]
        outputs = []

        logits = self.__(  self.classify(self.project(output)), 'logits'  )
        output = F.sigmoid(logits)
            
        return output

    
# ## Loss and accuracy function
def loss(output, target, loss_function=nn.MSELoss(), scale=1, *args, **kwargs):
    target = Variable(torch.LongTensor(target[0]), requires_grad=False)
    if Config().cuda: target = target.cuda()
    log.debug('i, o sizes: {} {}'.format(output.size(), target.size()))

    return loss_function(output, target.float())

# ### repr_function to build human readable output from model
from IPython.display import HTML
from IPython.display import display
def repr_function(output, feed, batch_index):
    results = []
    indices, (seq,), (classes,) = feed.nth_batch(batch_index)
    for i, o, s, c in zip(indices, output, seq, classes):
        orig_s = ' '.join(feed.data_dict[i].comment_text.split())
        s      = ' '.join([INPUT_VOCAB[i] for i in s])
        results.append([ str(z) for z in list(c)] + [orig_s, s])
        results.append([  '{:0.5f}'.format(z) for z in o.data.cpu().numpy().tolist()])
    del indices, seq, classes
    return results

def test_repr_function(output, feed, batch_index):
    results = []
    indices, (seq,), _ = feed.nth_batch(batch_index)
    for i, o, s in zip(indices, output, seq):
        o = [ '{:0.5f}'.format(z) for z in o.data.cpu().numpy().tolist()]
        s = ' '.join([INPUT_VOCAB[si] for si in s])
        results.append( [i] + o  + [s] )
    del indices, seq
    return results

def  experiment(eons=1000, epochs=1, checkpoint=1):
    try:
        try:
            model =  BiLSTMDecoderModel(Config(), len(INPUT_VOCAB), len(CHAR_VOCAB), len(OUTPUT_VOCAB))
            if Config().cuda:  model = model.cuda()
            model.load_state_dict(torch.load('sigmoid_model.pth'))
            log.info('loaded the old image for the model')
        except:
            log.exception('failed to load the model')
            model =  BiLSTMDecoderModel(Config(), len(INPUT_VOCAB), len(CHAR_VOCAB), len(OUTPUT_VOCAB))
            if Config().cuda:  model = model.cuda()


        train_feed, test_feed, predictor_feed = {}, {}, {}
        trainer, predictor = {}, {}

        dev_test_feed = [p for points in classified_datapoints.values() for p in points[1]]
        max_size = max( sorted(   [len(i[0]) for i in classified_datapoints.values()]   )[:-1] )
        #max_size = max( sorted(   [len(i[0]) for i in classified_datapoints.values()]   ) )
        
        print('dev_test_feed - len',len(dev_test_feed))
        for label in classified_datapoints.keys():
            if len(classified_datapoints[label][0]) < 1: continue
            print('label: {} and size: {}'.format(label, len(classified_datapoints[label][0])))
            train_feed[label]      = DataFeed(classified_datapoints[label][0], batchop=batchop, batch_size=max(32, min(int(len(classified_datapoints[label][0])/600), 1))  )
            #test_feed[label]       = DataFeed(classified_datapoints[label][1], batchop=batchop, batch_size=32)
            test_feed[label]       = DataFeed(dev_test_feed, batchop=batchop, batch_size=16)
            predictor_feed[label]  = DataFeed(classified_datapoints[label][1], batchop=batchop, batch_size=12)
            
            turns = int(max_size/train_feed[label].size) + 1            
            trainer[label] = Trainer(model=model, 
                                     loss_function=partial(loss, scale=turns), accuracy_function=loss, 
                                     checkpoint=checkpoint, epochs=epochs,
                                     feeder = Feeder(train_feed[label], test_feed[label]))

            predictor[label] = Predictor(model=model, feed=predictor_feed[label], repr_function=repr_function)

        test_predictor_feed = DataFeed(test_datapoints, batchop=test_batchop, batch_size=64)
        test_predictor = Predictor(model=model, feed=test_predictor_feed, repr_function=test_repr_function)


        label_trainer_triples = sorted( [(l, t, train_feed[l].size) for l, t in trainer.items()], key=lambda x: x[2] )
        log.info('trainers built {}'.format(pformat(label_trainer_triples)))
                                    
        dump = open('results/experiment_attn.csv', 'w').close()
        for e in range(eons):
            dump = open('results/experiment_attn.csv', 'a')
            dump.write('\n#========================after eon: {}\n'.format(e))
            dump.close()
            log.info('on {}th eon'.format(e))
        
            if not e % 1:
                test_results = ListTable()
                test_dump = open('results/experiment_attn_over_test_{}.csv'.format(e), 'w')
                test_dump.write('|'.join(['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']) + '\n')
                log.info('running over test')

                for i in tqdm(range(test_predictor_feed.num_batch)):
                    log.debug('i: {}'.format(i))
                    output, results = test_predictor.predict(i)
                    test_results += results

                test_dump.write(repr(test_results))            
                test_dump.close()

            for label, _, _ in reversed(label_trainer_triples):
                if not sum(label) and e % 10: continue
                label_desc = '-'.join([OUTPUT_VOCAB[l] for l in [i for i, x in enumerate(label) if x == 1]] )
                log.info('=================================== training for {} datapoints ========================================'.format(label_desc))

                with open('results/experiment_attn.csv', 'a') as dump:
                    output, results = predictor[label].predict(random.choice(range(predictor_feed[label].num_batch)))
                    dump.write(repr(results))
                    del output, results
                
                turns = int(max_size/train_feed[label].size/6) + 1
                log.info('========================  size: {} and turns: {}==========================================='.format(train_feed[label].size, turns))                
                for turn in range(turns):
                    log.info('==================================  label: {} and turn: {}/{}====================================='.format(label_desc, turn, turns))                
                    trainer[label].train()


    except :
        torch.save(model.state_dict(), open('sigmoid_model.pth', 'wb'))

        return locals()

exp_image = experiment()
