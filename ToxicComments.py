from config import Config
from pprint import pprint, pformat
from logger import model_logger
log = model_logger.getLogger('main')
import logging
log.setLevel(logging.INFO)

import pickle

import random
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


    print('train: {}, test: {}'.format(len(train_datapoints), len(test_datapoints)))
    if Config().find_lengths:
        lengths = [len(word_tokenize(p.comment_text)) for p in tqdm(train_datapoints)]
        print('min/max lengths train_datapoints: {}/{}'.format(min(lengths), max(lengths)))
        lengths = [len(word_tokenize(p.comment_text)) for p in tqdm(test_datapoints)]
        print('min/max lengths test_datapoints: {}/{}'.format(min(lengths), max(lengths)))

    config = Config()
    seq_len_criteria = lambda p: len(word_tokenize(p.comment_text)) < config.seq_len_limit and len(word_tokenize(p.comment_text)) > 0
    train_datapoints = [p for p in tqdm(train_datapoints) if seq_len_criteria(p)]
    #test_datapoints = [p for p in tqdm(test_datapoints) if seq_len_criteria(p)]
    print('train: {}, test: {}'.format(len(train_datapoints), len(test_datapoints)))
    
    cond = lambda x: sum(x[2:]) >= 1
    classified_train_datapoints = [p for p in train_datapoints if cond(p)]

    sort_key = lambda p: len(word_tokenize(p.comment_text))
    split_index = int( len(classified_train_datapoints) * Config().split_ratio )
    classified_datapoints = sorted(classified_train_datapoints[:split_index], key=sort_key), sorted(classified_train_datapoints[split_index:], key=sort_key)

    split_index = int( len(train_datapoints) * Config().split_ratio )
    non_classified_datapoints = sorted(train_datapoints[:split_index], key=sort_key), sorted(train_datapoints[split_index:], key=sort_key)
    
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
                 test_datapoints,  train_datapoints, classified_datapoints, non_classified_datapoints], open('cache.pkl', 'wb'))
else:
    [CHAR_VOCAB, CHAR_INDEX,
     INPUT_VOCAB, OUTPUT_VOCAB,
     OUTPUT_IDS, PAD,
     WORD_INDEX_DICT, WORD_FREQ_PAIRS,
     test_datapoints, train_datapoints, classified_datapoints, non_classified_datapoints] = pickle.load(open('cache.pkl', 'rb'))
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


# ## Baseline model
class Model(nn.Module):
    def __init__(self, Config, input_vocab_size, output_vocab_size):
        super(Model, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.hidden_dim = Config.hidden_dim

        self.embed = nn.Embedding(self.input_vocab_size, self.hidden_dim)
        self.encode = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.classify = [nn.Linear(self.hidden_dim, 2)
                         for i in range (self.output_vocab_size)]

        self.log = model_logger.getLogger('model')
        self.log.setLevel(logging.INFO)
        if Config.cuda:
            self.cuda()
            [i.cuda() for i in self.classify]
        
    def init_hidden(self, batch_size):
        ret = torch.zeros(batch_size, self.hidden_dim)
        if Config().cuda: ret = ret.cuda()
        return Variable(ret)
    
    def forward(self, seq):
        seq = Variable(torch.LongTensor(seq))
        if Config().cuda: seq = seq.cuda()
        batch_size = seq.size()[0]
        self.log.debug('{} seq size: {}'.format(type(seq.data), seq.size()))
        seq_emb = self.embed(seq).transpose(1,0)
        output = self.init_hidden(batch_size)
        for token_emb in seq_emb:
            self.log.debug('token_emb := {}'.format(token_emb))
            self.log.debug('output := {}'.format(output))
            output = self.encode(token_emb, output)
                    
        self.log.debug('output := {}'.format(output))    
        ret = torch.stack([F.log_softmax(classify(output), dim=-1) for classify in self.classify])
        self.log.debug('ret := {}'.format(ret))
        self.log.debug('ret size: {}'.format(ret.size()))
        return ret

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


# ## Attention based model
import math
class AttModel(nn.Module):
    def __init__(self, Config, input_vocab_size, output_vocab_size):
        super(AttModel, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.hidden_dim = Config.hidden_dim
        self.embed_dim = Config.embed_dim
        
        self.embed = nn.Embedding(self.input_vocab_size, self.embed_dim)
        self.embed_class = nn.Embedding(self.output_vocab_size, self.hidden_dim)

        self.encode = nn.GRUCell(self.embed_dim, self.hidden_dim)
        
        self.dropout = nn.Dropout(0.2)
        self.attend = nn.Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim))

        self.classify = nn.Linear(self.hidden_dim, 2)
        self.log = model_logger.getLogger('model')
        self.size_log = self.log.getLogger('size')
        self.log.setLevel(logging.INFO)
        self.size_log.setLevel(logging.INFO)
        
        self.attend.data.normal_(0, 0.1)
        if Config.cuda:
            self.cuda()
            
    def logsize(self, tensor, name=''):
        self.size_log.debug('{} <- {}'.format(tensor.size(), name))
        
    def init_hidden(self, batch_size):
        ret = torch.zeros(batch_size, self.hidden_dim)
        if Config().cuda: ret = ret.cuda()
        return Variable(ret)
    
    def forward(self, seq, classes=OUTPUT_IDS):
        seq = Variable(torch.LongTensor(seq))
        classes = Variable(torch.LongTensor(classes))
           
        if Config().cuda: 
            seq = seq.cuda()
            classes = classes.cuda()
            
        pad_mask = (seq > 0).float()
        self.log.debug('seq {}'.format(seq))
        self.log.debug('classes {}'.format(classes))
        self.log.debug('pad_mask {}'.format(pad_mask))
         
        batch_size = seq.size()[0]
        self.log.debug('{} seq size: {}'.format(type(seq.data), seq.size()))
        
        seq_emb = F.tanh(self.embed(seq)).transpose(1,0)              ;self.logsize(seq_emb, 'seq_emb')
        seq_emb = self.dropout(seq_emb)
        self.log.debug('seq_emb {}'.format(seq_emb))

        seq_repr = []
        output = self.init_hidden(batch_size)                     ;self.logsize(output, 'output')
        for token_emb in seq_emb:
            self.log.debug('token_emb := {}'.format(token_emb))
            self.log.debug('output := {}'.format(output))
            output = self.encode(token_emb, output)               ;self.logsize(output, 'output')
            output = self.dropout(output)
            seq_repr.append(output)

        seq_repr = torch.stack(seq_repr).transpose(1,0)           ;self.logsize(seq_repr, 'seq_repr')
        
        outputs = []
        attend = self.attend
        self.logsize(attend, 'attend')
        for class_ in classes:
            class_emb = self.embed_class(class_)                  ;self.logsize(class_emb, 'class_emb')
            self.log.debug('class_emb: {}'.format(class_emb))
            self.log.debug('attend: {}'.format(attend))

            attn = torch.mm(F.tanh(class_emb), attend)        ;self.logsize(attn, 'attn')
            self.log.debug('attn: {}'.format(attn))

            attn = attn.expand(seq_repr.size()[0], *attn.size()) ;self.logsize(attn, 'attn')
            attended_outputs = torch.bmm(F.tanh(attn), seq_repr.transpose(1,2))                
        
            self.logsize(attended_outputs, 'attended_outputs')
            self.log.debug('{}'.format(attended_outputs))

            attended_outputs = attended_outputs.squeeze(1) * pad_mask        ;self.logsize(attended_outputs, 'attended_outputs')
            self.log.debug('{}'.format(attended_outputs))

            output = F.tanh(attended_outputs).unsqueeze(-1) * seq_repr      ;self.logsize(output, 'output')
            self.log.debug('output {}'.format(output))

            output = output.sum(1).squeeze(1)                ;self.logsize(output, 'output')
            output = self.dropout(output)
            self.log.debug('output {}'.format(output))
            
            output = self.classify(F.tanh(output))                  ;self.logsize(output, 'output')
            self.log.debug('output {}'.format(output))

            output = F.log_softmax(output, dim=-1)
            self.log.debug('output {}'.format(output))

            outputs.append(output)
            
        ret = torch.stack(outputs)
        self.log.debug('ret {}'.format(ret))

        return ret

class DecoderModel(nn.Module):
    def __init__(self, Config, input_vocab_size, char_input_vocab_size, output_vocab_size):
        super(DecoderModel, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.char_input_vocab_size = char_input_vocab_size
        
        self.output_vocab_size = output_vocab_size
        self.hidden_dim = Config.hidden_dim
        self.embed_dim = Config.embed_dim
        
        self.embed = nn.Embedding(self.input_vocab_size, self.embed_dim)
        self.embed_char = nn.Embedding(self.char_input_vocab_size, self.char_embed_dim)
        self.embed_class = nn.Embedding(self.output_vocab_size, self.hidden_dim)

        self.encode = nn.GRUCell(self.embed_dim, self.hidden_dim)
        self.decode = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        
        self.dropout = nn.Dropout(0.2)

        self.classify = nn.Linear(self.hidden_dim, 2)

        self.log = model_logger.getLogger('model')
        self.size_log = self.log.getLogger('size')
        self.log.setLevel(logging.INFO)
        self.size_log.setLevel(logging.INFO)

        if Config.cuda:
            self.cuda()
            
    def logsize(self, tensor, name=''):
        self.size_log.debug('{} <- {}'.format(tensor.size(), name))
        
    def init_hidden(self, batch_size):
        ret = torch.zeros(batch_size, self.hidden_dim)
        if Config().cuda: ret = ret.cuda()
        return Variable(ret)
    
    def forward(self, seq, classes=OUTPUT_IDS):
        seq = Variable(torch.LongTensor(seq))
        classes = Variable(torch.LongTensor(classes))
           
        if Config().cuda: 
            seq = seq.cuda()
            classes = classes.cuda()
            
        pad_mask = (seq > 0).float()
        self.log.debug('seq {}'.format(seq))
        self.log.debug('classes {}'.format(classes))
        self.log.debug('pad_mask {}'.format(pad_mask))
         
        batch_size = seq.size()[0]
        self.log.debug('{} seq size: {}'.format(type(seq.data), seq.size()))
        
        seq_emb = F.tanh(self.embed(seq)).transpose(1,0)              ;self.logsize(seq_emb, 'seq_emb')
        seq_emb = self.dropout(seq_emb)
        self.log.debug('seq_emb {}'.format(seq_emb))

        seq_repr = []
        output = self.init_hidden(batch_size)                     ;self.logsize(output, 'output')
        for token_emb in seq_emb:
            self.logsize(token_emb, 'token_emb')
            self.log.debug('token_emb := {}'.format(token_emb))
            self.log.debug('output := {}'.format(output))
            output = self.encode(token_emb, output)               ;self.logsize(output, 'output')
            output = self.dropout(output)
        
        outputs = []
        for class_ in classes:
            class_emb = self.embed_class(class_).expand_as(output)       ;self.logsize(class_emb, 'class_emb')
            class_emb = F.tanh(class_emb)                           ;self.logsize(class_emb, 'class_emb')
            self.log.debug('class_emb: {}'.format(class_emb))

            output = self.decode(class_emb, output)
            output = self.dropout(output)
            output = F.tanh(output)                                  ;self.logsize(output, 'output')
            _output = self.classify(output)                          ;self.logsize(_output, '_output')
            self.log.debug('output {}'.format(_output))

            outputs.append(F.log_softmax(_output, dim=-1))
            
        ret = torch.stack(outputs)                  ;self.logsize(ret, 'ret')
        self.log.debug('ret {}'.format(ret))
        
        return ret

    
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
        self.embed_char = nn.Embedding(self.char_input_vocab_size, self.char_embed_dim)
        self.embed_class = nn.Embedding(self.output_vocab_size, self.hidden_dim)

        self.fencode = nn.LSTMCell(self.embed_dim, self.hidden_dim)
        self.bencode = nn.LSTMCell(self.embed_dim, self.hidden_dim)
        
        self.decode = nn.GRUCell(self.hidden_dim, 2*self.hidden_dim)
        
        self.dropout = nn.Dropout(0.2)
        self.classify = nn.Linear(2*self.hidden_dim, 2)

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
           
        if Config().cuda: 
            seq = seq.cuda()
            classes = classes.cuda()
            
        batch_size, seq_size = seq.size()
        seq_emb = self.__(   self.dropout( F.tanh(self.embed(seq)).transpose(1,0) ), 'seq_emb'   )

        foutputs, boutputs = [], []
        foutput = self.init_hidden(batch_size), self.init_hidden(batch_size)
        boutput = self.init_hidden(batch_size), self.init_hidden(batch_size)
        for i in range(seq_size):
            foutput = self.__(  self.fencode(seq_emb[ i], foutput), 'foutput'   )
            boutput = self.__(  self.bencode(seq_emb[-i], boutput), 'boutput'   )
            foutput = self.dropout(foutput[0]), self.dropout(foutput[1])
            boutput = self.dropout(boutput[0]), self.dropout(boutput[1])
            foutputs.append(foutput[0])
            boutputs.append(boutput[0])

        boutputs = list(reversed(boutputs))
        foutputs, boutputs = torch.stack(foutputs), torch.stack(boutputs)
        seq_repr = self.__(  torch.cat([foutputs, boutputs], dim=-1), 'seq_repr'  )

        output = seq_repr[-1]
        outputs = []
        for class_ in classes:
            class_emb = self.__( self.embed_class(class_), 'class_emb' )
            class_emb = self.__( F.tanh(class_emb).expand(batch_size, *class_emb.size()), 'class_emb' )

            output = self.__(  F.tanh( self.dropout(self.decode(class_emb, output)) ), 'output'  )
            logits = self.__(  self.classify(output), 'logits'  )
            outputs.append(F.log_softmax(logits, dim=-1))
            
        ret = self.__(  torch.stack(outputs), 'ret'  )
        return ret

    
# ## Loss and accuracy function
def loss(output, target, loss_function=nn.NLLLoss(), *args, **kwargs):
    loss = 0
    target = target[0]
    target = Variable(torch.LongTensor(target), requires_grad=False)
    if Config().cuda: target = target.cuda()
    output = output.transpose(1,0)
    log.debug('i, o sizes: {} {}'.format(output.size(), target.size()))
    batch_size = output.size()[0]
    for i, t in zip(output, target):

        loss += loss_function(i, t.squeeze()).mean()
        log.debug('loss size: {}'.format(loss.size()))

    del target
    return (loss/batch_size)

def accuracy(output, target, *args, **kwargs):
    accuracy, f1 = 0.0, 0.0
    accuracy, f1 = Variable(torch.Tensor([accuracy])), Variable(torch.Tensor([f1]))
    if Config().cuda:
        accuracy, f1 = accuracy.cuda(), f1.cuda()
    
    target = target[0]
    target = Variable(torch.LongTensor(target), requires_grad=False)
    if Config().cuda: target = target.cuda()
    output = output.transpose(1,0)
    batch_size = output.size()[0]
    class_size = output.size()[1]
    log.debug('i, o sizes: {} {}'.format(output.size(), target.size()))
    for i, t in zip(output, target):
        correct = (i.max(dim=1)[1] == t).sum()
        accuracy += correct.float()/class_size  
        #correct = (i.max(dim=1)[1] != t).sum()
        #accuracy += (correct == 0)
        i = i.max(dim=1)[1]

        tp = ( i * t ).sum().float()
        fp = ( i > t ).sum().float()
        fn = ( i < t ).sum().float()

        p = tp/ (tp + fp)
        r = tp/ (tp + fn)

        if tp.data[0] > 0:
            f1 += 2 * p * r/ (p + r)
        
    del target
    return (f1/batch_size)
    


# ### repr_function to build human readable output from model
from IPython.display import HTML
from IPython.display import display
def repr_function(output, feed, batch_index):
    results = []
    output = output.transpose(1,0)
    indices, (seq,), (classes,) = feed.nth_batch(batch_index)
    for i, o, s, c in zip(indices, output, seq, classes):
        orig_s = ' '.join(feed.data_dict[i].comment_text.split())
        s      = ' '.join([INPUT_VOCAB[i] for i in s])
        results.append([ str(z) for z in list(c)] + [orig_s, s])
        o = o.max(dim=1)[1]
        results.append([ str(z) for z in o.data.cpu().numpy().tolist()])
    del indices, seq, classes
    return results

def test_repr_function(output, feed, batch_index):
    results = []
    output = output.transpose(1,0)
    indices, (seq,), _ = feed.nth_batch(batch_index)
    for i, o, s in zip(indices, output, seq):
        o = o.max(dim=1)[1]
        o = [ str(z) for z in o.data.cpu().numpy().tolist()]
        s = ' '.join([INPUT_VOCAB[si] for si in s])
        results.append( [i] + o  + [s] )
    del indices, seq
    return results


# ## Experiment on model 1
import random
def  experiment(epochs=10, checkpoint=1, train_datapoints=train_datapoints):
    model =  Model(Config(), len(INPUT_VOCAB), len(OUTPUT_VOCAB))
    if Config().cuda:  model = model.cuda()
        
    split_index = int( len(train_datapoints) * 0.85 )
    train_feed = DataFeed(train_datapoints[:split_index], batchop=batchop, batch_size=128)
    test_feed = DataFeed(train_datapoints[split_index:], batchop=batchop, batch_size=120)

    trainer = Trainer(model=model, loss_function=loss, accuracy_function=accuracy, 
                    checkpoint=checkpoint, epochs=epochs,
                    feeder = Feeder(train_feed, test_feed))

    predictor = Predictor(model=model, repr_function=repr_function, feed=test_feed)

    for e in range(1):
        output, results = predictor.predict(random.choice(range(test_feed.num_batch)))
        display(HTML(results._repr_html_()))
        del output, results
        trainer.train()
        
        
# ## Experiment on model using attention
import random
import gc
def  experiment(eons=1000, epochs=1, checkpoint=1):
    model =  AttModel(Config(), len(INPUT_VOCAB), len(OUTPUT_VOCAB))
    if Config().cuda:  model = model.cuda()

    classified_train_feed = DataFeed(classified_datapoints[0], batchop=batchop, batch_size=128)
    classified_test_feed  = DataFeed(classified_datapoints[1], batchop=batchop, batch_size=128)
    classified_trainer = Trainer(model=model, 
                                 loss_function=loss, accuracy_function=accuracy, 
                                 checkpoint=checkpoint, epochs=epochs,
                                 feeder = Feeder(classified_train_feed, classified_test_feed))

    classified_predictor_feed     = DataFeed(classified_datapoints[1], batchop=batchop, batch_size=12)
    classified_predictor = Predictor(model=model, feed=classified_predictor_feed, repr_function=repr_function)

    non_classified_train_feed     = DataFeed(non_classified_datapoints[0], batchop=batchop, batch_size=128)
    non_classified_test_feed      = DataFeed(non_classified_datapoints[1], batchop=batchop, batch_size=128)
    non_classified_trainer = Trainer(model=model, 
                                     loss_function=loss, accuracy_function=accuracy, 
                                     checkpoint=checkpoint, epochs=epochs,
                                     feeder = Feeder(non_classified_train_feed, non_classified_test_feed))    

    non_classified_predictor_feed = DataFeed(non_classified_datapoints[1], batchop=batchop, batch_size=12)
    non_classified_predictor = Predictor(model=model, feed=non_classified_predictor_feed, repr_function=repr_function)

    test_predictor_feed = DataFeed(test_datapoints, batchop=test_batchop, batch_size=128)
    test_predictor = Predictor(model=model, feed=test_predictor_feed, repr_function=test_repr_function)
        
    dump = open('results/experiment_attn.csv', 'w')
    for e in range(eons):
        dump.write('#========================after eon: {}\n'.format(e))
        log.info('on {}th eon'.format(e))
        output, results = classified_predictor.predict(random.choice(range(classified_predictor_feed.num_batch)))
        dump.write(repr(results))
        del output, results
        output, results = non_classified_predictor.predict(random.choice(range(non_classified_predictor_feed.num_batch)))
        dump.write(repr(results))
        del output, results
        
        non_classified_trainer.train()
        for i in range(int(non_classified_train_feed.size/classified_train_feed.size/4)):
            classified_trainer.train()
        
        if not e % 100:
            test_results = ListTable()
            test_dump = open('results/experiment_attn_over_test_{}.csv'.format(e), 'w')
            test_dump.write('|'.join(['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', '\n']))
            log.info('running over test')
        
            for i in tqdm(range(test_predictor_feed.num_batch)):
                log.debug('i: {}'.format(i))
                output, results = test_predictor.predict(i)
                test_results += results
                
            test_dump.write(repr(test_results))            
            test_dump.close()

    model = model.cpu()
    torch.save(model.state_dict(), open('attn_model.pth', 'wb'))
    dump.close()

# ## Experiment on model using attention
import random
import gc
def  experiment(eons=1000, epochs=1, checkpoint=1):
    try:
        model =  BiLSTMDecoderModel(Config(), len(INPUT_VOCAB), len(CHAR_VOCAB), len(OUTPUT_VOCAB))
        if Config().cuda:  model = model.cuda()

        classified_train_feed = DataFeed(classified_datapoints[0], batchop=batchop, batch_size=128)
        classified_test_feed  = DataFeed(classified_datapoints[1], batchop=batchop, batch_size=128)
        
        classified_trainer = Trainer(model=model, 
                                     loss_function=loss, accuracy_function=accuracy, 
                                     checkpoint=checkpoint, epochs=epochs,
                                     feeder = Feeder(classified_train_feed, classified_test_feed))

        classified_predictor_feed     = DataFeed(classified_datapoints[1], batchop=batchop, batch_size=12)
        classified_predictor = Predictor(model=model, feed=classified_predictor_feed, repr_function=repr_function)

        non_classified_train_feed     = DataFeed(non_classified_datapoints[0], batchop=batchop, batch_size=128)
        non_classified_test_feed      = DataFeed(non_classified_datapoints[1], batchop=batchop, batch_size=128)
        non_classified_trainer = Trainer(model=model, 
                                         loss_function=loss, accuracy_function=accuracy, 
                                         checkpoint=checkpoint, epochs=epochs,
                                         feeder = Feeder(non_classified_train_feed, non_classified_test_feed))    

        non_classified_predictor_feed = DataFeed(non_classified_datapoints[1], batchop=batchop, batch_size=12)
        non_classified_predictor = Predictor(model=model, feed=non_classified_predictor_feed, repr_function=repr_function)

        test_predictor_feed = DataFeed(test_datapoints, batchop=test_batchop, batch_size=128)
        test_predictor = Predictor(model=model, feed=test_predictor_feed, repr_function=test_repr_function)

        dump = open('results/experiment_attn.csv', 'w')
        for e in range(eons):
            dump.write('#========================after eon: {}\n'.format(e))
            log.info('on {}th eon'.format(e))
            output, results = classified_predictor.predict(random.choice(range(classified_predictor_feed.num_batch)))
            dump.write(repr(results))
            del output, results
            output, results = non_classified_predictor.predict(random.choice(range(non_classified_predictor_feed.num_batch)))
            dump.write(repr(results))
            del output, results

            #non_classified_trainer.train()
            #for i in range(int(non_classified_train_feed.size/classified_train_feed.size)):
            classified_trainer.train()

            if e and not e % 10:
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

    except KeyboardInterrupt:
        torch.save(model.state_dict(), open('attn_model.pth', 'wb'))
        dump.close()
        return locals()

exp_image = experiment()
