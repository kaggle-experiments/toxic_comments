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
from datafeed import DataFeed, MultiplexedDataFeed
from utilz import tqdm, ListTable

from itertools import cycle
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
    class_count = [0] * 6  #num of output classes
    for datapoint in train_datapoints:
        classified_datapoints[tuple(datapoint[2:])].append(datapoint)
        class_count = [x+y for x, y in zip(class_count, datapoint[2:])]
            
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

    CHAR_VOCAB = ['<<PAD>>', '<<UNK>>', '<<EOS>>'] + list(set([c for dp in tqdm(datapoints) for c in ''.join(dp.comment_text.split())]))
    CHAR_INDEX = defaultdict(lambda : CHAR_VOCAB.index('<<UNK>>'))
    CHAR_INDEX.update({c: i for i, c in enumerate(CHAR_VOCAB)})

    OUTPUT_VOCAB = ['toxic','severe_toxic','obscene', 'threat','insult','identity_hate']
    INPUT_VOCAB  = [word for dp in tqdm(datapoints) for word in word_tokenize(dp.comment_text)]
    INPUT_VOCAB  = INPUT_VOCAB + OUTPUT_VOCAB
    for word in INPUT_VOCAB:
        WORD_FREQ[word] += 1

    WORD_FREQ_PAIRS = sorted(WORD_FREQ.items(), key=lambda x: -x[1])
    INPUT_VOCAB     = [ x[0] for x in WORD_FREQ_PAIRS ]

    print(WORD_FREQ_PAIRS[:100], WORD_FREQ_PAIRS[-100:])
    print('Vocab size: {}'.format(len(INPUT_VOCAB)))
    
    INPUT_VOCAB = ['<<PAD>>', '<<UNK>>', '<<EOS>>'] + INPUT_VOCAB + OUTPUT_VOCAB
    WORD_INDEX  = defaultdict(lambda : INPUT_VOCAB.index('<<UNK>>'))
    INPUT_VOCAB = INPUT_VOCAB[ :Config().input_vocab_size ]    
    WORD_INDEX.update( {w: i for i, w in enumerate(INPUT_VOCAB)} )

    OUTPUT_WORD_INDEX = {w: i for i, w in enumerate(OUTPUT_VOCAB)}
    OUTPUT_IDS = [OUTPUT_WORD_INDEX[i] for i in OUTPUT_VOCAB]

    PAD = WORD_INDEX[INPUT_VOCAB[0]]

    print('selvakumar is so stupid that he has no sense of purpose', WORD_INDEX['selvakumar is so stupid that he has no sense of purpose'])

    # caching
    pickle.dump([CHAR_VOCAB, dict(CHAR_INDEX),
                 INPUT_VOCAB, OUTPUT_VOCAB,
                 OUTPUT_IDS, PAD,
                 dict(WORD_INDEX), WORD_FREQ_PAIRS,
                 test_datapoints,  train_datapoints, classified_datapoints, class_count], open('cache.pkl', 'wb'))
else:
    [CHAR_VOCAB, CHAR_INDEX_DICT,
     INPUT_VOCAB, OUTPUT_VOCAB,
     OUTPUT_IDS, PAD,
     WORD_INDEX_DICT, WORD_FREQ_PAIRS,
     test_datapoints, train_datapoints, classified_datapoints, class_count] = pickle.load(open('cache.pkl', 'rb'))
    WORD_INDEX = defaultdict(lambda : INPUT_VOCAB.index('<<UNK>>'))
    WORD_INDEX.update(WORD_INDEX_DICT)

    CHAR_INDEX = defaultdict(lambda : CHAR_VOCAB.index('<<UNK>>'))
    CHAR_INDEX.update(CHAR_INDEX_DICT)

    
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
    def pad_seq_(seq):
        return seq[:maxlen] + [PAD]*(maxlen-len(seq))

    if len(seqs) == 0:
        return seqs
    
    if type(seqs[0]) == type([]):
        maxlen = maxlen if maxlen else seq_maxlen(seqs)
        seqs = [ pad_seq_(seq) for seq in seqs ]
    else:
        seqs = pad_seq_(seqs)
    return seqs

def batchop(datapoints, *args, **kwargs):
    indices = [d.id for d in datapoints]
    
    slimit = Config().seq_len_limit
    wlimit = Config().word_len_limit

    seq = []
    for d in datapoints:
        words = word_tokenize(d.comment_text)[:slimit]
        seqi = [WORD_INDEX[w] for w in words] + [WORD_INDEX['<<EOS>>']]
        seq.append(seqi)

    seq = pad_seq(seq)

    seq_char = []
    for d in datapoints:
        words = word_tokenize(d.comment_text)[:slimit]
        seq_chari = [[CHAR_INDEX[c] for c in w] + [CHAR_INDEX['<<EOS>>']]  for w in words]
        seq_chari = pad_seq(seq_chari, wlimit) 
        seq_chari = seq_chari + [[WORD_INDEX['<<EOS>>']] * wlimit]
        seq_char.append(seq_chari)

    seq_char = pad_seq(seq_char, PAD=[PAD] * wlimit)
    
    target = [(d.toxic, d.severe_toxic, d.obscene, d.threat, d.insult, d.identity_hate)
              for d in datapoints]

    seq, seq_char, target = np.array(seq), np.array(seq_char), np.array(target)
    return indices, (seq, seq_char), (target,)


def test_batchop(datapoints, *args, **kwargs):
    indices = [d.id for d in datapoints]
    
    slimit = Config().seq_len_limit
    wlimit = Config().word_len_limit

    seq = []
    for d in datapoints:
        words = word_tokenize(d.comment_text)[:slimit]
        seqi = [WORD_INDEX[w] for w in words] + [WORD_INDEX['<<EOS>>']]
        seq.append(seqi)

    seq = pad_seq(seq)

    seq_char = []
    for d in datapoints:
        words = word_tokenize(d.comment_text)[:slimit]
        seq_chari = [[CHAR_INDEX[c] for c in w] + [CHAR_INDEX['<<EOS>>']]  for w in words]
        seq_chari = pad_seq(seq_chari, wlimit) 
        seq_chari = seq_chari + [[WORD_INDEX['<<EOS>>']] * wlimit]
        seq_char.append(seq_chari)

    seq_char = pad_seq(seq_char, PAD=[PAD] * wlimit)

    seq, seq_char = np.array(seq), np.array(seq_char)
    return indices, (seq, seq_char), ()


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
        self._embed_char = nn.Embedding(self.char_input_vocab_size, self.char_embed_dim)
        self.embed_class = nn.Embedding(self.output_vocab_size, self.embed_dim)

        self.char_conv   = []

        filters_remaining = self.char_embed_dim                         #this need not be the case and  filters_remaining'ed vector will be concat to word_embedding
        for filter_size in cycle(Config.char_conv_filter_sizes):
            output_channels = min(filter_size, filters_remaining)
            self.char_conv.append(nn.Conv1d(self.char_embed_dim, output_channels, filter_size))
            filters_remaining -= output_channels

            if filters_remaining == 0:
                break


        self.word_conv   = []

        filters_remaining = self.embed_dim                         #this need not be the case and  filters_remaining'ed vector will be concat to word_embedding
        for filter_size in cycle(Config.word_conv_filter_sizes):
            output_channels = min(filter_size, filters_remaining)
            self.word_conv.append(nn.Conv1d(self.embed_dim + self.char_embed_dim, output_channels, filter_size))
            filters_remaining -= output_channels

            if filters_remaining == 0:
                break

            
            
        self.fencode = nn.LSTMCell(self.embed_dim + self.char_embed_dim, self.hidden_dim)
        self.bencode = nn.LSTMCell(self.embed_dim + self.char_embed_dim, self.hidden_dim)

        self.attend = nn.Parameter(torch.FloatTensor(self.embed_dim, 2*self.hidden_dim)) # class_emb @ self.attend @ seq_repr
        self.decode = nn.GRUCell(self.embed_dim + 2*self.hidden_dim, 2*self.hidden_dim) # convolved_seq + attended_output ;; fencode + bencode
        
        self.dropout = nn.Dropout(0.01)
        self.project = nn.Linear(2*self.hidden_dim, Config.project_dim)
        self.classify = nn.Linear(Config.project_dim, 2)

        self.log = model_logger.getLogger('model')
        self.size_log = self.log.getLogger('size')
        self.log.setLevel(logging.DEBUG)
        self.size_log.setLevel(logging.INFO)
            
        if Config.cuda:
            self.cuda()

    def cpu(self):
        super(BiLSTMDecoderModel, self).cpu()
        for conv in self.char_conv: conv.cpu()
        for conv in self.word_conv: conv.cpu()
        return self
    
    def cuda(self):
        super(BiLSTMDecoderModel, self).cuda()
        for conv in self.char_conv: conv.cuda()
        for conv in self.word_conv: conv.cuda()
        return self
    
    def __(self, tensor, name=''):
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            for i in range(len(tensor)):
                self.size_log.debug('{}[{}] -> {}'.format(name, i, tensor[i].size()))

        else:
            self.size_log.debug('{} -> {}'.format(name, tensor.size()))
                
        return tensor
        
    def init_hidden(self, batch_size):
        ret = torch.zeros(batch_size, self.hidden_dim)
        if Config().cuda: ret = ret.cuda()
        return Variable(ret, requires_grad=self.training)


    def embed_char(self, seq_char):
        batch_size, seq_size, word_size = seq_char.size()
        seq_char = seq_char.view(batch_size, -1)
        seq_char_emb = self._embed_char(seq_char)
        seq_char_emb = seq_char_emb.view(batch_size, seq_size, word_size, self.char_embed_dim)

        conv_result = []
        for seq in seq_char_emb:
            convs = []
            for filter in self.char_conv:
                seq = self.__( seq, 'seq_char[i]')
                res = self.__(  filter(seq.transpose(-2, -1)), filter )
                res = self.__(  res.mean(-1), '  after reduction')
                convs.append(res)

            convs = self.__( torch.cat(convs, -1), 'one seq' )
            conv_result.append(convs)            
        conv_result = self.__( torch.stack(conv_result), 'full batch')
        return conv_result


    def convolve_seq(self, seq_emb):
        convs = []
        batch_size, seq_size, embed_size = seq_emb.size()
        for filter in self.word_conv:
            if filter.kernel_size[0] < seq_size:
                res = self.__(  filter(seq_emb.transpose(-2, -1)), filter )
                res = self.__(  res.mean(-1), '  after reduction')
                convs.append(res)
            else:
                res = Variable(torch.zeros(batch_size, filter.out_channels), requires_grad=self.training)
                if Config().cuda: res = res.cuda()
                convs.append(res)
                
        convs = self.__( torch.cat(convs, -1), 'convolved seq' )
        return convs

    
    def forward(self, seq, seq_char, classes=OUTPUT_IDS):
        seq      = self.__( Variable(torch.LongTensor(seq)), 'seq')
        seq_char = self.__( Variable(torch.LongTensor(seq_char)), 'seq_char')
        classes  = self.__( Variable(torch.LongTensor(classes)),  'classes')

        dropout  = self.dropout
        
        if not self.training:
            dropout = lambda i: i
        
        if Config().cuda: 
            seq = seq.cuda()
            seq_char = seq_char.cuda()
            classes = classes.cuda()
            
        batch_size, seq_size = seq.size()
        pad_mask = (seq > 0).float()
        seq_emb = self.__(   dropout( F.tanh(self.embed(seq)) ), 'seq_emb'   )
        seq_char_emb = self.__(   dropout( F.tanh(self.embed_char(seq_char)) ), 'seq_char_emb'   )

        seq_emb = self.__(  torch.cat([seq_emb, seq_char_emb], -1), 'seq_emb')
        seq_emb = seq_emb.transpose(1, 0)

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
        output   = self.__(  seq_repr[-1], 'output')
        seq_repr = self.__(  seq_repr.transpose(1,0), 'seq_repr'  )

        convolved_seq = dropout( F.tanh(self.convolve_seq(seq_emb.transpose(1, 0))) )
        
        outputs = []
        for class_ in classes:
            class_emb = self.__( self.embed_class(class_), 'class_emb' ).squeeze(0)
            class_emb = self.__( F.tanh(class_emb).expand(seq_repr.size()[1], *class_emb.size()), 'class_emb' )

            self.__( self.attend, 'self.attend')
            attn = self.__(  torch.mm(F.tanh(class_emb), self.attend),   'attn')
            attn = self.__(  attn.expand(seq_repr.size()[0], *attn.size()), 'attn')

            attended_outputs = self.__(  torch.bmm(F.tanh(attn), seq_repr.transpose(1,2)), 'attended_outputs')
            attended_outputs = self.__(  attended_outputs.sum(dim=1), 'attended_outputs')
            attended_outputs = self.__(  attended_outputs.squeeze(1) * pad_mask, 'attended_outputs')
            
            attended_output = self.__(  F.tanh(attended_outputs).unsqueeze(-1) * seq_repr, 'attended_output')
            attended_output = self.__(  attended_output.sum(dim=1).squeeze(), 'attended_output')

            inp = self.__(  torch.cat([convolved_seq, attended_output], dim=-1), 'inp')
            output = self.__(  F.tanh( dropout(self.decode(inp, output)) ), 'output'  )
            logits = self.__(  self.classify(self.project(output)), 'logits'  )
            outputs.append(F.log_softmax(logits, dim=-1))
            
        ret = self.__(  torch.stack(outputs), 'ret'  )
        return ret

    
# ## Loss and accuracy function
def loss(output, target, loss_function=nn.NLLLoss(), scale=1, *args, **kwargs):
    loss = 0
    target = target[0]
    target = Variable(torch.LongTensor(target), requires_grad=False)
    if Config().cuda: target = target.cuda()
    output = output.transpose(1,0)
    log.debug('i, o sizes: {} {}'.format(output.size(), target.size()))
    batch_size = output.size()[0]
    for i, t in zip(output, target):

        loss += scale * loss_function(i, t.squeeze()).mean() # + 1/( (i.max(dim=1)[1] * t ).sum().float() + 0.001)
        log.debug('loss size: {}'.format(loss.size()))

    del target
    return (loss/batch_size)

def accuracy(output, target, *args, **kwargs):
    accuracy, f1 = 0.0, 0.0
    accuracy, f1 = Variable(torch.Tensor([accuracy]), requires_grad=False), Variable(torch.Tensor([f1]), requires_grad=False)
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
        #correct = (i.max(dim=1)[1] == t).sum()
        #accuracy += correct.float()/class_size  
        correct = (i.max(dim=1)[1] != t).sum()
        accuracy += (correct == 0).float()
        
    del target
    return (accuracy/batch_size)
    
def f1score_function(output, target, *args, **kwargs):
    p, r, f1 = 0.0, 0.0, 0.0
    p, r, f1 = Variable(torch.Tensor([p]), requires_grad=False), Variable(torch.Tensor([r]), requires_grad=False), Variable(torch.Tensor([f1]), requires_grad=False)
    if Config().cuda:
        p, r, f1 = p.cuda(), r.cuda(), f1.cuda()
    
    target = target[0]
    target = Variable(torch.LongTensor(target), requires_grad=False)
    if Config().cuda: target = target.cuda()
    output = output.transpose(1,0)
    batch_size = output.size()[0]
    class_size = output.size()[1]
    log.debug('i, o sizes: {} {}'.format(output.size(), target.size()))
    for i, t in zip(output, target):
        i = i.max(dim=1)[1]
        tp = ( i * t ).sum().float()
        fp = ( i > t ).sum().float()
        fn = ( i < t ).sum().float()

        if tp.data[0] > 0:
            p = tp/ (tp + fp)
            r = tp/ (tp + fn)
            f1 += 2 * p * r/ (p + r)
        
    del target
    return (p/batch_size), (r/batch_size), (f1/batch_size)
    


# ### repr_function to build human readable output from model
from IPython.display import HTML
from IPython.display import display
def repr_function(output, feed, batch_index):
    results = []
    output = output.transpose(1,0)
    indices, (seq, _), (classes,) = feed.nth_batch(batch_index)
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
    indices, (seq, _), _ = feed.nth_batch(batch_index)
    for i, o, s in zip(indices, output, seq):
        o = o.max(dim=1)[1]
        o = [ str(z) for z in o.data.cpu().numpy().tolist()]
        s = ' '.join([INPUT_VOCAB[si] for si in s])
        results.append( [i] + o  + [s] )
    del indices, seq
    return results

def  experiment(eons=1000, epochs=100, checkpoint=10):
    try:
        try:
            model =  BiLSTMDecoderModel(Config(), len(INPUT_VOCAB), len(CHAR_VOCAB), len(OUTPUT_VOCAB))
            if Config().cuda:  model = model.cuda()
            model.load_state_dict(torch.load('attn_model.pth'))
            log.info('loaded the old image for the model')
        except:
            log.exception('failed to load the model')
            model =  BiLSTMDecoderModel(Config(), len(INPUT_VOCAB), len(CHAR_VOCAB), len(OUTPUT_VOCAB))
            if Config().cuda:  model = model.cuda()

        print('**** the model', model)

        train_feed, test_feed, predictor_feed = {}, {}, {}
        trainer, predictor = {}, {}

        max_size = max( sorted(   [len(i[0]) for i in classified_datapoints.values()]   )[:-1] )
        #max_size = max( sorted(   [len(i[0]) for i in classified_datapoints.values()]   ) )
        
        for label in classified_datapoints.keys():
            if len(classified_datapoints[label][0]) < 1: continue

            label_desc = '-'.join([OUTPUT_VOCAB[l] for l in [i for i, x in enumerate(label) if x == 1]] )
            print('label: {} and size: {}'.format(label, len(classified_datapoints[label][0])))
            train_feed[label]      = DataFeed(label_desc, classified_datapoints[label][0], batchop=batchop, batch_size=max(128, int(len(classified_datapoints[label][0])/600))   )
            test_feed[label]       = DataFeed(label_desc, classified_datapoints[label][1], batchop=batchop, batch_size=32)
            predictor_feed[label]  = DataFeed(label_desc, classified_datapoints[label][1], batchop=batchop, batch_size=12)
            
            turns = int(max_size/train_feed[label].size) + 1            
            trainer[label] = Trainer(name=label_desc,
                                     model=model, 
                                     loss_function=partial(loss, loss_function=nn.NLLLoss(max(class_count)/torch.Tensor(class_count)), scale=1), accuracy_function=accuracy, f1score_function=f1score_function, 
                                     checkpoint=checkpoint, epochs=epochs,
                                     feeder = Feeder(train_feed[label], test_feed[label]))

            predictor[label] = Predictor(model=model, feed=predictor_feed[label], repr_function=repr_function)

        test_predictor_feed = DataFeed('test', test_datapoints, batchop=test_batchop, batch_size=128)
        test_predictor = Predictor(model=model, feed=test_predictor_feed, repr_function=test_repr_function)

        all_class_train_feed      = MultiplexedDataFeed('atrain',  train_feed.values(), batchop=batchop, batch_size=256)
        all_class_test_feed       = MultiplexedDataFeed('atest',   test_feed.values(), batchop=batchop, batch_size=256)
        all_class_predictor_feed  = MultiplexedDataFeed('apredict',predictor_feed.values(), batchop=batchop, batch_size=256)
        
        all_class_trainer = Trainer(name='all_class_trainer',
                                    model=model, 
                                    loss_function=partial(loss, scale=1), accuracy_function=accuracy, f1score_function=f1score_function, 
                                    checkpoint=checkpoint, epochs=epochs,
                                    feeder = Feeder(all_class_train_feed, all_class_test_feed))
        
        all_class_predictor = Predictor(model=model, feed=all_class_predictor_feed, repr_function=repr_function)

        label_trainer_triples = sorted( [(l, t, train_feed[l].size) for l, t in trainer.items()], key=lambda x: x[2] )
        log.info('trainers built {}'.format(pformat(label_trainer_triples)))

        dump = open('results/experiment_attn.csv', 'w').close()
        for e in range(eons):
            dump = open('results/experiment_attn.csv', 'a')
            dump.write('#========================after eon: {}\n'.format(e))
            dump.close()
            log.info('on {}th eon'.format(e))

            
            if e and not e % 1:
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

            with open('results/experiment_attn.csv', 'a') as dump:
                output, results = all_class_predictor.predict(random.choice(range(all_class_predictor_feed.num_batch)))
                dump.write(repr(results))
                del output, results
                
            all_class_trainer.train()
            
    except:
        log.exception('####################')
        torch.save(model.state_dict(), open('attn_model.pth', 'wb'))

        return locals()

exp_image = experiment()
