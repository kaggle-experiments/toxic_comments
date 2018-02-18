
from config import Config
from pprint import pprint, pformat
from logger import trainer_logger
log = trainer_logger.getLogger('main')
log.setLevel(Config.Log.TRAINER.level)

from debug import memory_consumed
from utilz import ListTable
from tqdm import tqdm as _tqdm

import torch

from torch import optim, nn

from collections import namedtuple

Feeder = namedtuple('Feeder', ['train', 'test'])

def tqdm(a):
    return _tqdm(a) if Config().tqdm else a

class FLAGS:
    CONTINUE_TRAINING = 0
    STOP_TRAINING = 1

class Runner(object):
    def __init__(self, model, *args, **kwargs):
        self._model = model

    def run(self, input):
        model_output = self.model(*input)
        return model_output

    @property
    def model(self):
        return self._model
    
class Averager(list):
    def __init__(self, filename=None, *args, **kwargs):
        super(Averager, self).__init__(*args, **kwargs)
        if filename:
            open(filename, 'w').close()

        self.filename = filename
        
    @property
    def avg(self):
        return sum(self)/len(self)
        
    def __str__(self):
        if len(self) > 0:
            return 'min/max/avg/latest: {:0.5f}/{:0.5f}/{:0.5f}/{:0.5f}'.format(min(self), max(self), self.avg, self[-1])
        
        return '<empty>'

    def append(self, a):
        super(Averager, self).append(a.data[0])

    def empty(self):
        del self[:]

    def write_to_file(self):
        if self.filename:
            with open(self.filename, 'a') as f:
                f.write(self.__str__() + '\n')
    
class Snapshot(list):
    def __init__(self, filepath='snapshot.pth'):
        self.filepath = filepath
    def save(self):
        try:
            log.info('saving the best model to file {}'.format(self.filepath))
            torch.save(self.best_model(), self.filepath)
            log.info('saved.')
        except:
            log.exception('unable to save snapshot of the best model')

    def best_model(self):
        sorted(self, key=lambda i: i[0])[0][1]

        
class Trainer(object):
    def __init__(self, name, runner=None, model=None,
                 feeder = None,
                 optimizer=None,
                 loss_function = None,
                 accuracy_function=None,
                 f1score_function=None,
                 epochs=10000, checkpoint=1,
                 directory='results',
                 *args, **kwargs):
        
        self.__build_model_group(runner, model, *args, **kwargs)
        self.__build_feeder(feeder, *args, **kwargs)

        self.epochs     = epochs
        self.checkpoint = checkpoint

        self.optimizer     = optimizer     if optimizer     else optim.Adam(self.runner.model.parameters())

        # necessary metrics
        self.train_loss = Averager(filename = '{}/{}.{}.{}'.format(directory, name, 'metrics',  'train_loss'))
        self.test_loss  = Averager(filename = '{}/{}.{}.{}'.format(directory, name, 'metrics', 'test_loss'))
        self.accuracy_function = accuracy_function if accuracy_function else self._default_accuracy_function

        self.accuracy   = Averager(filename = '{}/{}.{}.{}'.format(directory, name, 'metrics', 'accuracy'))
        self.loss_function = loss_function if loss_function else nn.NLLLoss()

        # optional metrics
        self.f1score_function = f1score_function
        self.precision = Averager(filename = '{}/{}.{}.{}'.format(directory, name, 'metrics', 'precision'))
        self.recall = Averager(filename = '{}/{}.{}.{}'.format(directory, name, 'metrics', 'recall'))
        self.f1score   = Averager(filename = '{}/{}.{}.{}'.format(directory, name, 'metrics', 'f1score'))

        self.metrics = [self.train_loss, self.test_loss, self.accuracy, self.precision, self.recall, self.f1score]
        
        self.model_snapshots = Snapshot()
        

    def __build_model_group(self, runner, model, *args, **kwargs):
        assert model is not None or runner is not None, 'both model and runner are None, fatal error'
        if runner:
            self.runner = runner
        else:
            self.runner = Runner(model)
            
    def __build_feeder(self, feeder, *args, **kwargs):
        assert feeder is not None, 'feeder is None, fatal error'
        self.feeder = feeder
        
    def train(self, test_drive=False):
        self.runner.model.train()
        for epoch in range(self.epochs):
            log.critical('memory consumed : {}'.format(memory_consumed()))            
                
            for j in tqdm(range(self.feeder.train.num_batch)):
                self.optimizer.zero_grad()
                _, i, t = self.feeder.train.next_batch()
                output = self.runner.run(i)
                loss = self.loss_function( output, t )
                self.train_loss.append(loss)

                loss.backward()
                self.optimizer.step()
                
                if test_drive and j >= test_drive:
                    log.info('-- {} -- loss: {}'.format(epoch, self.train_loss))
                    return

                del _, i, t
                del output, loss
            
            log.info('-- {} -- loss: {}'.format(epoch, self.train_loss))            
            if self.do_every_checkpoint(epoch) == FLAGS.STOP_TRAINING:
                log.info('loss trend suggests to stop training')
                return
            
            
            for m in self.metrics:
                m.write_to_file()
                
        self.model_snapshots.save()
        self.runner.model.eval()
        
    def do_every_checkpoint(self, epoch, early_stopping=True):
        if epoch % self.checkpoint != 0:
            return
        self.runner.model.eval()
        self.test_loss.empty()
        self.accuracy.empty()
        for j in tqdm(range(self.feeder.test.num_batch)):
            _, i, t = self.feeder.train.next_batch()
            output =  self.runner.run(i)

            loss = self.loss_function(output, t)
            self.test_loss.append(loss)
            accuracy = self.accuracy_function(output, t)
            self.accuracy.append(accuracy)

            if self.f1score_function:
                precision, recall, f1score = self.f1score_function(output, t)
                self.precision.append(precision)
                self.recall.append(recall)
                self.f1score.append(f1score)
            del output, loss, accuracy
            del _, i, t
            
        log.info('-- {} -- loss: {}, accuracy: {}'.format(epoch, self.test_loss, self.accuracy))
        log.info('-- {} -- precision: {}'.format(epoch, self.precision))
        log.info('-- {} -- recall: {}'.format(epoch, self.recall))
        log.info('-- {} -- f1score: {}'.format(epoch, self.f1score))

        self.model_snapshots.append((self.test_loss.avg, self.runner.model.state_dict()))
        if early_stopping:
            return self.loss_trend()


    def loss_trend(self):
        if len(self.test_loss) > 4:

            losses = self.test_loss[-3:]
            count = 0
            for l, r in zip(losses, losses[1:]):
                if l < r:
                    count += 1
            else:
                if count > 2:
                    return FLAGS.STOP_TRAINING

        return FLAGS.CONTINUE_TRAINING


    def _default_accuracy_function(self):
        return -1

    
class Predictor(object):
    def __init__(self, runner=None, model=None,
                 feed = None,
                 repr_function = None,
                 *args, **kwargs):
        
        self.__build_model_group(runner, model, *args, **kwargs)
        self.__build_feed(feed, *args, **kwargs)
        self.repr_function = repr_function
        
    def __build_model_group(self, runner, model, *args, **kwargs):
        assert model is not None or runner is not None, 'both model and runner are None, fatal error'
        if runner:
            self.runner = runner
        else:
            self.runner = Runner(model)
            
    def __build_feed(self, feed, *args, **kwargs):
        assert feed is not None, 'feed is None, fatal error'
        self.feed = feed
        
    def predict(self,  batch_index=0):
        _, i, *__ = self.feed.nth_batch(batch_index)
        self.runner.model.eval()
        output = self.runner.run(i)
        results = ListTable()
        results.extend( self.repr_function(output, self.feed, batch_index) )
        output_ = output.data
        del output
        del _, i, __
        return output_, results
