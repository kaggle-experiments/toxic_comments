from config import Config
from pprint import pprint, pformat
from logger import trainer_logger
log = trainer_logger.getLogger('main')
log.setLevel(Config.Log.TRAINER.level)

from debug import memory_consumed
from utilz import ListTable
from tqdm import tqdm as _tqdm

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
    
class LossCollector(list):
    def __init__(self, *args, **kwargs):
        super(LossCollector, self).__init__(*args, **kwargs)

    def backward(self):
        if len(self) > 0:
            return self[-1].backward()

    def __str__(self):
        if len(self) > 0:
            return '{}'.format(self[-1])
        
        return '<empty>'

    def append(self, a):
        super(LossCollector, self).append(a.data[0])

    def empty(self):
        del self[:]
    
class Snapshot(object):
    def __init__(self, filepath='snapshot.pth'):
        self.filepath = filepath
        self.history = []
    def save(self):
        try:
            log.info('saving the best model to file {}'.format(self.filepath))
            torch.save(self.best_model().state_dict(), self.filepath)
            log.info('saved.')
        except e:
            log.exception('unable to save snapshot of the best model')

    def best_model():
        sorted(self.history, key=lambda i: i[0])[0]

        
class Trainer(object):
    def __init__(self, runner=None, model=None,
                 feeder = None,
                 optimizer=None,
                 loss_function = None,
                 accuracy_function=None,
                 epochs=10000, checkpoint=1,
                 *args, **kwargs):
        
        self.__build_model_group(runner, model, *args, **kwargs)
        self.__build_feeder(feeder, *args, **kwargs)

        self.epochs     = epochs
        self.checkpoint = checkpoint

        self.optimizer     = optimizer     if optimizer     else optim.SGD(self.runner.model.parameters(), lr=0.01, momentum=0.1)
        self.loss_function = loss_function if loss_function else nn.NLLLoss()
        self.accuracy_function = accuracy_function if accuracy_function else self._default_accuracy_function
        self.train_loss = LossCollector()
        self.test_loss  = LossCollector()
        
        self.model_snapshots = []
        

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
                i, t = self.feeder.train.next_batch()
                output = self.runner.run(i)
                loss = self.loss_function( output, t )
                self.train_loss.append(loss)

                loss.backward()
                #print( [i.weight for i in self.runner.model.classify] )
                self.optimizer.step()
                
                if test_drive and j >= test_drive:
                    log.info('-- {} -- loss: {}'.format(epoch, self.train_loss))
                    return
                
                del i, t
                
            log.info('-- {} -- loss: {}'.format(epoch, self.train_loss))            
            if self.do_every_checkpoint(epoch) == FLAGS.STOP_TRAINING:
                log.info('loss trend suggests to stop training')
                self.model_snapshots.save()
                
    def do_every_checkpoint(self, epoch, early_stopping=True):
        if epoch % self.checkpoint != 0:
            return

        self.test_loss.empty()
        for j in tqdm(range(self.feeder.test.num_batch)):
            i, t = self.feeder.train.next_batch()
            outputs =  self.runner.run(i)
            loss = self.loss_function(outputs, t)
            self.test_loss.append(loss)

                    
        log.info('-- {} -- loss: {}, accuracy: {}'.format(epoch, self.test_loss, self.accuracy_function(outputs, t)))
        del i, t

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
        i = self.feed.nth_batch(batch_index)
        output = self.runner.run(i)
        output = self.repr_function(output, batch_index)
        results = ListTable().extend(output)
        return output, results
