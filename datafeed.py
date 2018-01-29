from config import Config
from pprint import pprint, pformat
from logger import datafeed_logger
log = datafeed_logger.getLogger('main')
log.setLevel(Config.Log.DATAFEED.level)

from debug import memory_consumed

class DataFeed(object):

    def __init__(self,  datapoints, batchop, batch_size=1, vocab=None, sort_key=lambda x: len(x[1])):
        self._offset     = 0
        self._size       = len(datapoints)
        self._batch_size = batch_size
        self._batchop    = batchop
        self.vocab = vocab
        self._batch_cache = {}
        if len(datapoints):
            self.bind(sorted(datapoints, key=sort_key))

    def bind(self, datapoints):
        self._size = len(datapoints)
        self._data = datapoints
        self._data_dict = {}
        for d in datapoints:
            self._data_dict[d.id] = d
        self.reset_offset()

    @property
    def data(self):
        return self._data

    @property
    def data_dict(self):
        return self._data_dict
    
    @property
    def size(self):
        return self._size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_batch(self):
        return int(self.size/self.batch_size)
    
    @property
    def offset(self):
        return self._offset
    
    def batch(self):
        self._offset += self.batch_size
        return self._batchop(
            self.data[ self.offset - self.batch_size   :   self.offset ],
            self.vocab
        )
        
    def next_batch(self):
        try:
            if self.offset + self.batch_size > self.size:
                self.reset_offset()
            return self.batch()
        except:
            log.exception('batch failed')
            return self.next_batch()

    def nth_batch(self, n):
        return self._batchop(
            self.data[ n * self.batch_size   :   (n+1) * self.batch_size ],
            self.vocab
        )
        

    def reset_offset(self):
        self._offset = 0
