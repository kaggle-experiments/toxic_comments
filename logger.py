import logging

logging.basicConfig(format="%(asctime)s:%(name)s:%(levelname)-7s:%(funcName)20s:: %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
class MyLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super(MyLogger, self).__init__(name, level)
        self.name = name

    def getLogger(self, name=''):
        if name:
            logger.info('creating logger for {} under {}'.format(name, self.name))
            return logging.getLogger('{}.{}'.format(self.name, name))
        else:
            return self

    def log(self, lvl, msg, *args, **kwargs):
        for m in msg.splitlines():
            super(MyLogger, self).log(lvl, msg, *args, **kwargs)
            
logging.setLoggerClass(MyLogger)
preprocess_logger = logging.getLogger('PREPROCESS')

model_logger = logging.getLogger('MODEL')
trainer_logger = logging.getLogger('TRAINER')
datafeed_logger = logging.getLogger('DATAFEED')
utilz_logger = logging.getLogger('UTILZ')
torch_wrapper_logger = logging.getLogger('TORCH_WRAPPER')
