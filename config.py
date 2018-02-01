import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class Config(Base):
    split_ratio = 0.85
    input_vocab_size = 50000
    hidden_dim = 300
    embed_dim = 300
    bidirectional = True
    output_vocab_size = 5
    num_recurrent_layers = 10
    batch_size = 2
    cuda = True
    tqdm = True
    find_lengths = False
    seq_len_limit = 500
    flush = False
    class Embed(Base):
        class _default(Base):
            size = 200
        class Word(Base):
            size = 300

    class Log(Base):
        class _default(Base):
            level=logging.CRITICAL
        class PREPROCESS(Base):
            level=logging.DEBUG
        class MODEL(Base):
            level=logging.INFO
        class TRAINER(Base):
            level=logging.INFO
        class DATAFEED(Base):
            level=logging.DEBUG

#Tests
assert Config.Embed.Char.size == 200
assert Config.Embed.Word.size == 300
