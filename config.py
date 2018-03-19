import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class Config(Base):
    split_ratio = 0.70
    input_vocab_size = 30000
    hidden_dim = 200
    embed_dim = 50
    char_embed_dim = 25
    char_conv_filter_sizes = [2, 3, 4, 5, 6]
    word_conv_filter_sizes = [4, 8, 16, 32, 64]
    project_dim = 36
    output_vocab_size = 5
    batch_size = 2
    cuda = True
    tqdm = True
    find_lengths = False
    seq_len_limit = 500
    word_len_limit = 10
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
            level=logging.INFO

#Tests
assert Config.Embed.Char.size == 200
assert Config.Embed.Word.size == 300
