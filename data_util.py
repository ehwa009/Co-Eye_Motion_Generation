import re
import unicodedata


class Lang:
    
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2 # include SOS and EOS


def unicode_to_ascii(string):
    return ''.join(
        c for c in unicodedata.normalize('NFD', string) 
        if unicodedata.category(c) != 'Mn'
    )
    

def normalize_string(string):
    string = unicode_to_ascii(string.lower().strip())
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z.!?]+", r" ", string)

    return string