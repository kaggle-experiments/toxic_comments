from nltk import word_tokenize
import re
from utilz import squeeze

import resources as R


class Datapoint(object):

    def __init__(self, dataitem, idx=None):
        """
        Encapsulates data attributes

        """
        self._idx = idx if idx else dataitem.idx # unique index
        self._text     = dataitem.text # raw text
        self._entities = dataitem.entities # list of entity named tuples
        self._e1       = dataitem.relation.e1 # DRUG 1
        self._e2       = dataitem.relation.e2 # DRUG 2
        self._sequence = self.mask_sequence() # masked sequence

        # update sequence to separate fused DRUG mentions
        self._sequence = self.mutate_sequence()

        try:
            self._e1_idx, self._e2_idx = self.fetch_indices()
        except Exception as e:
            print('\n\n', self._sequence)
            print(e)

        self._relation = dataitem.relation.type

    def fetch_indices(self):
        """
        """
        seq = self._sequence
        def fetch_index(idx):
            tag = 'DRUG_{}'.format(idx)
            #x = seq.find(tag)
            #return len(word_tokenize(seq[:x]))
            return word_tokenize(seq).index(tag)

        return fetch_index(1), fetch_index(2)

    def mutate_sequence(self):
        """
        FILL DIS UP!!
        """
        # compile REGEX
        fused_drug_pattern_ = re.compile('(DRUG_\d+|DRUG_N)')
        tokens = []
        for token in word_tokenize(self._sequence):
            match_ = fused_drug_pattern_.search(token)
            if match_ and len(token) > 6:

                if len(fused_drug_pattern_.findall(token)) > 1:
                    if '/' in token:
                        delim = '/'
                    else:
                        delim = '-'
                    tokens.extend(token.split(delim))

                # when you spot fused DRUG pattern
                #  insert spaces ruthlessly
                s, e = match_.start(), match_.end()
                mutated_token = token[:s] + ' ' + token[s:e] + ' ' + token[e:]
                tokens.extend(mutated_token.split(' '))
            else: # add normal token to list
                tokens.append(token)

        return ' '.join(tokens)

    def mask_sequence(self):
        """
        Replace drugs in text with DRUG_1, DRUG_2, DRUG_N

        Args:
            None

        Returns:
            masked sequence
        """
        # fetch raw text
        chi2mask = { i:0 for i,ch in enumerate(self._text) }
        tags = [ '_', 'DRUG_1', 'DRUG_2', 'DRUG_N' ]

        def _update_entity_mask(ent, s, e):
            if ent   == self._e1: # drug 1
                tag = 1
            elif ent == self._e2: # drug 2
                tag = 2
            else: # other drugs
                tag = 3
            for i in range(s,e+1):
                chi2mask[i] = tag

        for ent in self._entities:
            # get entity offset in text
            #delimiter = '-' if '-' in ent.charOffset else ';'
            try:
                for se in ent.charOffset.strip().split(';'):
                    s, e = [ int(x) for x in se.split('-') ]
                    _update_entity_mask(ent, s, e)
            except:
                print(ent.charOffset)

        masked_sequence = ''
        i = 0
        try:
            while i < len(self._text):
                ch = self._text[i]
                if not chi2mask[i]:
                    masked_sequence += ch
                    i +=1
                else:
                    masked_sequence += tags[chi2mask[i]]
                    while i < len(chi2mask) and chi2mask[i] > 0:
                        i += 1
        except Exception as e:
            print('ERROR')
            print(masked_sequence)

        return masked_sequence

    def __str__(self):
        return '{}\n{}\ne1 : {} @ {}\t e2 : {} @ {}\ntype : {}\nentities : {}'.format(
                self._text, self._sequence, 
                self._e1.text, self._e1_idx,
                self._e2.text, self._e2_idx,
                self._relation,
                self._entities)

    def seqlen(self):
        return len(word_tokenize(self._sequence))


if __name__ == '__main__':
    # fetch dataitems from endpoint
    from sem_eval2013_task9 import _pipe_0
    train, test = _pipe_0()
    # create datapoints
    trainset = [ Datapoint(di) for di in train ]
    print('\n\n', trainset[200])
    print('\n\n', trainset[210])
