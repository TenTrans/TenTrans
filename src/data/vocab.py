# from torchtext import vocab
# from torchtext.data import Dataset
from collections import Counter, defaultdict
from typing import List
import numpy as np
import subprocess

UNK_WORD = '<unk>'
BOS_WORD = '<s>'
PAD_WORD = '<pad>'


class Vocabulary:
    """ Vocabulary represents mapping between tokens and indices. """
    def __init__(self,
                 tokens: List[str] = None,
                 file: str = None,
                 max_vocab=-1) -> None:
        """
        Create vocabulary from list of tokens or file.
        Special tokens are added if not already in file or list.
        File format: token with index i is in line i.
        :param tokens: list of tokens
        :param file: file to load vocabulary from
        """
        # don't rename stoi and itos since needed for torchtext
        # warning: stoi grows with unknown tokens, don't use for saving or size

        # special symbols
        # self.specials = [BOS_WORD, PAD_WORD, UNK_WORD]
        self.specials = [PAD_WORD, UNK_WORD, BOS_WORD]
        self.stoi = {}
        self.itos = []
        if tokens is not None:
            self._from_list(tokens)
        elif file is not None:
            self._from_file(file)

        self.pad_index = self.stoi[PAD_WORD]
        self.bos_index = self.stoi[BOS_WORD]
        self.unk_index = self.stoi[UNK_WORD]
        self.eos_index = self.bos_index

        if max_vocab > 0: self.max_vocab(max_vocab)

    def max_vocab(self, num):
        self.itos = self.itos[:num]
        self.stoi = {self.itos[i]: i for i in range(len(self.itos))}

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, i):
        return self.itos[i]

    def __contains__(self, w):
        return w in self.stoi

    def __eq__(self, y):
        if len(self) != len(y):
            return False
        return all(self.itos[i] == y[i] for i in range(len(y)))

    def index(self, word, no_unk=False):
        if no_unk:
            return self.stoi[word]
        else:
            return self.stoi.get(word, self.unk_index)

    def encode(self, text):
        ids = [self.index(w) for w in text.split()]
        return ids

    def decode(self, token_ids, no_special=False):
        if no_special:
            txt = [self[idx] for idx in token_ids if self[idx] not in [BOS_WORD, PAD_WORD]]
        else:
            txt = [self[idx] for idx in token_ids]
        return txt
    

    def _from_list(self, tokens: List[str] = None) -> None:
        """
        Make vocabulary from list of tokens.
        Tokens are assumed to be unique and pre-selected.
        Special symbols are added if not in list.
        :param tokens: list of tokens
        """
        self.add_tokens(tokens=self.specials + tokens)
        assert len(self.stoi) == len(self.itos)

    def _from_file(self, file: str) -> None:
        """
        Make vocabulary from contents of file.
        File format: token with index i is in line i.
        :param file: path to file where the vocabulary is loaded from
        """
        tokens = []
        with open(file, "r") as open_file:
            for line in open_file:
                tokens.append(line.strip("\n").split()[0])
        self._from_list(tokens)

    def __str__(self) -> str:
        return self.stoi.__str__()

    def to_file(self, file: str) -> None:
        """
        Save the vocabulary to a file, by writing token with index i in line i.
        :param file: path to file where the vocabulary is written
        """
        with open(file, "w") as open_file:
            for t in self.itos:
                open_file.write("{}\n".format(t))

    def add_tokens(self, tokens: List[str]) -> None:
        """
        Add list of tokens to vocabulary
        :param tokens: list of tokens to add to the vocabulary
        """
        for t in tokens:
            new_index = len(self.itos)
            # add to vocab if not already there
            if t not in self.stoi:
                self.itos.append(t)
                self.stoi[t] = new_index

    def is_unk(self, token: str) -> bool:
        """
        Check whether a token is covered by the vocabulary
        :param token:
        :return: True if covered, False otherwise
        """
        return self.stoi[token] == DEFAULT_UNK_ID()

    def __len__(self) -> int:
        return len(self.itos)

    def binarize_data(self, path):
        sents = []
        position = []
        unk_words = {}
        f = open(path, 'r', encoding='utf8')
        for lineno, line in enumerate(f):
            if lineno % 100000 == 0:
                print(lineno)

            line = line.strip()
            line_ = line.split()
            ids = self.encode(line)
            for i in range(len(ids)):
                if ids[i] == self.unk_index:
                    unk_words[line_[i]] = unk_words.get(line_[i], 0) + 1
            position.append([len(sents), len(sents) + len(ids)])
            sents.extend(ids)
            sents.append(self.eos_index)

        position = np.int64(position)
        if len(self) <= 1<< 16:
            sents = np.uint16(sents)
        else:
            sents = np.int32(sents)

        data = {
            'positions': position,
            'sents': sents,
            'word2id': dict(self.stoi),
            'unk_words':unk_words
        }
        return data

    def binarize_shard_data(self, path, shard_num):

        f = open(path, 'r', encoding='utf8')
        total_len = get_file_num(path)
        shard_len = total_len // shard_num

        shard_id = 0
        datas = []
        sents = []
        position = []
        unk_words = {}
        print(f"Total lines: {total_len}")
        for lineno, line in enumerate(f):
            line = line.strip()
            line_ = line.split()
            ids = self.encode(line)
            position.append([len(sents), len(sents) + len(ids)])
            sents.extend(ids)
            sents.append(self.eos_index)
            
            for i in range(len(ids)):
                if ids[i] == self.unk_index:
                    unk_words[line_[i]] = unk_words.get(line_[i], 0) + 1

            if lineno % 100000 == 0:
                 print("process line ",lineno)           
            
            if (lineno + 1) % shard_len == 0:
                position = np.int64(position)
                if len(self) <= 1<< 16:
                    sents = np.uint16(sents)
                else:
                    sents = np.int32(sents)
                data = {
                    'positions': position,
                    'sents': sents,
                    'word2id': dict(self.stoi),
                    'unk_words':unk_words
                }
     
                sents = []
                position = []
                print(f"shard {shard_id} is finished")
                shard_id += 1
                yield data

def get_file_num(fname):
    p = subprocess.Popen(['wc', '-l', fname],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])