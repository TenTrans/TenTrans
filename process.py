import sys
from src.data.vocab import Vocabulary
import torch

assert len(sys.argv) >=4 , "python process.py vocab file  lang [shard_id](optional)"
vocab_path = sys.argv[1]
file_path = sys.argv[2]
lang = sys.argv[3]
shard_num = int(sys.argv[4]) if len(sys.argv) ==5 else 1


vocab = Vocabulary(file=vocab_path)

if shard_num == 1:
    data_bin = vocab.binarize_data(file_path)
    data_bin['lang'] = lang
    unk_words = (data_bin['sents'] == vocab.unk_index).sum()
    print("UNK words: {:.4f}%".format(100 * unk_words / len(data_bin['sents'])))
    print("UNK words: ", data_bin.get('unk_words',[]))
    print('Total sentences: {}'.format(len(data_bin['positions'])))
    torch.save(data_bin, file_path + ".pth", pickle_protocol=4)
else:

    data_bins = vocab.binarize_shard_data(file_path, shard_num)
    for shard_id, data_bin in enumerate(data_bins):
        data_bin['lang'] = lang
        unk_words = (data_bin['sents'] == vocab.unk_index).sum()
        print(f"Shard: {shard_id}")
        print("UNK words: {:.4f}%".format(100 * unk_words / len(data_bin['sents'])))
        print("UNK words: ", data_bin.get('unk_words',[]))
        print('Total sentences: {}'.format(len(data_bin['positions'])))
        torch.save(data_bin, file_path + f".pth.{shard_id}", pickle_protocol=4)
