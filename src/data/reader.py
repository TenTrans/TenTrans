from .input_item import InputItem
import os
import torch

chunksize = 10**6

na_values = {'seq1': "", 'seq2': ""}


class CSVDataReader(object):

    ATTR = ['seq1', 'seq2', 'lang1', 'lang2', 'label1', 'label2', 'score']

    def __init__(self, data_folder, delimiter="\t"):
        self.data_folder = data_folder
        self.delimiter = delimiter

    def getInputItems(self, filename):
        import csv
        from tqdm import tqdm
        import pandas as pd

        input_file = os.path.join(self.data_folder, filename)
        assert filename.endswith(".csv")
        items = []
        with pd.read_csv(input_file,
                         encoding="utf-8",
                         sep='\t',
                         dtype=str,
                         quoting=csv.QUOTE_NONE,
                         chunksize=chunksize,
                         na_values=na_values) as reader:
            for df in reader:
                for idx, row in tqdm(list(df.iterrows())):
                    content = {
                        attr: row[attr]
                        for attr in df.columns if row[attr] != 'NaN'
                    }
                    item = InputItem(content)
                    items.append(item)
        return items


class BinaryDataReader(object):
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def getInputItems(self, filename):
        if ":" not in filename:
            input_file = os.path.join(self.data_folder, filename)
            return torch.load(input_file)
        else:
            file1, file2 = filename.split(":")[0], filename.split(":")[1]
            return torch.load(os.path.join(
                self.data_folder,
                file1)), torch.load(os.path.join(self.data_folder, file2))
