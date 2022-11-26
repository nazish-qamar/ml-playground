import pandas as pd
import numpy as np
import os
import sys
import tarfile
import time
import urllib.request
import shutil

class IMDbReview:
    def __init__(self):
        self.url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        self.target = '../datasets/aclImdb_v1.tar.gz'
        self.df = None

    def download(self):
        #with urllib.request.urlopen(self.url) as response, open(self.target, 'wb') as out_file:
        #    shutil.copyfileobj(response, out_file)

        if not os.path.isdir('../datasets/aclImdb'):
            with tarfile.open(self.target, 'r:gz') as tar:
                tar.extractall(path="../datasets")

        self.preprocess()
        return self.df

    def preprocess(self):
        basepath = '../datasets/aclImdb'
        labels = {'pos': 1, 'neg': 0}
        df = pd.DataFrame()
        for s in ('test', 'train'):
            for l in ('pos', 'neg'):
                path = os.path.join(basepath, s, l)
                for file in sorted(os.listdir(path)):
                    with open(os.path.join(path, file),
                              'r', encoding='utf-8') as infile:
                        txt = infile.read()
                    df = df.append([[txt, labels[l]]],
                                   ignore_index=True)
        df.columns = ['review', 'sentiment']
        np.random.seed(0)
        self.df = df.reindex(np.random.permutation(df.index))
        self.df.to_csv('../datasets/movie_data.csv', index=False, encoding='utf-8')
