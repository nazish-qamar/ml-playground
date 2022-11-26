import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version


class OutOfCoreLearning:
    def __init__(self):
        self.vect = None
        self.doc_stream = None
        self.clf = None
        self.stop = stop = stopwords.words('english')

    def tokenizer(self, text):
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
        text = re.sub('[\W]+', ' ', text.lower()) + \
               ' '.join(emoticons).replace('-', '')
        tokenized = [w for w in text.split() if w not in self.stop]
        return tokenized

    def stream_docs(self, path):
        with open(path, 'r', encoding='utf-8') as csv:
            next(csv)  # skip header
            for line in csv:
                text, label = line[:-3], int(line[-2])
                yield text, label

    def get_minibatch(self, size):
        docs, y = [], []
        try:
            for _ in range(size):
                text, label = next(self.doc_stream)
                docs.append(text)
                y.append(label)
        except StopIteration:
            return None, None
        return docs, y

    def training(self, path_):
        self.vect = HashingVectorizer(decode_error='ignore',
                                      n_features=2 ** 21,
                                      preprocessor=None,
                                      tokenizer=self.tokenizer)

        self.clf = SGDClassifier(loss='log', random_state=1)
        self.doc_stream = self.stream_docs(path=path_)

        classes = np.array([0, 1])
        for _ in range(45):
            X_train, y_train = self.get_minibatch(size=1000)
            if not X_train:
                break
            X_train = self.vect.transform(X_train)
            self.clf.partial_fit(X_train, y_train, classes=classes)

    def compute_accuracy(self):
        X_test, y_test = self.get_minibatch(size=5000)
        X_test = self.vect.transform(X_test)
        print('Accuracy: %.3f' % self.clf.score(X_test, y_test))
        clf = self.clf.partial_fit(X_test, y_test)


model_ = OutOfCoreLearning()
model_.training(path_="../datasets/movie_data.csv")
model_.compute_accuracy()
