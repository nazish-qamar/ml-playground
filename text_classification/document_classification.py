from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from helper.load_and_process_imdb import IMDbReview
from helper.text_cleaning import Text_Cleaner


class Document_Classifier:
    def __init__(self, df):
        self.X_train = df.loc[:2500, 'review'].values
        self.y_train = df.loc[:2500, 'sentiment'].values
        self.X_test = df.loc[2500:3000, 'review'].values
        self.y_test = df.loc[2500:3000, 'sentiment'].values
        self.stop = stopwords.words('english')
        self.porter = PorterStemmer()
        self.param_grid = None

    def tokenizer(self, text):
        return text.split()

    def tokenizer_porter(self, text):
        return [self.porter.stem(word) for word in text.split()]

    def create_param_grid(self):
        self.param_grid = [{'vect__ngram_range': [(1, 1)],
                            'vect__stop_words': [self.stop, None],
                            'vect__tokenizer': [self.tokenizer, self.tokenizer_porter],
                            'clf__penalty': ['l1', 'l2'],
                            'clf__C': [1.0, 10.0, 100.0]},
                           {'vect__ngram_range': [(1, 1)],
                            'vect__stop_words': [self.stop, None],
                            'vect__tokenizer': [self.tokenizer, self.tokenizer_porter],
                            'vect__use_idf': [False],
                            'vect__norm': [None],
                            'clf__penalty': ['l1', 'l2'],
                            'clf__C': [1.0, 10.0, 100.0]},
                           ]

    def text_classification(self):
        self.create_param_grid()
        tfidf = TfidfVectorizer(strip_accents=None,
                                lowercase=False,
                                preprocessor=None)
        lr_tfidf = Pipeline([('vect', tfidf),
                             ('clf', LogisticRegression(random_state=0, solver='liblinear'))])
        print("Starting Grid Search!!")
        gs_lr_tfidf = GridSearchCV(lr_tfidf, self.param_grid,
                                   scoring='accuracy',
                                   cv=5,
                                   verbose=2,
                                   n_jobs=1)

        print("Fit with the best parameters!!")
        gs_lr_tfidf.fit(self.X_train, self.y_train)

        print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
        print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
        clf = gs_lr_tfidf.best_estimator_
        print('Test Accuracy: %.3f' % clf.score(self.X_test, self.y_test))


df = IMDbReview().download()
df['review'] = df['review'].apply(Text_Cleaner().preprocessor)

Document_Classifier(df).text_classification()