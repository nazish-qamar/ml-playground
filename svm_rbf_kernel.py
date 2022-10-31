import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC


class LoadChessData:
    def __init__(self):
        pass

    def load_chess_data(self):
        # Dataset available at https://www.kaggle.com/code/brnhurtado/starter-chess-game-dataset-lichess-b7c900fc-7/data?select=games.csv
        df = pd.read_csv('datasets/games.csv', encoding='utf-8')
        # Difference between white rating and black rating - independent variable
        df['rating_difference'] = df['white_rating'] - df['black_rating']

        # White wins flag (1=win vs. 0=not-win) - dependent (target) variable
        df['white_win'] = df['winner'].apply(lambda x: 1 if x == 'white' else 0)

        return df


class SvmRbf:
    def __init__(self, C, gamma):
        self.C = C
        self.gamma = gamma
        self.model = None
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

    def fit(self, X, y):
        # Create training and testing samples
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Fit the model
        self.model = SVC(kernel='rbf', probability=True, C=self.C, gamma=self.gamma)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        score_ = self.model.score(self.X_test, self.y_test)
        print('Accuracy Score: ', score_)


df = LoadChessData.load_chess_data()
X = df[['rating_difference', 'turns']]
y = df['white_win'].values

C_ = 1
gamma_ = 0.1
model = SvmRbf(C_, gamma_)
model.fit(X, y)
model.evaluate()
