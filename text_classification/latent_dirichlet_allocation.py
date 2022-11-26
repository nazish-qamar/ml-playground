import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class LatentDirichletAllocationAlgorithm:
    def __init__(self):
        self.X_topics = None

    def lda(self, df_movie):
        count = CountVectorizer(stop_words='english',
                                max_df=.1,
                                max_features=5000)
        X = count.fit_transform(df_movie['review'].values)
        lda_ = LatentDirichletAllocation(n_components=10,
                                         random_state=123,
                                         learning_method='batch')
        self.X_topics = lda_.fit_transform(X)

        # lda_.components_.shape
        n_top_words = 5
        feature_names = count.get_feature_names_out()

        for topic_idx, topic in enumerate(lda_.components_):
            print("Topic %d:" % (topic_idx + 1))
            print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

    def verify_result(self):
        action = self.X_topics[:, 9].argsort()[::-1]

        for iter_idx, movie_idx in enumerate(action[:3]):
            print('\nAction movie #%d:' % (iter_idx + 1))
            print(df['review'][movie_idx][:300], '...')


df = pd.read_csv('../datasets/movie_data.csv', encoding='utf-8')
model_ = LatentDirichletAllocationAlgorithm()
model_.lda(df)
model_.verify_result()