import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count = CountVectorizer()
docs = np.array([
        'I would like to travel to the end of the world',
        'It would be really exciting to try something like that',
        'I think I would head towards the north, or may be south'])

bag = count.fit_transform(docs)

print("Vocabulary: ", count.vocabulary_)
print("Feature Vector: ", bag.toarray())

np.set_printoptions(precision=2)

tfidf = TfidfTransformer(use_idf=True,
                         norm='l2',
                         smooth_idf=True)
print("tfidf: ", tfidf.fit_transform(count.fit_transform(docs)).toarray())