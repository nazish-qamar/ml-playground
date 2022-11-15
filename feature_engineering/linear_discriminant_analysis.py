import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from helper.load_data import LoadData


class LDA:
    def __init__(self, d, X_train_std, y_train, mean_vecs):
        self.d = d
        self.X_train_std = X_train_std
        self.y_train = y_train
        self.mean_vecs = mean_vecs
        self.S_W = np.zeros((d, d))
        self.S_B = np.zeros((d, d))

    def within_class_scatter_matrix(self):
        # number of features
        for label, mv in zip(range(1, 4), self.mean_vecs):
            class_scatter = np.zeros((self.d, self.d))  # scatter matrix for each class
            for row in self.X_train_std[self.y_train == label]:
                row, mv = row.reshape(self.d, 1), mv.reshape(self.d, 1)  # make column vectors
                class_scatter += (row - mv).dot((row - mv).T)
            self.S_W += class_scatter  # sum class scatter matrices

        print('Class label distribution: %s' % np.bincount(self.y_train)[1:])
        if(np.bincount(self.y_train)[1] != np.bincount(self.y_train)[2]
                or np.bincount(self.y_train)[2] != np.bincount(self.y_train)[3]):
            #compute covariance matrix instead
            self.S_W = np.zeros((self.d, self.d))
            for label, mv in zip(range(1, 4), self.mean_vecs):
                class_scatter = np.cov(self.X_train_std[self.y_train == label].T)
                self.S_W += class_scatter

        return self.S_W

    def between_class_scatter_matrix(self):
        mean_overall = np.mean(self.X_train_std, axis=0)
        for i, mean_vec in enumerate(self.mean_vecs):
            n = self.X_train_std[self.y_train == i + 1, :].shape[0]
            mean_vec = mean_vec.reshape(d, 1)  # make column vector
            mean_overall = mean_overall.reshape(d, 1)  # make column vector
            self.S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

        return self.S_B


_, X_train, X_test, y_train, y_test = LoadData().load_data(standardize=False, dataName="wine")

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

np.set_printoptions(precision=4)
# Each class mean vectors
mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print('MV %s: %s\n' % (label, mean_vecs[label - 1]))

d= 13 #d= number of features
lda_obj = LDA(d, X_train_std, y_train, mean_vecs)

# within class scatter matrix
S_W = lda_obj.within_class_scatter_matrix()
S_B = lda_obj.between_class_scatter_matrix()

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

# Sort the eigenvalue and eigenvector from high to low
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

print('Eigenvalues in descending order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])

tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

plt.bar(range(1, 14), discr, alpha=0.5, align='center', label='Individual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid', label='Cumulative "discriminability"')
plt.ylabel('"Discriminability" ratio')
plt.xlabel('Linear discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()

w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)

#for project the dataset to new feature space
X_train_lda = X_train_std.dot(w)