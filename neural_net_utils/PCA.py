import numpy as np
import os.path as osp
from sklearn.decomposition import PCA as PCA_sklearn


class PCA():
    def __init__(self, n_components = None, mode = 'cov'):
        self.n_components = n_components
        self.mode = mode
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.components_ = None

    def fit_eig(self, input):
        if self.mode == 'cov':
            c = np.cov(input)
        elif self.mode == 'corr':
            c = np.corrcoef(input)

        self.singular_values_, self.components_ = np.linalg.eig(c)

        return self

    def fit_svd(self, input):
        # center
        self.mean_ = np.mean(input, axis=0)
        input -= self.mean_

        if self.mode == 'corr':
            # standardize
            self.std_ = np.std(input, axis = 0)
            input /= self.std_

        self.components_, self.singular_values_, _ = np.linalg.svd(input)

        return self

    def transform(self, input):
        pass

    def fit_transform(self, input):
        pass

    def inverse_transform(self, input):
        pass

def test():
    dir = '/home/eric/dataset_test/samples/sample92'
    ydiag = np.load(osp.join(dir, 'y_diag.npy'))
    mode = 'corr'

    pca = PCA(mode = mode).fit_eig(ydiag)
    print(pca.singular_values_)

    pca = PCA(mode = mode).fit_svd(ydiag.copy())
    print(pca.std_)
    print(pca.singular_values_)

    print('sklearn')
    pca = PCA_sklearn(svd_solver = 'full').fit(ydiag)
    print(pca.singular_values_)
    print(pca.explained_variance_)


if __name__ == '__main__':
    test()
