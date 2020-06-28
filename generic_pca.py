import numpy as np
from numpy import linalg as la
from scipy import io
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from scipy import linalg
# Make all matlib functions accessible at the top level via M.func()
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline


def gpca(feature_total):
    '''
    feature_total = []
    c = []
    for i in range(history.shape[0]):
        c.append(history[i, 12].item())
    c = np.array(c)
    c = c.reshape(history.shape[0], -1)

    def indices(a, func):
        # np.array([i for (i, val) in enumerate(a) if func(val)]).reshape(-1, 1)
        return [i for (i, val) in enumerate(a) if func(val)]

    cc = indices(c, lambda x: x == 1)

    for j in range(len(cc)):
        # Matlab to python index start 0, cuz index -1
        
        try:
            selfeature = history[c[cc[j]]-1, 6-1][0]
        except:
            idx = np.asscalar(c[cc[j]])
            selfeature = history[idx-1, 6-1][0]
        feature_total.append(selfeature)
        

    feature_total = np.hstack(feature_total)
    '''
    '''
    #이부분은 PCA로 하면 소스코드의 길이를 많이 줄일 수 있음.
    # (EigenVector를 표현하는 방법은 Matlab과 Python이 약간 다르나 고윳값 표현에 있어서 동치(Equivalent)기 때문에 Classification의 영향을 미치지는 않음)
    # mean_centered = StandardScaler().fit_transform(feature_total.reshape(-1, 1))
    cov_matrix = np.cov(feature_total.T)

    # EigenVectors
    # https://blogs.sas.com/content/iml/2017/08/07/eigenvectors-not-unique.html
    # https://kr.mathworks.com/matlabcentral/answers/224516-why-matlab-function-eigs-has-different-results-for-the-same-input-data
    eig_vals = np.real(la.eig(cov_matrix)[1])
    # eig_vals =  np.real(linalg.eig(cov_matrix,left=False, right=True)[1])

    eig_vals = -np.sign(eig_vals[0, :]).transpose() * eig_vals
    # eig_vals[:,:0] = -eig_vals[:,:0]
    # eig_vals[:,:2] = -eig_vals[:,:2]

    total_variance = np.sum(eig_vals)
    # the cumulative sum of the eigenvalues is your % variance explained
    Trainfeature = np.dot(feature_total, eig_vals[:, :3])
    '''
    feature_total = StandardScaler().fit_transform(feature_total)
    pca_estimator = PCA(n_components=3, svd_solver='full')
    Trainfeature = pca_estimator.fit_transform(feature_total)
    return Trainfeature, pca_estimator.get_params(), pca_estimator


if __name__ == '__main__':
    history = io.loadmat('history1.mat')
    history = history['history']

    featrue, total = gpca(history=history, n=2)
    feature_pca = np.loadtxt('feature_pca1.csv', delimiter=',')
    print(featrue)
    print(feature_pca)

    #feature_tot = np.loadtxt('feature_total.csv', delimiter=',')

    res1 = np.round(featrue - feature_pca, 4)
    # res2 = np.round(feature_tot-total, 4)
    # print(res1)
    # print(res2)
