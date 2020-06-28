import numpy as np
import matplotlib.pyplot as plt
import scipy.io
class Descending:
    """ for np_sortrows: sort column in descending order """
    def __init__(self, column_index):
        self.column_index = column_index

    def __int__(self):  # when cast to integer
        return self.column_index


def np_sortrows(M, columns=None):
    """  sorting 2D matrix by rows
    :param M: 2D numpy array to be sorted by rows
    :param columns: None for all columns to be used,
                    iterable of indexes or Descending objects
    :return: returns sorted M
    """
    if len(M.shape) != 2:
        raise ValueError('M must be 2d numpy.array')
    if columns is None:  # no columns specified, use all in reversed order
        M_columns = tuple(M[:, c] for c in range(M.shape[1]-1, -1, -1))
    else:
        M_columns = []
        for c in columns:
            M_c = M[:, int(c)]
            if isinstance(c, Descending):
                M_columns.append(M_c[::-1])
            else:
                M_columns.append(M_c)
        M_columns.reverse()

    return M[np.lexsort(M_columns), :]



def SVMranking(confusionmatrix=None):
    # confusionmatrix에서 데이터 연산으로 combination 생성
    # combination[i] = [i , j , confusionmatrix[i,j]+confusionmatrix[j,i], confusionmatrix의 i열 합 + j열 합 - confusionmatrix[i][i] - confusionmatrix[j][j]]
    combination = np.array([0, 0, 0, 0])
    
    for i in range(0, confusionmatrix.shape[0] - 1):
        for j in range(i + 1, confusionmatrix.shape[0]):
            Combi = np.array([i + 1, j + 1, confusionmatrix[i, j] +
                            confusionmatrix[j, i],
                            (np.sum(confusionmatrix[:, i]) + np.sum(confusionmatrix[:, j])) - 
                            (confusionmatrix[i, i] + confusionmatrix[j, j])])
            combination = np.r_[combination, Combi]
    
    # 4열짜리 2차원배열로 정리
    combination = combination.reshape(-1, 4)[1:,:]

    # 세번째열을 기준으로 정렬 후 세번째열이 같은값이면 네번째열을 기준으로 정렬
    combination = combination[combination[:, 2].argsort()]

    R = np_sortrows(combination, [2, 3, 0, 1])
    Ranking = np.flip(R, axis=0)
    # print(Ranking)
    Ranking = Ranking[Ranking[:, 2]>0]
    return Ranking, combination

if __name__ == "__main__":
    from scipy import io
    confmat = io.loadmat('confmat.mat')
    confmat= confmat['confmat']
    # print(confmat)
    
    SVMranking(confusionmatrix=confmat)