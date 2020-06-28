import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_result(history, classes):
    classes=classes.ravel()
    c=np.zeros((history.shape[0],1))
    for a in range(0, history.shape[0]):
        c[a,0] = history[a,12]
    cc = np.argwhere(c == 1)
    cc=cc[:,0]

    fig = plt.figure()
    ax=fig.add_subplot(111,projection='3d')  #3D
    ax = Axes3D(fig) #3D
    ax.set_xlim(left=2,right=-2)#3D

    for j in range(0, len(cc)):  # axis=1 because length in .m returns num of columns
        feature_pca = history[cc[j],7]

        for i in range(0, max(classes)):  # need to test this part
            ff = feature_pca[np.where(classes==i+1)][:]

            ax.scatter(ff[:, 1], ff[:, 0], ff[:, 2], s=10, alpha=1)    #3D
            # plt.scatter(ff[:,0],ff[:,1],s=20,marker='o',alpha=0.5) #2D

    plt.grid()
    plt.show()