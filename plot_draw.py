# scattering 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import os
import matplotlib

def plot_draw(x, y, z, xtitle, ytitle, ztitle, 
                    labels,pred_labels,save_path=None, title='Result'):
    """[플롯을 그려주는 코드]

    Arguments:
        x {[float]} -- [x축 점들]
        y {[float]} -- [y축 점들]]
        z {[float]} -- [z축 점들]
        xtitle {[str]} -- [x축 제목]
        ytitle {[str]} -- [y축 제목]]
        ztitle {[str]} -- [z축 제목]
        labels {[str]} -- [해당 라벨]

    Keyword Arguments:
        save_path {[str]]} -- [저장폴더, 경로와 파일명까지되야함. ex) ./path/to/dir/save.png] (default: {None})
        title {str} -- [그래프 제목] (default: {'Result'})
    """
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d') # Axe3D object
    makers= ['o', 'v', '^', '>', '<', '.', 'D', 'x']
    for xs, ys, zs, rl, pl in zip(x, y, z, labels, pred_labels):
        if rl != pl:
            ax.scatter(xs, ys, zs, c='r', s = 20, marker=makers[pl])
        else:
            ax.scatter(xs, ys, zs, s = 20, alpha=0.5, cmap=plt.cm.Greens, marker=makers[pl])
    
    if save_path:
        plt.savefig(os.getcwd(), save_path)
    
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    ax.set_zlabel(ztitle)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    sample_size = 500
    x = np.cumsum(np.random.normal(0, 5, sample_size))
    y = np.cumsum(np.random.normal(0, 5, sample_size))
    z = np.cumsum(np.random.normal(0, 5, sample_size))
    import random
    label = [random.choice([0,1,2]) for i in range(len(x))]
    plot_draw(x, y, z, xtitle='x', ytitle='y', ztitle='z', labels=label)