#Reference
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
#https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
#https://tensorflow.blog/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/2-3-7-%EC%BB%A4%EB%84%90-%EC%84%9C%ED%8F%AC%ED%8A%B8-%EB%B2%A1%ED%84%B0-%EB%A8%B8%EC%8B%A0/
#https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#https://tensorflow.blog/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/2-3-7-%EC%BB%A4%EB%84%90-%EC%84%9C%ED%8F%AC%ED%8A%B8-%EB%B2%A1%ED%84%B0-%EB%A8%B8%EC%8B%A0/
#https://datascienceschool.net/view-notebook/69278a5de79449019ad1f51a614ef87c/

import numpy as np
import copy
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import math
#Preprocessing
from sklearn.preprocessing import StandardScaler

#random variable
from scipy.stats import reciprocal, uniform
from scipy.stats import expon
from sklearn.utils.fixes import loguniform
def SVM_classfy(features, 
                            classes, 
                            estimator=None,
                            testdata=None,
                            **conf):
    """RBF커널에서는 gamma값에 따라 모양이 결정됨
    gamma의 값이 커짐에 따라서 각 샘플의 영향범위가 작고, 결정경계가 조금 더 불규칙적으로되서
    과적합 일으킬 가능성 있음.
    gamma를 감소시키면, 넓은 범위에 영향을 주기 때문에 부드러워지고 너무작으면 과소적합 발생
    적당한 gamma 중요.
    
    
    C의 값이 커질수록 규제가 없고, 작을수록 규제가 커진다.
    규제가 없다는뜻은 과적합이 일으킬 가능성이 있고, 작을수록 과소적합을 일으킨다.
    적당한 C가 중요

    Arguments:
        features {[list]} -- [입력값]
        classes {[str]} -- [클래스]

    Keyword Arguments:
        estimator {[분류기]]} -- [이전에 분류기모델이 들어왔으면 값이들어감(지금은 svm)] (default: {None})
        testdata {[테스트데이터가 들어온경우]} -- [description] (default: {None})

    Returns:
        [float, pred, confmat, estimators] -- [정확도(0.0~1.0), 결과리스트, 혼합행렬, estimators list]
    """
    
    C=conf['C']
    gamma=conf['gamma']
    search=conf['search']
    n_splits=conf['n_splits']
    test_size=conf['test_size']
        
    #SVM Preprocessing
    if conf['preprocessing']:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    if search =='grid':
        # RBF Sigma to Gamma
        # https://kr.mathworks.com/matlabcentral/answers/328796-gaussian-kernel-scale-for-rbf-svm
        # rs = 2.0**np.square(np.divide(1, np.arange(-2.1,7)))
        parameters = {'C':C,
                                    'gamma': gamma}
        svc = SVC(kernel='rbf')
        cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
        if estimator:
            estimator = estimator.set_params(estimator=estimator.best_estimator_,
                                                                    cv = cv,
                                                                    n_iter=conf['n_iter'], 
                                                                    param_distributions=parameters, 
                                                                    n_jobs=-1)
        else:
            estimator = GridSearchCV(svc,param_grid=parameters, cv=cv, verbose=2, n_jobs=-1)
        estimator.fit(features, classes)
        pred = estimator.best_estimator_.predict(features)
        totalAccuracy = accuracy_score(classes, pred)
        confmat = confusion_matrix(classes, pred)
        
    elif search=='random':
        #C: reciprocal(20, 200000)
        #g :expon(scale=1.0)
        # 'C':uniform(1, 10),
        # 'gamma': reciprocal(0.001, 0.1)
        '''
        C: stats.uniform(2, 10),
        gamma: stats.uniform(0.1, 1)}
        '''
        parameters = {'C':uniform(2**C[0], 2**C[1]),
                                    'gamma': uniform(2**gamma[0], 2**gamma[1])}
        svc = SVC(kernel='rbf')
        cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
        
        if estimator:
            estimator = estimator.set_params(estimator=estimator.best_estimator_,
                                                                    cv = cv,
                                                                    n_iter=conf['n_iter'], 
                                                                    param_distributions=parameters, 
                                                                    n_jobs=-1)
        else:
            estimator = RandomizedSearchCV (svc,
                                                                param_distributions=parameters, 
                                                                cv=cv,
                                                                n_iter=conf['n_iter'], 
                                                                verbose=2, 
                                                                n_jobs=-1)
        estimator.fit(features, classes)
        pred = estimator.best_estimator_.predict(features)
        totalAccuracy = accuracy_score(classes, pred)
        confmat = confusion_matrix(classes, pred)
    elif search == 'bayes':
        pass
    
    print("The best parameters are %s with a score of %0.2f"% (estimator.best_params_, estimator.best_score_))
    return totalAccuracy, pred, confmat, estimator

if __name__ == '__main__':

    selfeatures = np.loadtxt('selfeature.csv', delimiter=',')
    classes = np.loadtxt('classes.csv', delimiter=',')
    n = np.loadtxt('n.csv', delimiter=',')
    print(selfeatures.shape, classes.shape, n.shape)
    
    totalAccuracy, results, confmat = SVM_classfy(features=selfeatures, classes=classes) 
    