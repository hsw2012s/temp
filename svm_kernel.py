from sklearn import SVC
from scipy.stats import expon, reciprocal
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

param_grid = [
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
        'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]


svm_clf = SVC()
grid_search = GridSearchCV(svm_clf, param_grid, cv=5, scoring='accuracy', 
                        verbose=2, n_jobs=-1)
grid_search.fit(housing_prepared, housing_labels)


grid_kernel_svm_clf = Pipeline([
        ('feature')
    
    ])