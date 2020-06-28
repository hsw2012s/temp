import os
import numpy as np
import pickle
import gzip
from collections import OrderedDict
import json
import joblib
from scipy.stats import reciprocal, uniform
from scipy.stats import expon

class AutoML_Config():
    ODICT = "odict"
    def __init__(self, *args, **kwargs):
        '''
        Parameter
        ---------
        
        '''
        
        self.__dict__[self.ODICT] = OrderedDict()

    #Save GA Model Path
    
    # self.__dict__['GA']['path'] = os.path.join(os.getcwd(), 'ga_model.pkl')
    #Save SVM Model Path
            
    #SVM Model
    # self.__dict__['SVM']['path'] = os.path.join(os.getcwd(), 'svm_model.pkl')        

        
    def __getattr__(self, item):
        return self.__dict__[self.ODICT][item]


    
    def __setattr__(self, key, value):
        self.__dict__[self.ODICT][key] = value    
        
    def save_ML_Model(self, model, path):
        joblib.dump(model, path) # DIFF
    
    def load_ML_Model(self, path):    
        return joblib.load(path)
    
    #Parameter Save
    # def saveConfig(self, path):
    #     #History Path
    #     savePath=os.path.join(os.getcwd(), path)
    #     with open(savePath, 'w', encoding='utf-8') as f:
    #         print(json.dumps(self.__dict__, ensure_ascii=False, indent='\t'))
    #         json.dump(self.__dict__, f, ensure_ascii=False, indent='\t')
    pass
    
def set_config(default_dict=None, **kwargs):
    """ Get pre-defined Genetic Algorithm configuration.

    :return: Config object for GA.
    """
    
    #Input Path
    conn = AutoML_Config()
    for k, v in kwargs.items():
        setattr(conn, k, v)
    if default_dict:
        for k, v in default_dict.items():
            if  not (k in vars(conn)[AutoML_Config.ODICT].keys()):
                setattr(conn, k, v)
        
    return conn


def create_train_config_file(filename):
    # svm_source = set_svm_config()
    config = AutoML_Config()
    #https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
    svm_default_dict = {
                                'C': list(2.0**np.arange(-5, 17, 2)),
                                'gamma':list(2.0**np.arange(-7,3)),
                                #'gamma':np.arange( 0.0, 10.0+0.0, 0.1 ).tolist(),
                                'n_splits':1,
                                'test_size':0.33,
                                'search':'grid',
                                'preprocessing': True
                            }

    svm_rand_dict = {
                                'C':[1, 1],#[0.1, 17]
                                'gamma':[1, 1],#[0.1, 10]
                                #'gamma':np.arange( 0.0, 10.0+0.0, 0.1 ).tolist(),
                                'n_splits':1,
                                'n_iter':1,
                                'test_size':0.33,
                                'search':'random',
                                'preprocessing': True
                            }
    ga_default_dict={
                        'cv':3, 
                        'scoring':'accuracy',
                        'max_features':3,
                        'n_population':100,
                        'crossover_proba':0.5,
                        'mutation_proba':0.2,
                        'n_generations':150,
                        'crossover_independent_proba':0.5,
                        'mutation_independent_proba':0.05,
                        'tournament_size':3,
                        'n_gen_no_change':10,
                        'caching':True
                        }
    
    config.input = set_config(input_excel_path = os.path.join('Dataset','dataset2','TWF' ,'TWF_Label.xlsx'),
                                            input_csv_path = os.path.join('Dataset', 'dataset2','TWF','*', '*.csv'),
                                            target_List = {'Normal':0, 'Rubbing':1, 'Unbalance':2, 'Misalignment':3} ,
                                            threshold_percent = 0.95,
                                            overlap_rate=0.5,
                                            segment_length=0.06,
                                            SR=6145,
                                            T=0.34,
                                            Maxfreq=1500,
                                            totol_length=2.67,
                                            test_size=0.33,
                                            model_path= 'model.pkl')
    
    
    config.GA = set_config(ga_default_dict)
    
    config.SVM =  set_config(svm_rand_dict)
    
    config.PCA= set_config(selfeature=3,
                                        svd_solver='full')
    
    with open(filename, 'w') as config_file:
        json.dump(config, 
                        config_file, 
                        ensure_ascii=False,
                        default=lambda o: vars(o)[AutoML_Config.ODICT], 
                        indent=4)
        print(f"Success json data saved. (filename: {filename})")


def loadConfig(path, encoding='utf-8'):
    loadPath = os.path.join(os.getcwd(), path)
    if os.path.isfile(loadPath):
        with open(loadPath, 'r', encoding=encoding) as f:
            data = json.load(f)
        return data
    else:
        print(f'Cannot Load File: {path}')
        return None
    
def create_test_config_file(filename):
    # svm_source = set_svm_config()
    config = AutoML_Config()
    config.input = set_config(SamplingRate=6145,
                                                            T=0.34,
                                                            overlap_rate=0.5,
                                                            threshold_percent = 0.95,
                                                            segment_length=0.06,
                                                            Maxfreq=1500,
                                                            totol_length=2.67,
                                                            test_size=0.33,
                                                            preprocessing=True,
                                                            n_splits=1,
                                                            n_iter=1,
                                                            target_List = {'Normal':0, 'Rubbing':1, 'Unbalance':2, 'Misalignment':3},
                                                            search='random',
                                                            model_path= 'model.pkl')
    
    with open(filename, 'w') as config_file:
        json.dump(config, 
                        config_file, 
                        ensure_ascii=False,
                        default=lambda o: vars(o)[AutoML_Config.ODICT], 
                        indent=4)
        print(f"Success json data saved. (filename: {filename})")
        
if __name__ == '__main__':
    FILE_NAME = "config.json"
    # create_train_config_file(FILE_NAME)
    create_test_config_file(FILE_NAME)
    # loadConfig(path='test.json')