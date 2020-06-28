#Data Structure Lib
import numpy as np
from collections import OrderedDict
import json
#Data Parser Lib
import os
import glob
from os import listdir
from os.path import isfile, join
#Data Zipped Lib
import gzip
import joblib
import pickle
"""Machine Learning Library"""
    # filename= 'test_output.json'
#Train and Test data split
from sklearn.model_selection import train_test_split
from ga_selection import GASelection
from SVMranking import SVMranking
from SVM_classfy import SVM_classfy
from arranged_case import arranged_case
from feature_normalize import preprocessing_raw_data
#PCA
from generic_pca import gpca
from numpy import linalg as LA
#COPY
import copy
## Config File
import config
# Classfiication Report
from sklearn.metrics import classification_report
#Plot Lib
from plot_draw import plot_draw
import matplotlib.pyplot as plt
#Sys Lib
import time
from dateutil.parser import parse as dparse
#output json parser
from result_json_parser import AutoML_Config as ac
from result_json_parser import set_config
from datetime import datetime
import struct
#Preprocessing
from sklearn.preprocessing import StandardScaler
####################
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import svm
import time
# SUOD
from suod.models.base import SUOD
from suod.models.parallel_processes import _parallel_fit
from suod.models.parallel_processes import _parallel_predict
from suod.models.parallel_processes import _parallel_decision_function
from suod.models.parallel_processes import _partition_estimators
from suod.utils.utility import _unfold_parallel
from suod.utils.utility import get_estimators
import scipy as sp
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
from combo.models.score_comb import maximization, average, aom, moa
from suod.utils.utility import _unfold_parallel
from suod.utils.utility import get_estimators
from suod.models.base import SUOD
from scipy.special import erf
from sklearn.preprocessing import MinMaxScaler
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.abod import ABOD
from pyod.models.mcd import MCD
from combo.models.score_comb import majority_vote, maximization, average
start = time.time()

def load_ML_Model(path):
    """[학습될 모델을 꺼내오는 코드]]
    Arguments:
        path {[path]} -- [모델을 불러오는 코드임]
    Returns:
        [byte] -- [저장했던 모델을 반환]
    """
    if os.path.exists(path):
        return joblib.load(path)
    else:
        raise EOFError

def loadConfig(path, readByte=4):
    """[바이트단위로 읽어들이는 코드]
    Arguments:
        path {Str} -- [경로]
    Keyword Arguments:
        readByte {int} -- [읽을 단위] (default: {4})
    Returns:
        list -- [바이트단위로 쪼갠 test 데이터]]
    """
    loadPath = os.path.join(os.getcwd(), path)
    if os.path.isfile(loadPath):
        with open(loadPath, 'rb') as f:
            lines = bytearray(f.read())
        return [struct.unpack('f', lines[(idx)*readByte:(1+idx)*readByte])[0]\
                        for idx, i in enumerate(lines[::readByte])]
            
        # return [struct.unpack('d', i) for i in lines]
    else:
        print(f'Cannot Load File: {path}')
        raise OSError 


if __name__ == '__main__':
    #postgre sql에서 다운받았다고 가정
    
    #print(conf.keys())
    conf = config.loadConfig('test_config.json')
    print(conf['input'])
    
    X = np.empty((0,3), dtype=np.float32)
    result_arr = []
    X_num = 0
    #test 데이터를 postgre sql로 받아왔다고 가정
    json_files = [f for f in listdir('./json') if f.endswith(".json")]
    json_files.sort()
    print(f"find {len(json_files)} json files")
    
    for i in range(len(json_files)):
        fname = json_files[i]
        
        test_input = config.loadConfig('./json/'+fname, encoding='utf-8-sig')
        print("Json Data as followed")
        for k, v in test_input.items():
            print(k, v)
        test_json = ac()
        ############
        """
            파일경로, DBindex, AreaID, EquipmentID
        """
        test_json.file_path= set_config(Dbindex=test_input['dbindex'],             
                                                                AreaID=test_input['AreaID'],
                                                                EquipmentID=test_input['EquipmentID'])
        #############from dateutil.parser import parse
        """
            진단을 수행한 날짜
        """
        test_json.Test_Date = datetime.now().strftime('%Y.%m.%d')
        #############
        """
            데이터 취득일
        """
        tr_time = datetime.strptime(test_input['mes_datetime'], '%Y-%m-%d %H:%M:%S')
        test_json.Data_Date = tr_time.strftime('%Y.%m.%d')
        #############
        """
            Option
        """
        test_json.Option=ac()
        """
            Option -> Segment -> SamplingRate, T, Overlap rate
        """
        test_json.Option.Segment = set_config(SamplingRate= conf['input']['SamplingRate'],
                                                                        T = conf['input']['T'],
                                                                        Overlap_rate = conf['input']['overlap_rate'])
        
        """
            Binary Data open and Preprocessing Data
        """
        test_data = loadConfig(os.path.join('json', test_input['Spectrum']['Value']),readByte=4)
        features_n, feature_names = preprocessing_raw_data(date=test_data,
                                                                                            ov_r=conf['input']['overlap_rate'],
                                                                                            max_freq=conf['input']['Maxfreq'],
                                                                                            seg_len=conf['input']['segment_length'],
                                                                                            tot_length=conf['input']['totol_length'])
        
        """
            Option -> FeatureExtract -> Maxfreq, FeatureName
        """
        test_json.Option.FeatureExtract = set_config(Maxfreq= conf['input']['Maxfreq'],
                                                                                    Feature_Name= feature_names) #특징이름)
        """
            Option -> Standardization or Normalization
        """
        
        test_json.Option.Standardization = set_config(
                                                                                            Standard_deviation=(np.std(features_n, axis=0)).tolist(), 
                                                                                            #학습데이터 특징열들의 표준편차값
                                                                                            #(Normalization의 경우 min값)
                                                                                            Mean=(np.mean(features_n, axis=0)).tolist()
                                                                                        )
        
        """
            Result
        """
        test_json.Result = ac()

        features_n = StandardScaler().fit_transform(features_n)
        
        features_pca, _, pca_estimator = gpca(features_n)
        feature_inv = pca_estimator.inverse_transform(features_pca)

        temp = features_n - feature_inv
        feature_sum = np.sum(temp, axis=0)
        sum_sort = np.sort(feature_sum)
     
        pca1 = feature_names[np.where(feature_sum==sum_sort[0])[0][0]]
        pca2 = feature_names[np.where(feature_sum==sum_sort[1])[0][0]]
        pca3 = feature_names[np.where(feature_sum==sum_sort[2])[0][0]]
        
        print(f"selected features :{pca1}, {pca2}, {pca3}")
        
        """
            Result -> Performance
        """
        
        """
            One class Svm
        """
        anomaly_algorithms = [
        ("Robust covariance", EllipticEnvelope()),
        ("One-Class SVM", svm.OneClassSVM(kernel="rbf", gamma=0.001)),
        ("Isolation Forest", IsolationForest(random_state=42)),
        ("Local Outlier Factor", LocalOutlierFactor())]
            
        X = np.append(arr=X, values=features_pca, axis=0)
        X_num = X.shape[0]
        base_estimators = [
            LOF(),
            IForest(),
            OCSVM(kernel="rbf", gamma=0.001)]
        
        model = SUOD(base_estimators=base_estimators, n_jobs=2,  # number of workers(if -1 it use full core)
             rp_flag_global=True,  # global flag for random projection
             bps_flag=True,  # global flag for balanced parallel scheduling
             approx_flag_global=False,  # global flag for model approximation
             contamination=0.2)

        # X_train, X_test = train_test_split(X, test_size=0, random_state=123)\
        model.fit(X)
        model.approximate(X)
        predicted_labels = model.predict(X)

        sum_labels = np.sum(predicted_labels, axis=1)/3
        sum_labels = np.where(sum_labels>=0.5, -1, 1) # -1 abnormal, 1 normal
        result_label = np.average(sum_labels)
        result_label = result_label.tolist()

        # Add outliers
        fig = plt.figure()
        colors = np.array(['r', 'b'])
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:,2], color=colors[(sum_labels + 1) // 2])
        ax.set_xlabel(pca1)
        ax.set_ylabel(pca2)
        ax.set_zlabel(pca3)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)
            
        plt.savefig(f"./result/{fname[:-5]}_fig.png", dpi=300)
        plt.close()
        with open('./json/'+fname, "r", encoding="utf-8-sig") as json_file:
            json_data = json.load(json_file)
            mes_date = json_data['mes_datetime']
            result_arr.append([dparse(mes_date), result_label])
        print(f"=====Complete {i+1} files in {len(json_files)}=====")
        print("")

        json_file.close()
        result_dict = dict()
        result_dict['selelcted_feautures'] = str([pca1, pca2, pca3])
        result_dict['classes'] = sum_labels[-features_n.shape[0]:].tolist()
        if result_label>=0.5:
            result_dict['result'] = 'normal'
        else:
            result_dict['result'] = 'abnormal'

        with open('./result/'+fname[:-5]+'_result.json', "w") as json_file:
            json.dump(result_dict, json_file, indent=4)
    result_arr.sort()
    time_arr = [i[0] for i in result_arr]
    status_arr = [i[1] for i in result_arr]
  
    plt.plot(time_arr, status_arr)
    plt.gcf().autofmt_xdate()
    plt.ylim(0.5, 1)
    plt.savefig(f"./result/result_fig.png")
    plt.close()

