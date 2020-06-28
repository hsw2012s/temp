#Data Structure Lib
import numpy as np
from collections import OrderedDict
#Data Parser Lib
import os
import glob
#Data Zipped Lib
import gzip
import joblib
import pickle
#Model Pipleline (직렬화)
from sklearn.pipeline import Pipeline
"""Machine Learning Library"""
#Train and Test data split
from sklearn.model_selection import train_test_split
from ga_selection import GASelection
from SVMranking import SVMranking
from SVM_classfy import SVM_classfy
from arranged_case import arranged_case
from feature_normalize import preprocessing_date
#PCA
from generic_pca import gpca
#COPY
import copy
## Config File
import config
# Classfiication Report
from sklearn.metrics import classification_report
#Plot Lib
from plot_draw import plot_draw
#Class str -> number
from sklearn.preprocessing import LabelEncoder
#output json parser
from result_json_parser import AutoML_Config as ac
from result_json_parser import set_config
from datetime import datetime
####################
def save_ML_Model(model, path):
    joblib.dump(model, path) # DIFF
    
def load_ML_Model(path):    
    return joblib.load(path)

if __name__ == '__main__':
    #Input: excel label path
    #Input: label format 
    conf = config.loadConfig('config.json')
    ###############Input에 대한 설정#####################
    #Input Excel Path
    xl_path = conf['input']['input_excel_path']
    #Input: csv path.
    data_csv_path = conf['input']['input_csv_path']
    #target_list_label
    target_list_label = conf['input']['target_List']
    ####################################################
    
    #Preprocessing Variable    
    ov_r = conf['input']['overlap_rate']
    sampling_rate = conf['input']['SR']
    T = conf['input']['T']
    seg_len = conf['input']['segment_length']
    tot_len = conf['input']['totol_length']
    test_size = conf['input']['test_size']
    maxfreq= conf['input']['Maxfreq']
    #History  Variable
    Per= conf['input']['threshold_percent']
    ############################################################
    
    #################Output(Json)에 대한 설정)#######################
    
    #이부분은 DB 연결 후 작성해야됨
    train_json = ac()
    train_json.file_path= set_config(Dbindex=3,             
                                                            AreaID='full',
                                                            EquipmentID=1)
    #데이터 취득일(Database에서 취득일을 얻을 수 있도록 바꿔야됨)
    train_json.Data_Date = datetime.now().strftime('%Y.%m.%d')  
    #진단을 수행한 날짜
    train_json.Test_Date = datetime.now().strftime('%Y.%m.%d')
    
    train_json.Option = ac()
    train_json.Option.Segment = set_config(
                                                                    SamplingRate=sampling_rate, #샘플링 레이트
                                                                    T=T,#샘플링 시간
                                                                    overlap_rate=ov_r#오버랩 비율
                                                                    )

    ################################################################

    
    # features_n, classes, feature_names = preprocessing_date(xl_path, 
    #                                                                 data_csv_path, 
    #                                                                 target_list_label,
    #                                                                 ov_r=ov_r,
    #                                                                 max_freq=maxfreq,
    #                                                                 seg_len=seg_len,
    #                                                                 tot_length=tot_len)
    

    # with gzip.open('pre_features_n_zip.pkl', 'wb') as f:
    #     pickle.dump(features_n, f)
    # with gzip.open('pre_classes_zip.pkl', 'wb') as f:
    #     pickle.dump(classes, f)
    # with gzip.open('feature_names_zip.pkl', 'wb') as f:
    #     pickle.dump(feature_names, f)
    
    
    with gzip.open('pre_features_n_zip.pkl', 'rb') as f:
        features_n =  pickle.load(f)
    with gzip.open('pre_classes_zip.pkl', 'rb') as f:
        classes =  pickle.load(f)
    with gzip.open('feature_names_zip.pkl', 'rb') as f:
        feature_names =  pickle.load(f)
        

    train_json.Option.FeatureExtract = set_config(
                                                                                    Maxfreq=maxfreq, 
                                                                                    Featrue_Name= feature_names #특징이름
                                                                                )
    
    train_json.Option.Standardization = set_config(
                                                                                        Standard_deviation=(np.std(features_n, axis=0)).tolist(), 
                                                                                        #학습데이터 특징열들의 표준편차값
                                                                                        #(Normalization의 경우 min값)
                                                                                        Mean=(np.mean(features_n, axis=0)).tolist()
                                                                                    )
    
    
    #Multi Layer Perceptron
    if False:
        X_train, X_test, y_train, y_test = train_test_split(features_n, 
                                                            classes, 
                                                            test_size=0.2, 
                                                            random_state=42)
        
        print('Xtrain size:{}, X_test  size:{}, y_train size:{}, y_test size:{}'.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        scalar = StandardScaler()
        X_train = scalar.fit_transform(X_train)
        X_test = scalar.fit_transform(X_test)
        mlp = MLPClassifier(alpha=1e-5, hidden_layer_sizes=[256, 256], random_state=0, max_iter=1, warm_start=True)
        for i in range(10):
            mlp.fit(X_train, y_train)
        mlp.max_iter = 100000
        mlp.fit(X_train, y_train)
        print(classification_report(mlp.predict(X_test),
                                            y_test, 
                                            target_names=list(conf['input']['target_List'].keys())))
        print(mlp.get_params())
    # exit()
    estimator = None
    #GA Hyperparameters..
    train_json.GA = set_config(**conf['GA'])
    ##GA set result json Init
    ##학습결과 Json Init
    train_json.Result=ac()
    train_json.Result.FeatureSelection = ac()

    feature_names = np.array(feature_names)
    while True:
        #Generic Algorithm Variable.
        print("Generic Algorithm Hyperparametor as followed")
        for k, v in conf['GA'].items():
            print(f'{k}: {v}')
        #Genetic Algorithm Hyper parameter Setting
        hist, selfeat, shist, ga_estimator = GASelection(
                                                                        feature_n=features_n, 
                                                                        classes=classes, 
                                                                        cv=conf['GA']['cv'],
                                                                        scoring=conf['GA']['scoring'],
                                                                        max_features=conf['GA']['max_features'],
                                                                        n_population=conf['GA']['n_population'],
                                                                        crossover_proba=conf['GA']['crossover_proba'],
                                                                        mutation_proba=conf['GA']['mutation_proba'],
                                                                        n_generations=conf['GA']['n_generations'],
                                                                        crossover_independent_proba=conf['GA']['crossover_independent_proba'],
                                                                        mutation_independent_proba=conf['GA']['mutation_independent_proba'],
                                                                        tournament_size=conf['GA']['tournament_size'],
                                                                        n_gen_no_change=conf['GA']['n_gen_no_change'],
                                                                        caching=conf['GA']['caching']
                                                                    )
        selfeature = features_n[:, selfeat]
        
        
        sel_feature_names = feature_names[selfeat] # 선택된 특징 이름들 array
        print(f"Select Feature: {sel_feature_names}")
        selfeat = selfeat.flatten()

        #Support Vector Machine Hyperparametors.
        print("\nSupport Vector Machine Hyperparametor Range as followed")
        for k, v in conf['SVM'].items():
            print(f"{k}: {v}")
        print()
        ga_svm_acc, ga_svm_pred, ga_svm_confmat, estimator = SVM_classfy(
                                                                                                                features=selfeature, 
                                                                                                                classes=classes.ravel(), 
                                                                                                                estimator=estimator,
                                                                                                                **conf['SVM'])
        
        '''
        #Debugging용으로는 dict를 불러와서 사용함.
        with gzip.open('svm_classfy_zip.pkl', 'rb') as f:
            svm_classfy =  pickle.load(f)
        ga_svm_acc = svm_classfy['best_accuracy']
        predict_result = svm_classfy['predict_result']
        confmat = svm_classfy['confmat']
        '''
        print('SVM done!')
        
        ga_svm_report = classification_report(
                                                        estimator.best_estimator_.predict(selfeature), 
                                                        classes.ravel(), 
                                                        target_names=list(conf['input']['target_List'].keys()),
                                                        output_dict=True
                                                        )
        print(classification_report(
                                                        estimator.best_estimator_.predict(selfeature), 
                                                        classes.ravel(), 
                                                        target_names=list(conf['input']['target_List'].keys())
                                                        ))

        #GA를 통해 나온 Bool형태의 Matrix mapping.
        selfeature = features_n[:, selfeat]
        selfeat = selfeat.flatten()
        #
        print("==================PCA==========================")
        #PCA한번 한다음에 SVM평가
        feature_pca, pca_param, pca_estimator  = gpca(features_n)
        #GA->PCA결과
        train_json.Result.FeatureSelection = set_config(
                                                                        #전체 선택된 특징 이름(특징_센서 포인트)
                                                                        total_features_name=feature_names.tolist(),
                                                                        #특징 선택된 횟수
                                                                        hist = hist.tolist(),
                                                                        #선택된 특징 위치
                                                                        selfeature = selfeat.tolist(),
                                                                        #선택된 특징 나열
                                                                        shist = shist.tolist(),
                                                                        #전체 선택된 특징값(포인트)
                                                                        total_features=features_n.tolist(),
                                                                        #최종 GA 선택 특징 3개 포인트
                                                                        GA_selection_feature= selfeature.tolist(), 
                                                                        #최종 GA 선택 특징 3개의 이름
                                                                        GA_selection_feature_name= sel_feature_names.tolist(), 
                                                                        #최종 PCA or 선택 특징 3개값(포인트)
                                                                        PCA_or_selection_feature= feature_pca.tolist(),
                                                                        #PCA 파라미터들
                                                                        PCA_parameters = pca_param
                                                                        )   
        print('PCA done!')
        
        print("==================PCA-SVM=====================")
        pca_svm_acc, pca_svm_pred, pca_svm_cfmat, estimator = SVM_classfy(features=feature_pca,
                                                                                                                    classes=classes.ravel(), 
                                                                                                                    estimator=estimator,
                                                                                                                    **conf['SVM'])
        pca_svm_report = classification_report(estimator.best_estimator_.predict(feature_pca), 
                                                classes, 
                                                target_names=list(conf['input']['target_List'].keys()),
                                                output_dict=True)
        print(classification_report(estimator.best_estimator_.predict(feature_pca), 
                                                classes, 
                                                target_names=list(conf['input']['target_List'].keys())))
        print("PCA-SVM done!")
        if ga_svm_acc > pca_svm_acc:
            best_accuracy = ga_svm_acc
            performance = ga_svm_report
            confmat = ga_svm_confmat
            predict_result=ga_svm_pred
            model_pipleine = Pipeline([('GA', ga_estimator), ('SVM', estimator.best_estimator_)])
        else:
            best_accuracy = pca_svm_acc
            performance = pca_svm_report
            confmat = pca_svm_cfmat
            predict_result= pca_svm_pred
            model_pipleine = Pipeline([('PCA', pca_estimator), ('SVM', estimator.best_estimator_)])
        if Per <= best_accuracy:
            print(f'Current Best Accuracy: {best_accuracy}')
            ##SVM 설정값 저장
            train_json.Result.SVMstruct=set_config(**estimator.best_estimator_.get_params())
            #최종 분류 성능
            #train_json.Result.Performance = best_accuracy
            train_json.Result.Performance = performance
            train_json.Result.Confmat = confmat.tolist()
            #번호로 Label찾기위해 key-value값 교환
            tmp = {v:k for k, v in target_list_label.items()}
            train_json.Result.Label = [tmp[i] for i in predict_result] #학습시 사용된 Label순서
            break
        
    ''''요청작업'''
    le = LabelEncoder().fit(classes)
    real_label = le.transform(classes)
    pred_label = le.transform(predict_result)
    
    plot_draw(x=selfeature[:,0], 
                    y=selfeature[:,1], 
                    z=selfeature[:,2],
                    xtitle=sel_feature_names[0],
                    ytitle=sel_feature_names[1],
                    ztitle=sel_feature_names[2],
                    labels=real_label,
                    pred_labels=pred_label)

    #Model Save
    save_ML_Model(model_pipleine, path=conf['input']['model_path'])
    #PostGre SQL 넣어주면 됨.
    import json
    filename= 'train_output.json'
    with open(filename, 'w') as config_file:
        json.dump(train_json, 
                        config_file, 
                        ensure_ascii=False,
                        default=lambda o: vars(o)[ac.ODICT], 
                        indent=4)
        print(f"Success json data saved. (filename: {filename})")