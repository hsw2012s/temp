from collections import OrderedDict
import time
import os
from datetime import datetime
import json
import joblib
class AutoML_Config():
    ODICT = "constructure"
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
        """
            Get pre-defined Genetic Algorithm configuration.
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


def main(filename):
    #time of we run the script
    
    # svm_source = set_svm_config()
    config = AutoML_Config()
    #DataBase 불러와서 해당내용 저장
    config.file_path= config.set_config(Dbindex=3,
                                                                AreaID='full',
                                                                EquipmentID=1)
    #진단을 수행한 날짜
    config.Test_Date = datetime.now().strftime('%Y.%m.%d')
    #데이터 취득일(Database에서 취득일을 얻을 수 있도록 바꿔야됨)
    config.Data_Date = datetime.now().strftime('%Y.%m.%d')
    
    config.Option = AutoML_Config()
    config.Option.Segment = set_config(
                                                                                    SamplingRate=1234, #샘플링 레이트
                                                                                    T=123,#샘플링 시간
                                                                                    Overlap_rate=''#오버랩 비율
                                                                                )
    config.Option.FeatureExtract = set_config(
                                                                                                    Maxfreq=1000, #주파수영역 특징 범위(...?)
                                                                                                    Featrue_Name=['peak_value', 'RMS'] #특징이름
                                                                                                )
    config.Option.Standardization = set_config(
                                                                                                    Standard_deviation=[0.14, 0.21], 
                                                                                                    #학습데이터 특징열들의 표준편차값
                                                                                                    #(Normalization의 경우 min값)
                                                                                                    Mean=[1.2, 1.3]
                                                                                                )
    config.Result = AutoML_Config()
    config.Result.FeatureSelection = set_config(
                                                                                                        #전체 선택된 특징 이름(특징_센서 포인트)
                                                                                                        total_features_name=['RMS_MIH', 'Kurtosis_MOV'],
                                                                                                        #전체 선택된 특징값(포인트)
                                                                                                        total_features=['RMS_MIH', 'Kurtosis_MOV'],
                                                                                                        #최종 PCA or 선택 특징 3개의 이름
                                                                                                        PCA_or_selection_feature_name=['PCA1', 'PCA2', 'PCA3'], 
                                                                                                        #최종 PCA or 선택 특징 3개값(포인트)
                                                                                                        PCA_or_selection_feature=[] 
                                                                                                    )
    config.Result.SVMstruct = config.set_config(
                                                                                            support_vector=[], #Support vectors.
                                                                                            n_support=[], #Number of support vectors for each class.
                                                                                            #이부분은 추후 구현예정
                                                                                        )
    config.Result.Performance = 100
    config.Result.Confmat = [100, 0, 0, 100]
    config.Result.Result = ['Normal']
    
    
    
    with open(filename, 'w') as config_file:
        json.dump(config, 
                        config_file, 
                        ensure_ascii=False,
                        default=lambda o: vars(o)[AutoML_Config.ODICT], 
                        indent=4)
        print(f"Success json data saved. (filename: {filename})")


def loadConfig(path):
    loadPath = os.path.join(os.getcwd(), path)
    if os.path.isfile(loadPath):
        with open(loadPath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    else:
        print(f'Cannot Load File: {path}')
        return None



    pass
if __name__ == '__main__':
    main(filename='test_result.json')
    