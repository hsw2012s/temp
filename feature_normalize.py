import numpy as np
from feature_engineering import SStandardScaler
from feature_making import feature_extraction, split_length
import pandas as pd
import os
import glob
#Processing Lib
from tqdm import tqdm

def preprocessing_raw_data(date,
                                            ov_r=0.5, 
                                            seg_len=0.06, 
                                            max_freq=1500,
                                            tot_length=0.48):
    """wave데이터를 30개의 특징으로 바꾸는곳

    Arguments:
        date {[list]} -- raw wave format

    Keyword Arguments:
        ov_r {float} -- [overlap rate] (default: {0.5})
        seg_len {float} -- [segment 길이] (default: {0.06})
        max_freq {int} -- [최대 진동] (default: {1500})
        tot_length {float} -- [데이터 길이] (default: {0.48})

    Returns:
        [2d np.array, list] -- [2차원으로 되어있는 데이터개수, 30개특징, 특징이름]
    """
    date = np.array(date)
    
    #wave분할
    wave_list = split_length(date, 
                                    ov_r=ov_r, 
                                    seg_len=seg_len, 
                                    tot_length=tot_length)
    
    # wave to featrue-name list
    fealist_feaName = [feature_extraction(wave, maxfre=max_freq) for wave in wave_list]
    fealist  = []
    feaName  = None
    #mapping
    for f_list in fealist_feaName:
        fealist.append(list(f_list.values()))
        feaName = list(f_list.keys())
    fealist = np.array(fealist)
    
    return fealist, feaName

def preprocessing_date(xl_path, 
                                        data_csv_path, 
                                        target_list_label,
                                        ov_r=0.5, 
                                        seg_len=0.06, 
                                        max_freq=1500,
                                        tot_length=0.48):
    """레이블과 데이터 입력, 클래스 이름이 명시된 경로를 입력으로 받음

    Arguments:
        xl_path {str} -- excel 경로
        data_csv_path {str} -- raw data csv경로(pandas dataframe으로 이루어져야함)
        target_list_label {str} -- label이름이 명시된 파일 경로

    Keyword Arguments:
        ov_r {float} -- overlap rate (default: {0.5})
        seg_len {float} -- segment length (default: {0.06})
        max_freq {int} -- max frequency (default: {1500})
        tot_length {float} -- total length (default: {0.48})

    Returns:
        [np.array, np.array, list] -- 훈련 데이터(x), 훈련데이터-정답(y), 30개 특징이름들
    """
    xl = pd.read_excel(xl_path)
    date = list(xl['date'].apply(lambda x: str(x).replace('-','')[:8])) #dataset1
    # date = list(xl['date'].apply(lambda x: str(x).split('.')[-1])) #dataset2
    label = list(xl['Label'])
    date_label = dict(zip(date, label))

    data_list = [i for i in glob.iglob(data_csv_path) if i.endswith('csv')]
    tot_date = {}
    #데이터 리스트로부터 value를 뽑음
    for idx, i in enumerate(tqdm(data_list)):
        date = i.split(os.sep)[-1][:-4].split('_')[0] #dataset1
        # date = i.split(os.sep)[-1].split('.')[0]#dataset2
        tot_wave = pd.read_csv(i)['value'].to_numpy()
        #wave분할
        wave_list = split_length(tot_wave, 
                                                ov_r=ov_r, 
                                                seg_len=seg_len, 
                                                tot_length=tot_length)
        for wave in wave_list:
            wav_date = feature_extraction(wave, maxfre=max_freq)
            try:
                cls_label = date_label[date]
                tot_date[cls_label].append(wav_date)
            except:
                tot_date[cls_label] = [wav_date]
    #예시 출력
    # feature_list = np.array(list(tot_date['Normal'][0].keys()))
    # print("Feature List Example:{}".format(feature_list))

    feature_names  = None
    #클래스-순서-(특징이름-특징값) -> 클래스-순서-특징값으로 변환
    for _cls, lis in tot_date.items():
        for idx, f_list in enumerate(lis):
            tot_date[_cls][idx]=list(f_list.values())
            feature_names = list(f_list.keys())
    
    #각 클래스별 개수 출력
    for _cls, lis in tot_date.items():
        print('Class:{}, Num:{}'.format(_cls, len(lis)))

    target_list = [] 
    date_list=[]
    for lis in tot_date.keys():
                #타겟 영어 -> 숫자로 변환
        tar_list = [target_list_label[lis] for i in tot_date[lis]]
        #이름들이 숫자로 변환됨 list가 됨
        target_list.extend(tar_list)
    for _date in tot_date.values():
        #특징이름-값들이 들어있음
        date_list.extend(_date)
        
    #numpy array 형태로 변환
    target_list = np.array(target_list)
    
    
    date_list=np.array(date_list)
    print('TargetSize:{}'.format(len(target_list)))
    print('DataSize:{}'.format(len(date_list)))
    date_list = np.nan_to_num(date_list)
    np.where(date_list[:] >= np.finfo(np.float64).max)
    print(np.where(date_list==np.isnan(date_list)))
    return date_list, target_list, feature_names
