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
#Train and Test data split
from sklearn.model_selection import train_test_split
#COPY
import copy
## Config File
import config
# Classfiication Report
from sklearn.metrics import classification_report
#Plot Lib
from plot_result import plot_result
    
####################

def main():
    with gzip.open('pre_features_n_zip.pkl', 'rb') as f:
        features_n =  pickle.load(f)
    with gzip.open('pre_classes_zip.pkl', 'rb') as f:
        classes =  pickle.load(f)
    
    
        
if __name__ == '__main__':
    main()
    