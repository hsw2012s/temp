from scipy import io
import numpy as np
import copy
import os
import math
import scipy.optimize as optimize
from scipy import special
from scipy.stats import kurtosis 
from scipy.stats import skew
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html
from scipy.stats import moment
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moment.html
from scipy import signal
from collections import OrderedDict
#https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.special.erfc.html
from tqdm import tqdm

def llnorm(par, data):
    n = len(data)
    mu, sigma = par
    ll = -np.sum(-n/2 * math.log(2*math.pi*(sigma**2)) - ((data-mu)**2)/(2 * (sigma**2)))
    return ll


def likelihood(mean, cov): # Wikipedia
    def calc_loglikelihood(residuals):
        return -0.5 * (np.log(np.linalg.det(cov)) + residuals.T.dot(np.linalg.inv(cov)).dot(residuals) + 2 * np.log(2 * np.pi))

    # mean = np.array([y1, y2]), cov = np.array([[c1, 0], [0, c2]])
    residuals = (cov - mean)

    loglikelihood = np.apply_along_axis(calc_loglikelihood, 1, residuals)
    loglikelihoodsum = loglikelihood.sum()

    return loglikelihoodsum

def normlike(mean, std, data):
    censoring = 0
    freq = 1
    mu = mean
    sigma = std
    #Return Nan for out of range parameters.
    if sigma <=0 :
        sigma = np.nan
    
    #Computer the individual log-liklihood terms.
    z = (data - mu) / sigma
    L = -(0.5)*z**2 - np.log(np.sqrt(2*np.pi)*sigma)
    ncen = np.sum(freq*censoring)
    if ncen > 0:
        cens = (censoring ==1)
        zcen = z[cens]
        
        Scen = 0.5*special.erfc(zcen/np.sqrt(2))
        L[cens] = np.log(Scen)
    # Sum up the individual contributions, and return the negative log-likelihood.
    nlogL = -np.sum(freq * L)
    # Compute the negative hessian at the parameter values, and invert to get
    # the observed information matrix.
    return nlogL
    
def compute_histogram_bins(data, desired_bin_size):
    min_val = np.min(data)
    max_val = np.max(data)
    min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
    max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
    n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
    bins = np.linspace(min_boundary, max_boundary, n_bins)
    return bins

def histogram(data):
    
    minx = np.min(data)
    maxx = np.max(data)
    delta = (maxx-minx)/(data.size-1)
    ncell = np.ceil(np.sqrt(data.size))
    lower_bound = minx-delta/2
    upper_bound = maxx+delta/2
    
    result = [0 for i in range(data.size)]
    y = np.round((data-lower_bound)/(upper_bound-lower_bound)*ncell+0.5)
    for i in range(data.size):
        index = y[i]
        if index >= 1 and index <=ncell:
            idx = int(index)
            result[idx] = result[idx]+1
    return result, lower_bound, upper_bound, ncell

def entropy(x, approach='unbiased', base=np.exp(1)):
    h, lower, upper, ncell = histogram(x)
    estimate=0
    sigma=0
    count=0
    for n in range(int(ncell)):
        if h[n] != 0:
            logf = np.log(h[n])
        else:
            logf = 0
        count = count + h[n]
        estimate = estimate - h[n]*logf
        sigma= sigma+h[n]*logf**2
        
    # Biased Estimate
    estimate = estimate / count
    sigma = np.sqrt((sigma/count-estimate**2)/(count-1))
    estimate = estimate + np.log(count)+np.log((upper-lower)/ncell)
    nbias = -(ncell-1)/(2*count)
    # Conversion to unbiased Estimate
    if approach[0] == 'u':
        estimate = estimate - nbias
        nbias = 0
    
    # Conversion to minimum mse estimate
    if approach[1] == 'm':
        estimate = estimate - nbias
        nbias =0
        _lambda = estimate**2 / (estimate**2+sigma**2)
        nbias = (1-_lambda)*estimate
        estimate = _lambda*estimate
        sigma = _lambda*sigma
    
    estimate = estimate/np.log(base)
    nbias = nbias / np.log(base)
    sigma = sigma / np.log(base)
    
    return estimate, nbias, sigma, [lower, upper, ncell]

def split_length(wav, ov_r=0.5, seg_len=0.06, tot_length=0.48):
    """ Parameter
    ov_r: Overlay rate (겹치는 영역) default:0.5
    seg_len: Sample Length(T) 길이별 샘플링되는 시간 default: 0.06sec
    tot_lenth: 데이터길이 default: 0.48 sec
    
    Output
    result: np.array 결과(list)
    """
    result=[]
    
    l_wd = len(wav)
    
    term = int(l_wd/(tot_length/seg_len))
    ori_list = [(i) for i in range(0, l_wd, int(l_wd/(tot_length/seg_len)))] #원래 나눈영역
    ori_list = ori_list[:-1] #하나 짤릴수도 있다고해서 범위 하나 줄였음.
    for i in range(1, len(ori_list)):
        s, e = (ori_list[i-1], ori_list[i])
        result.append(wav[s:e])
    ov_list = [(i) for i in range(int(term*ov_r), l_wd, term)] #중간 겹치는 영역
    for i in range(1, len(ov_list)):
        s, e = (ov_list[i-1], ov_list[i])
        result.append(wav[s:e])
    
    
    return result
        
def feature_extraction(x, maxfre=1500):
    
    results = OrderedDict()
    rms = lambda V, axis=None: np.sqrt(np.mean(np.square(V), axis))
    peak = np.max(x)
    
    # fvector 1:  peak value
    results['Peak value']=peak
    # fvector 2: root-mean-square
    rootms = rms(x)
    results['RMS']=rootms
    # fvector 3: kurtosis
    
    _kurtosis = kurtosis(x, axis=0, fisher=False)
    results['Kurtosis']= _kurtosis
    # fvector 4: crest factor
    results['Crest factor'] = (peak/rootms)
    # fvector 5: clearance factor
    results['Clearance factor'] = (peak/(np.sum(np.sqrt(np.abs(x)))/x.size)**2)
    # fvector 6: impulse factor
    results['Impulse factor']= (peak/(np.sum(np.sqrt(np.abs(x)))/x.size))
    # fvector 7: shape factor
    results['Shape factor'] = (rootms/(np.sum(np.abs(x))/x.size))
    # fvector 8: skewness
    results['Skewness'] = (skew(x))
    # fvector 9: square mean root
    results['SMR'] = (((np.sum(np.sqrt(np.abs(x))))/x.size)**2)
    # fvector 10: 5th normalized moment
    results['5th normalized moment'] =(moment(x, 5)/np.std(x, ddof=1)**5)
    # fvector 11: 6th normalized moment
    results['6th nomalized moment'] = (moment(x, 6)/np.std(x, ddof=1)**6)
    # fvector 12: mean
    results['Mean'] = (np.mean(np.abs(x)))
    # fvector 13: shape factor2
    results['Shape factor2'] = (results['SMR']/(np.sum(np.abs(x))/x.size))
    # fvector 14: peak-to-peak
    results['Peak to peak'] = (np.max(x)-np.min(x))
    # fvector 15: kurtosis factor
    results['Kurtosis factor'] = (_kurtosis/rootms**4)
    # fvector 16: standard deviation
    results['Standard deviation'] = (np.std(x, ddof=1))
    # fvector 17: smoothness
    results['Smoothness'] = (1-(1+np.std(x, ddof=1)**2))
    # fvector 18: uniformity
    if np.mean(x)==0:
        results['Uniformity']=(np.mean(x)-(np.std(x, ddof=1)))
    else:
        results['Uniformity']=(1-(np.std(x, ddof=1))/np.mean(x))
    # fvector 19: Normal negative log-likelihood
    #likelihood(np.mean(x), x)
    #nnll = optimize.minimize(llnorm, [np.mean(x),np.std(x)], args = (x))
    results['Normal negative log-likelihood'] = (normlike(mean=np.mean(x), std=np.std(x, ddof=1), data=x))
    
    ## Entropy function depending on the histogram
    estimate, vbias, sigma, descriptor = (entropy(x))
    
    # fvector 20: entropy estimation value
    results['Entropy estimation value'] = (estimate)
    # fvector 21: entropy estimation error value
    results['Entropy estimation error value'] = (sigma)
    # fvector 22: Histogram upper bound
    results['Histogram upper bound'] = (descriptor[0])
    
    # fvector 23: Histogram lower bound
    results['Histogram lower bound'] = (descriptor[1])
    
    ## Frequency-Domain Fault Features
    ss = np.abs(np.fft.fft(x, axis=0))*2/x.size 
    
    s = ss[0 : maxfre+1]
    
    ff = [i for i in range(maxfre+1)]
    
    # fvector 24: frequency center
    f = np.array(ff).reshape(-1, 1)
    f_dot_s = f*s
    f_square_2_dot_s= (f**2)*s
    results['Frequency center'] = (np.sum(f_dot_s)/np.sum(s))
    # fvector 25: Mean Squre Frequency
    
    results['Mean Square Frequency']= (np.sum(f_square_2_dot_s)/np.sum(s))
    # fvector 26: rms frequency
    results['Rms of frequency'] = np.sqrt(np.sum(f_square_2_dot_s)/np.sum(s))
    # fvector 27: variance frequency
    smooth_square_dot_s = (results['Smoothness']**2)*s
    results['Variance frequency'] = (np.sum(f - (smooth_square_dot_s)/np.sum(s)))
    # fvector 28: root variance frequency
    results['Root variance frequency'] = (np.sqrt(np.sum(f - (smooth_square_dot_s)))/np.sum(s))
    # fvector 29: f overall
    results['Spectrum overall'] = (np.mean(ss*np.size(x)/2))
    # fvector 30: f rms overall
    results['Spectrum rms overall'] = rms(ss*np.size(x)/2)
    # return [*results.values()], [*results.keys()]
    return results

def Save_to_DB():
    features = []
    classes = []



    #DB에 저장될 데이터 정보입력
    #최초 DB.mat 생성시 주석처리 후 사용.
    #db_mat_dict = io.loadmat('Data Base\DB3.mat')

    #DB에 저장할 데이터 입력 (List)
    file_name = ['GF_M_B']
    fault_type = 7

    # Matrix File 개수 입력
    n_files = 60 
    # 주파수영역 특징 범위 입력
    max_frequency = 3000

    #classes = db_mat_dict['classes']

    _class = np.ones((n_files*len(file_name), 1))*fault_type

    classes = np.concatenate((classes, _class), axis=None)


    features=[]

    for fn in file_name:
        for k in tqdm(range(60)):
            fvector=[]
            for j in range(1, 4):
                path = os.path.join('data', fn, 'channel{}'.format(j), str(k+1))
                load_path =io.loadmat(path)
                segsig = load_path['segsig']            
                feature_list, feature_name = feature_extraction(x=segsig)
                
                fvector.append(feature_list)
            features.append(fvector)
    fea = np.array(features).reshape(60, -1)
    fea_n = np.array(feature_name, dtype=np.object).reshape(1, -1)


    save_to_path = os.path.join('Data Base', 'DB11.mat')
    io.savemat(save_to_path, mdict={'classes': classes, 'features':fea, 'features_name':fea_n}, oned_as='column')

if __name__ == '__main__':
    Save_to_DB()
    