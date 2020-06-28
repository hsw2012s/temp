import numpy as np
from scipy import io

def arranged_case(feature_input,class_input,rank,R):
    c1=np.where(class_input==rank[R-1,0])
    c2=np.where(class_input==rank[R-1,1])
    c1=c1[0]
    c2=c2[0]
    arr=np.append(c1,c2)

    feature_output=np.array([feature_input[arr[0],:]])
    class_output=np.array([class_input[arr[0],:]])
    c=np.arange(np.size(c1)+np.size(c2))
    c=np.array([c])
    c=np.reshape(c,(-1,1))

    for i in range(np.size(arr)):
        temp=feature_input[arr[i],:]
        feature_output=np.vstack([feature_output,temp])
        temp=class_input[arr[i],:]
        class_output=np.vstack([class_output,temp])

    feature_output = np.delete(feature_output, 0, axis=0)
    c[0:np.size(c1),0]=1
    c[np.size(c1):np.size(c1)+np.size(c2),0] = 2

    return feature_output,class_output,c

if __name__== '__main__':
    features_n=io.loadmat('features_n.mat')
    features_n=features_n['features_n']
    classes=io.loadmat('classes.mat')
    classes=classes['classes']
    Rank=io.loadmat('Rank.mat')
    Rank=Rank['Rank']

    feature_re,class_re,c=arranged_case(features_n,classes,Rank,1)
    print(feature_re)
    print(class_re)
    print(c)


