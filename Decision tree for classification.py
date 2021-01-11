import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#dataset=pd.read_csv('zoo.csv',
#names=['animal_name','hair','feathers','eggs','milk',
#                                                   'airbone','aquatic','predator','toothed','backbone',
#                                                   'breathes','venomous','fins','legs','tail','domestic','catsize','class'])
#
#dataset=dataset.iloc[:,1:]


dataset=pd.read_csv('c_d.csv')
#from sklearn.preprocessing import *
#
#lb=LabelEncoder()
#for i in range(dataset.shape[1]):
#    dataset.iloc[:,i]=lb.fit_transform(dataset.iloc[:,i])

def entropy(data,target_name):
    elements,counts=np.unique(data[target_name],return_counts=True)
    h=np.sum([(counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return h*-1


def info_gain(data,split_attribute,target_name):
    H_data=entropy(data,target_name)
    elements,counts=np.unique(data[split_attribute],return_counts=True)
    H_f=np.sum([(counts[i]/np.sum(counts))*entropy(data.where(elements[i]==data[split_attribute]).dropna(),target_name)for i in range(len(elements))])
    
    return H_data-H_f


def classification(data,original_data,features,target_name,parent_node=None):
    
    if len(np.unique(data[target_name]))<=1:
        return np.unique(data[target_name])[0]
    elif len(data)==0:
        return np.unique(original_data[target_name])[np.argmax(np.unique(original_data[target_name],return_counts=True)[1])]
    elif len(features)==0:
        return parent_node
    else:
        parent_node=np.unique(data[target_name])[np.argmax(np.unique(data[target_name],return_counts=True)[1])]
        best_feature_index=np.argmax([info_gain(data,feature,target_name) for feature in features])
        best_feature=features[best_feature_index]
        tree={best_feature:{}}
        features=[feature for feature in features if feature!=best_feature]
        for val in np.unique(data[best_feature]):
            sub_data=data.where(data[best_feature]==val).dropna()
            sub_tree=classification(sub_data,original_data,features,target_name,parent_node)
            tree[best_feature][val]=sub_tree
        return tree
            
        

    
def predict(tree,query,default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                res=tree[key][query[key]]
            except:
                return default
            res=tree[key][query[key]]
            if isinstance(res,dict):
                return predict(res,query,default)
            else:
                return res
        
        
from sklearn.model_selection import *
#
train,test=train_test_split(dataset,test_size=0.5,random_state=0)
#train=dataset.iloc[:80].reset_index(drop=True)
#test=dataset.iloc[80:].reset_index(drop=True)

tree=classification(train,dataset,train.columns[:-1],train.columns[-1])
    
queries=test.iloc[:,:-1].to_dict(orient='records')

y_predict=[]
for q in queries:
    y_predict.append(predict(tree,q))
    
    
accuracy=(np.sum(np.array(y_predict)==np.array(test[test.columns[-1]]))/len(test))*100

print(accuracy)



    
    
    
    