#coding:utf8
"""
使用sklearn实现的决策树
1.数据集：红酒数据集
"""

import sklearn
from sklearn.datasets import load_wine#红酒数据集
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier#引入决策树模型
import sklearn.tree as tree
import numpy as np
import pandas as pd
import graphviz
import matplotlib.pyplot as plt

wine_dataset=load_wine()
wine_data,wine_labels=wine_dataset.data,wine_dataset.target

#切分为训练集和测试集
train_data,test_data,train_labels,test_labels=train_test_split(wine_data,wine_labels,test_size=0.3,random_state=1000)

print("加载红酒数据集完成：train_data.shape:",train_data.shape,'\ttrain_labels.shape',train_labels.shape)

tree_model=DecisionTreeClassifier()#默认为gini系数
tree_model.fit(train_data,train_labels)
print("决策树构建完成：",tree_model)

test_predict0=tree_model.predict([test_data[0]])
print('test_predict0:',test_predict0,test_predict0==test_labels[0])


#得到决策树的准确率
acc=tree_model.score(test_data,test_labels)
print('决策树准确率为:',acc)

#获取每个属性的重要程度
importances_=tree_model.feature_importances_
df=pd.DataFrame(columns=np.arange(test_data[0].shape[0]))
df.loc[0]=importances_

print("各个属性的重要程度",df)

#保存决策树
with open("tree_model.dot",'w') as f:
    f=tree.export_graphviz(tree_model
                                # ,feature_names=  ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']
                                # ,class_names=["琴酒","雪莉","贝尔摩德"]
                                ,feature_names=  wine_dataset.feature_names
                                ,class_names=wine_dataset.target_names
                                ,filled=True,rounded=True,
                                out_file=f)

    print('决策树保存成功，tree_model.dot')


dot_data = tree.export_graphviz(tree_model
                                # ,feature_names=  ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']
                                # ,class_names=["琴酒","雪莉","贝尔摩德"]
                                ,feature_names=  wine_dataset.feature_names
                                ,class_names=wine_dataset.target_names
                                ,filled=True,rounded=True)
graph = graphviz.Source(dot_data)#画树
# graph.render(view=True)
graph.view()