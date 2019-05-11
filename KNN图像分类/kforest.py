"""
KNN 随机森林
0.随机k
1.随机数据集
2.随机距离度量
3.权重
"""

from classify import  Classify
from load import Load
import numpy as np

"""
随机数据集。
"""
load=Load()
def make_forest(n=10):

    forest=[]
    for i in range(n):
        train = load.load_train(2000)
        train_labels = np.argmax(train[1], axis=1)
        classify = Classify(6, train[0], train_labels)
        forest.append(classify)

    return forest


"""
预测样本x
"""
def predict(forest,x):
    predict_=np.array([])
    for KNN_tree in forest:
        #获取每个knn的预测
        clazz,confidence=KNN_tree.predit(x)
        # if confidence>0.6:
        predict_=np.r_[predict_,clazz]

    # print(predict_.shape[0])
    result=[(np.sum(predict_==i),i) for i in set(predict_)]
    result=np.array(result,dtype=[('count',np.int16),('clazz',np.int16)])
    result=np.sort(result,order='count')[::-1]
    out=(result[0][0]/len(forest),result[0][1])
    return out#p,clazz


#测试分类结果
def test():
    forest=make_forest(10)
    test = load.load_test(200)
    test_labels = np.argmax(test[1], axis=1)

    acc=np.array([])
    for x,y in zip(test[0],test_labels):
        p,clazz=predict(forest,x)
        acc=np.r_[acc,clazz==y]
        print(p,'\t',clazz,'\t',clazz==y)

    print('准确率：',np.sum(acc)/acc.shape[0])

if __name__ == "__main__":
    test()
