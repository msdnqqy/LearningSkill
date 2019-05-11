"""
KNN分类
1.距离度量方式
2.k个
"""
import distance
import numpy as np
from load import Load

np.random.seed(10000)
class Classify:
    def __init__(self,k=10,train_data=None,labels=None,distance_function=distance.L2):
        self.k=k
        self.train_data=train_data
        self.labels=labels
        self.distance_function=distance.L2

    #计算样本x,ys之间的距离
    def distance(self,x):
        dis=np.array([])
        for y in self.train_data:
            d=self.distance_function(x,y)
            dis=np.r_[dis,d]
        return dis

    def sort_distance(self,dis):
        a=np.argsort(dis)
        return a

    def get_topk(self,x):
        dis=self.distance(x)
        arg=self.sort_distance(dis)
        arg=arg[0:self.k]
        datas=self.train_data[arg]
        labels=self.labels[arg]
        return datas,labels

    #返回【支持数量，类别】
    def get_class(self,labels):
        l=[(np.sum(labels==i),i) for i in set(labels)]
        l=np.array(l,dtype=[('x',np.int16),('y',np.int16)])
        return np.sort(l,order='x')[::-1]

    def predit(self,x):
        topk=self.get_topk(x)
        c = self.get_class(topk[1])
        confidence=c[0][0]/self.k
        clazz=c[0][1]
        return clazz,confidence


if __name__=='__main__':
    load=Load()
    train,test=load.load_train(2000),load.load_test(200)
    print('获取数据成功：',train[0].shape,train[1].shape)

    #预处理标签
    train_labels,test_labels=np.argmax(train[1],axis=1),np.argmax(test[1],axis=1)
    classify=Classify(6,train[0],train_labels)

    ylabel,y_label=np.array([]),np.array([])
    for test_item,i in zip(test[0],np.arange(0,test[0].shape[0])):
        x=test_item
        # print(x.shape)
        y=test_labels[i]
        # topk=classify.get_topk(x)
        #
        # # print('k:',topk[1])
        # c=classify.get_class(topk[1])

        # print(i,'\t获取分类成功：',c[0],y)


        clazz,confidence=classify.predit(x)
        print('类别：',clazz,'\tconfidence:',confidence,'\t',clazz==y)
        y_label=np.r_[y_label,y]
        ylabel = np.r_[ylabel, clazz]

    print('准确率:',100*(np.sum(ylabel==y_label)/ylabel.shape[0]))

