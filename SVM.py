from sklearn import svm
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib
from sklearn.svm import SVC

from sklearn import preprocessing
from sklearn.model_selection import  cross_val_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier 


a = []
b = []
knn_s = []
for i in range(1):
#define converts(字典)
    def Image_label(s):
        it={b'N':0, b'P':1}
        return it[s]
    
    
    #1.读取数据集
    path='C:/Users/Mechrevo/Desktop/autumn/SICE2022/NPSVM.txt'
    data=np.loadtxt(path, dtype=float, delimiter=',', converters={66:Image_label} )
    #converters={4:Iris_label}中“4”指的是第5列：将第5列的str转化为label(number)
    
    #2.划分数据与标签
    x,y=np.split(data,indices_or_sections=(66,),axis=1) #x为数据，y为标签,axis是分割的方向，1表示横向，0表示纵向，默认为0
    x=x[:,0:66] #为便于后边画图显示，只选取前两维度。若不用画图，可选取前四列x[:,0:4]
    

    '''
    train_data,test_data,train_label,test_label =sklearn.model_selection.train_test_split(x,
                                                                                          y,
                                                                                          random_state=1,#作用是通过随机数来随机取得一定量得样本作为训练样本和测试样本
                                                                                          train_size=48/49,
                                                                                          test_size=1/49)
    '''
    
    kfolder = KFold(n_splits=49,random_state=1)
    for train, test in kfolder.split(x,y):
        print('Train: %s | test: %s' % (train, test),'\n')
        #print("x_test:")
        #print(y[test])
        #print(len(x))
        #print(len(y))
        print("===============================================================")
    #for X_train_i,X_test_i in kf.split(x):
        #print(x[X_train_i],x[X_test_i])
        
    #print(x)
    #test_data = train_data[i]
    #print("testdata")
    #print(test_data)
    #print(i)
    #print(len(test_data))
    #print(len(train_data))
    
    #train_data:训练样本，test_data：测试样本，train_label：训练样本标签，test_label：测试样本标签
        '''
    
        train_data,test_data,train_label,test_label =sklearn.model_selection.train_test_split(x,
                                                                                          y,
                                                                                          random_state=0,#作用是通过随机数来随机取得一定量得样本作为训练样本和测试样本
                                                                                          train_size=0.6,
                                                                                          test_size=0.4)
        '''
    #3.训练svm分类器
        #classifier=svm.SVC(kernel='linear', C=100).fit(x[train],y[train].ravel()) # ovr:一对多策略
        classifier=svm.SVC(C=0.1,kernel='rbf',gamma=100).fit(x[train],y[train].ravel()) 
        #ravel函数在降维时默认是行序优先
        '''    
    #4.训练lda分类器
        lda = LinearDiscriminantAnalysis()
        lda.fit(x[train],y[train].ravel())
        lda.predict(x[train]) # LDA是有监督方法，需要用到标签
        lda.score(x[test],y[test])
        '''       
    #5.训练knn分类器
        knn = KNeighborsClassifier(n_neighbors=4)
        knn.fit(x[train], y[train].ravel()) 
        iris_y_predict = knn.predict(x[test]) 
        probility=knn.predict_proba(x[test])
        #neighborpoint=knn.kneighbors(x[test],5,False)
        score=knn.score(x[test],y[test],sample_weight=None)
        knn_s.append(score)
        print('knn的预测结果:',knn_s)
        #print('Accuracy:',score)

           
        #4.计算svc分类器的准确率
        #print("svm训练集：",classifier.score(x[train],y[train]))
        #print("测试集：",classifier.score(x[test],y[test]))
         
    # #也可直接调用accuracy_score方法计算准确率
        
        from sklearn.metrics import accuracy_score
        #tra_label=classifier.predict(x[train]) #训练集的预测标签
        tes_label=classifier.predict(x[test]) #测试集的预测标签
        #print("train_rate：", accuracy_score(y[train],tra_label) )
        print("test_rate：", accuracy_score(y[test],tes_label))
        #a.append(accuracy_score(y[train],tra_label))
        b.append(accuracy_score(y[test],tes_label))
        #print("len(b)")
        #print(len(b))
        print('svm的预测结果:',b)
        i=i+1
#print("Mean is :{:.3}".format(np.mean(a))) 
print("SVM Mean is :{:.3}".format(np.mean(b)))
print("KNN Mean is :{:.3}".format(np.mean(knn_s)))  

#查看决策函数
# print('train_decision_function:',classifier.decision_function(train_data)) # (90,3)返回的是每一组X所对应的w,b，方程为w0Tx0 + w1Tx1 + b = 0
# print('predict_result:',classifier.predict(train_data))#返回的是训练模型对训练样本的预测分类
'''
# 5.绘制图形
# 确定坐标轴范围
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0维特征的范围
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1维特征的范围
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网络采样点
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
# 指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 设置颜色
cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0'])
cm_dark = matplotlib.colors.ListedColormap(['g', 'r'])

grid_hat = classifier.predict(grid_test)  # 预测分类值
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

# print(x1.flat)
# print(x2)
# print(grid_test.shape)
#
# print(grid_hat)

plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)     # 预测值范围的显示，相当于将预测每一类的区域进行了划分
plt.scatter(x[:, 0], x[:, 1], c=y[:,0], s=30,cmap=cm_dark)  # 训练样本点的显示
plt.scatter(test_data[:,0],test_data[:,1], c=test_label[:,0],s=30,edgecolors='k', zorder=2,cmap=cm_dark) #圈中测试集样本点
plt.xlabel('花萼长度', fontsize=13)
plt.ylabel('花萼宽度', fontsize=13)
plt.xlim(x1_min,x1_max)
plt.ylim(x2_min,x2_max)
plt.title('鸢尾花SVM二特征分类')
plt.show()

'''
# print(train_data)

# print(type(x))

'''
X = []
Y = []
Z = []
M = []#定义列表，分别用于接受不同组合的C，gamma以及性能指标值
for C in range(1,10,1):
    for gamma in range(1,11,1):
        #获得不同组合下的识别率，作为模型优劣评价的性能指标，这里需要注意的是，性能指标roc_auc，在本例中行不通，因为是多类问题，需要另外设置
        #获得的识别率是交叉验证后的平均值
        accuracy = cross_val_score(SVC(C=C/10,kernel='rbf',gamma=gamma/10),x,y.ravel(),cv=5,scoring='accuracy').mean()
        X.append(C/10)
        Y.append(gamma/10)
        Z.append(accuracy)
        M.append((C/10, gamma/10, accuracy))
print(M)
 
#5、以C，gamma，auc作为三个坐标变量绘图
 
#将列表转换成数组
X = np.array(X).reshape(9,10)
Y = np.array(Y).reshape(9,10)
Z = np.array(Z).reshape(9,10)
 
#绘制三维图形
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
# ax.scatter(Y,X,Z,c='r')
plt.xlabel('Gamma')
plt.ylabel('C')
plt.title('gamma_C_auc')
plt.show()
'''