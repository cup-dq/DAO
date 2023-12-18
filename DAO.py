import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, cohen_kappa_score
from sklearn import metrics
from imblearn.over_sampling import SMOTE,ADASYN,RandomOverSampler
from collections import Counter
from smote import read_csv
from sklearn.model_selection import RepeatedKFold
from scipy.spatial.distance import pdist
from scipy import linalg
import matplotlib.pyplot as plt
from mashi import mashi
# 计时
t1 = time.time()

# 读取数据集
x = read_csv()
column = x.columns.tolist()
y = x.pop('class')

a1,b1,c1 = mashi(read_csv())
c1.to_csv('原始数据.csv',index=0)

# 评价指标
G_mean = []
F1 = []
P = []
Recall = []
Kappa = []

# CART
clf1 = DecisionTreeClassifier(criterion='gini')
# ID3
clf2 = DecisionTreeClassifier(criterion='entropy')
# KNN
clf3 = KNeighborsClassifier(n_neighbors=5)
# Bayes
clf4 = GaussianNB()
# SVM
clf5 = SVC(kernel='rbf', gamma='auto')

# 进行10折交叉验证
rkf = RepeatedKFold(n_splits=10, n_repeats=1)
for ltrain, ltest in rkf.split(x, y):
    x_train = x.iloc[ltrain]
    y_train = y.iloc[ltrain]
    x_test = x.iloc[ltest]
    y_test = y.iloc[ltest]

    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # 拼接训练集
    train_data = pd.concat([x_train, y_train],axis=1)

    # #取出少数类数据集
    def Minority_class(data):
        ls = data.iloc[:,-1].tolist()
        num_1 = ls.count(1)
        num_2 = ls.count(2)
        if num_1 > num_2:
            label = 2
        else:
            label = 1
        Minority_data = data[data['class']==label]
        return num_1, num_2, label, Minority_data

    num1, num2, min_label, x_data = Minority_class(train_data)
    x_data = x_data.reset_index(drop=True)
    y_data = x_data.pop('class')
    data = x_data
    data_2 = pd.DataFrame(y_data)
    row = data.shape[0]
    data_size = len(data)
    ##  求马氏距离
    # 指标平均值位置
    data.loc[data_size] = data.mean()
    # 用来存储距离矩阵
    dis = np.zeros(data_size)
    # 求逆矩阵
    data_inv = np.linalg.pinv(np.cov(data.T))
    # 马氏距离计算
    for i in range(0, data_size):
        dis[i] = pdist(data.loc[[i, data_size]], 'mahalanobis', VI=data_inv).tolist()[0]

    data.drop([data.shape[0] - 1], inplace=True)
    #马氏距离排序

    def MD_sort():
        global data
        data['MD']=dis
        data = data.sort_values(by='MD', ascending=True)
        data = data.reset_index(drop=True)
        return data

    #随机生成少数类点
    def rich_data(data):
        rich_ls = []
        row,col = data.shape
        for i in range(abs(num1 - num2)):
            # 随机挑选一个点
            ran = random.randint(2, row-3)
            point_1 = np.array(data.iloc[ran, :])


            ls = [-2,-1,1,2]

            chazhi1 = np.array(data.iloc[ran + ls[0], -1]) - np.array(data.iloc[ran, -1])
            chazhi2 = np.array(data.iloc[ran + ls[1], -1]) - np.array(data.iloc[ran, -1])
            chazhi3 = np.array(data.iloc[ran + ls[2], -1]) - np.array(data.iloc[ran, -1])
            chazhi4 = np.array(data.iloc[ran + ls[3], -1]) - np.array(data.iloc[ran, -1])
            ran_index = np.argmin([abs(chazhi1), abs(chazhi2), abs(chazhi3), abs(chazhi4)])
            point_2 = np.array(data.iloc[ran + ls[ran_index], :])

            # 马氏距离大
            if ls[ran_index] > 0:
                new_point = point_1 + (random.random() * abs(point_2 - point_1))
                if (new_point.tolist()[-1] < a1) and (new_point.tolist()[-1] > b1):
                    rich_ls.append(new_point.tolist()[:-1])

            # 马氏距离小
            else:
                new_point = point_1 - (random.random() * abs(point_2 - point_1))
                if (new_point.tolist()[-1] < a1) and (new_point.tolist()[-1] > b1):
                    rich_ls.append(new_point.tolist()[:-1])

        rich_ls = pd.DataFrame(rich_ls)
        rich_ls['class'] = min_label
        # 重新定义索引
        rich_ls.columns = column
        # 返回新生成的数据
        return rich_ls

    # 整合总数据
    def total(data):
        total_data = pd.concat([train_data,data])
        total_data = total_data.reset_index(drop=True)
        return total_data

    new_x = total(rich_data(MD_sort()))
    new_y = new_x.pop('class')
    print(new_x.shape)



    # # 分类
    ls = [clf1, clf2, clf3, clf4, clf5]
    for clf in ls:
        # 拟合
        clf = clf.fit(new_x, new_y)
        # 预测
        y_predict = clf.predict(x_test)
        # 预测结果与测试集结果作比对
        y_predict = y_predict.tolist()
        y_test = list(y_test)
        # G-mean
        g_means = geometric_mean_score(y_test, y_predict, average='binary')
        G_mean.append(round(g_means, 4))
        # precision值
        p = precision_score(y_test, y_predict)
        P.append(round(p, 4))
        # f1值
        f1 = metrics.f1_score(y_test, y_predict, average='binary')
        F1.append(round(f1, 4))
        # Recall
        r = recall_score(y_test, y_predict)
        Recall.append(round(r, 4))
        # Kappa
        k = cohen_kappa_score(y_test, y_predict)
        Kappa.append(round(k, 4))

len = len(G_mean)
# CART
cart_index = np.arange(0, len, 5)
# ID3
id3_index = np.arange(1, len, 5)
# KNN
knn_index = np.arange(2, len, 5)
# Bayes
bayes_index = np.arange(3, len, 5)
# SVM
svm_index = np.arange(4, len, 5)


# 统计各分类器的指标
def base_classifier(base_index, zhibiao):
    lst = []
    for c in base_index:
        lst.append(zhibiao[c])
    return lst


cart_gmean = base_classifier(cart_index, G_mean)
cart_f1 = base_classifier(cart_index, F1)
cart_p = base_classifier(cart_index, P)
cart_recall = base_classifier(cart_index, Recall)
cart_kappa = base_classifier(cart_index, Kappa)

id3_gmean = base_classifier(id3_index, G_mean)
id3_f1 = base_classifier(id3_index, F1)
id3_p = base_classifier(id3_index, P)
id3_recall = base_classifier(id3_index, Recall)
id3_kappa = base_classifier(id3_index, Kappa)

knn_gmean = base_classifier(knn_index, G_mean)
knn_f1 = base_classifier(knn_index, F1)
knn_p = base_classifier(knn_index, P)
knn_recall = base_classifier(knn_index, Recall)
knn_kappa = base_classifier(knn_index, Kappa)

bayes_gmean = base_classifier(bayes_index, G_mean)
bayes_f1 = base_classifier(bayes_index, F1)
bayes_p = base_classifier(bayes_index, P)
bayes_recall = base_classifier(bayes_index, Recall)
bayes_kappa = base_classifier(bayes_index, Kappa)

svm_gmean = base_classifier(svm_index, G_mean)
svm_f1 = base_classifier(svm_index, F1)
svm_p = base_classifier(svm_index, P)
svm_recall = base_classifier(svm_index, Recall)
svm_kappa = base_classifier(svm_index, Kappa)

def algorithm1():
    print("******************************************")
    print("***********''''''''算法1''''''''***********")
    print('Gmean-CART', cart_gmean, '平均值:', np.mean(cart_gmean), '\nGmean-ID3', id3_gmean, '平均值:', np.mean(id3_gmean),
          '\nGmean-KNN', knn_gmean, '平均值:', np.mean(knn_gmean), '\nGmean-Bayes', bayes_gmean, '平均值:',
          np.mean(bayes_gmean), '\nGmean-SVM', svm_gmean, '平均值:', np.mean(svm_gmean))
    print('F1-CART', cart_f1, '平均值:', np.mean(cart_f1), '\nF1-ID3', id3_f1, '平均值:', np.mean(id3_f1), '\nF1-KNN', knn_f1,
          '平均值:', np.mean(knn_f1), '\nF1-Bayes', bayes_f1, '平均值:', np.mean(bayes_f1), '\nF1-SVM', svm_f1, '平均值:',
          np.mean(svm_f1))
    print('Precision-CART', cart_p, '平均值:', np.mean(cart_p), '\nPrecision-ID3', id3_p, '平均值:', np.mean(id3_p),
          '\nPrecision-KNN', knn_p, '平均值:', np.mean(knn_p), '\nPrecision-Bayes', bayes_p, '平均值:', np.mean(bayes_p),
          '\nPrecision-SVM', svm_p, '平均值:', np.mean(svm_p))
    print('Recall-CART', cart_recall, '平均值:', np.mean(cart_recall), '\nRecall-ID3', id3_recall, '平均值:',
          np.mean(id3_recall), '\nRecall-KNN', knn_recall, '平均值:', np.mean(knn_recall), '\nRecall-Bayes', bayes_recall,
          '平均值:', np.mean(bayes_recall), '\nRecall-SVM', svm_recall, '平均值:', np.mean(svm_recall))
    print('Kappa-CART', cart_kappa, '平均值:', np.mean(cart_kappa), '\nKappa-ID3', id3_kappa, '平均值:', np.mean(id3_kappa),
          '\nKappa-KNN', knn_kappa, '平均值:', np.mean(knn_kappa), '\nKappa-Bayes', bayes_kappa, '平均值:',
          np.mean(bayes_kappa), '\nKappa-SVM', svm_kappa, '平均值:', np.mean(svm_kappa))
    t2 = time.time()
    print("算法1运行时间为", t2 - t1)
    print('')
    with open("douban.txt", "a") as f:

        f.write('算法1')
        f.write('\r')

        f.write('CART')
        f.write('\r')
        f.write('Gmean-CART:')
        f.write(str(cart_gmean))
        f.write(' 平均值:')
        f.write(str(np.mean(cart_gmean)))
        f.write('\r')

        f.write('F1-CART:')
        f.write(str(cart_f1))
        f.write(' 平均值:')
        f.write(str(np.mean(cart_f1)))
        f.write('\r')
        f.write('Precision-CART:')
        f.write(str(cart_p))
        f.write(' 平均值:')
        f.write(str(np.mean(cart_p)))
        f.write('\r')
        f.write('Recall-CART:')
        f.write(str(cart_recall))
        f.write(' 平均值:')
        f.write(str(np.mean(cart_recall)))
        f.write('\r')
        f.write('Kappa-CART:')
        f.write(str(cart_kappa))
        f.write(' 平均值:')
        f.write(str(np.mean(cart_kappa)))
        f.write('\r\n')

        f.write('ID3')
        f.write('\r')
        f.write('Gmean-ID3:')
        f.write(str(id3_gmean))
        f.write(' 平均值:')
        f.write(str(np.mean(id3_gmean)))
        f.write('\r')

        f.write('F1-ID3:')
        f.write(str(id3_f1))
        f.write(' 平均值:')
        f.write(str(np.mean(id3_f1)))
        f.write('\r')

        f.write('Precision-ID3:')
        f.write(str(id3_p))
        f.write(' 平均值:')
        f.write(str(np.mean(id3_p)))
        f.write('\r')

        f.write('Recall-ID3:')
        f.write(str(id3_recall))
        f.write(' 平均值:')
        f.write(str(np.mean(id3_recall)))
        f.write('\r')

        f.write('Kappa-ID3:')
        f.write(str(id3_kappa))
        f.write(' 平均值:')
        f.write(str(np.mean(id3_kappa)))
        f.write('\r\n')

        f.write('KNN')
        f.write('\r')

        f.write('Gmean-KNN:')
        f.write(str(knn_gmean))
        f.write(' 平均值:')
        f.write(str(np.mean(knn_gmean)))
        f.write('\r')

        f.write('F1-KNN:')
        f.write(str(knn_f1))
        f.write(' 平均值:')
        f.write(str(np.mean(knn_f1)))
        f.write('\r')

        f.write('Precision-KNN:')
        f.write(str(knn_p))
        f.write(' 平均值:')
        f.write(str(np.mean(knn_p)))
        f.write('\r')

        f.write('Recall-KNN:')
        f.write(str(knn_recall))
        f.write(' 平均值:')
        f.write(str(np.mean(knn_recall)))
        f.write('\r')

        f.write('Kappa-KNN:')
        f.write(str(knn_kappa))
        f.write(' 平均值:')
        f.write(str(np.mean(knn_kappa)))
        f.write('\r\n')

        f.write('Bayes')
        f.write('\r')

        f.write('Gmean-Bayes:')
        f.write(str(bayes_gmean))
        f.write(' 平均值:')
        f.write(str(np.mean(bayes_gmean)))
        f.write('\r')

        f.write('F1-Bayes:')
        f.write(str(bayes_f1))
        f.write(' 平均值:')
        f.write(str(np.mean(bayes_f1)))
        f.write('\r')

        f.write('Precision-Bayes:')
        f.write(str(bayes_p))
        f.write(' 平均值:')
        f.write(str(np.mean(bayes_p)))
        f.write('\r')

        f.write('Recall-Bayes:')
        f.write(str(bayes_recall))
        f.write(' 平均值:')
        f.write(str(np.mean(bayes_recall)))
        f.write('\r')

        f.write('Kappa-Bayes:')
        f.write(str(bayes_kappa))
        f.write(' 平均值:')
        f.write(str(np.mean(bayes_kappa)))
        f.write('\r\n')

        f.write('SVM')
        f.write('\r')
        f.write('Gmean-SVM:')
        f.write(str(svm_gmean))
        f.write(' 平均值:')
        f.write(str(np.mean(svm_gmean)))
        f.write('\r')

        f.write('F1-SVM:')
        f.write(str(svm_f1))
        f.write(' 平均值:')
        f.write(str(np.mean(svm_f1)))
        f.write('\r')

        f.write('Precision-SVM:')
        f.write(str(svm_p))
        f.write(' 平均值:')
        f.write(str(np.mean(svm_p)))
        f.write('\r')

        f.write('Recall-SVM:')
        f.write(str(svm_recall))
        f.write(' 平均值:')
        f.write(str(np.mean(svm_recall)))
        f.write('\r')

        f.write('Kappa-SVM:')
        f.write(str(svm_kappa))
        f.write(' 平均值:')
        f.write(str(np.mean(svm_kappa)))
        f.write('\r\n')

        f.write("算法1运行时间为:")
        f.write(str(t2 - t1))
        f.write('\r\n')


algorithm1()


new_data = pd.concat([new_x, new_y],axis=1)
new_data.to_csv('算法数据.csv',index=0)






