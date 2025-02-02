
from mpl_toolkits.mplot3d import Axes3D
import warnings
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from time import time
from minepy import MINE
from sklearn import svm
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import naive_bayes
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from IPython.display import display
from datetime import datetime as dt
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

def data_processing_and_feature_selecting(data_path):
    """
    特征选择
    :param  data_path: 数据集路径
    :return: new_features,label: 经过预处理和特征选择后的特征数据、标签数据
    """
    new_features,label = None, None
    # -------------------------- 实现数据处理和特征选择部分代码 ----------------------------
    #导入医疗数据
    data_xls = pd.ExcelFile(data_path)
    data={}

    #查看数据名称与大小
    for name in data_xls.sheet_names:
            df = data_xls.parse(sheet_name=name,header=None)
            data[name] = df

    #获取 特征1 特征2 类标
    feature1_raw = data['Feature1']
    feature2_raw = data['Feature2']
    label = data['label']


    # 初始化一个 scaler，并将它施加到特征上
    scaler = MinMaxScaler()
    feature1 = pd.DataFrame(scaler.fit_transform(feature1_raw))
    feature2 = pd.DataFrame(scaler.fit_transform(feature2_raw))
    # ------------------------------------------------------------------------
    # 统计特征值和label的皮尔孙相关系数  对两类特征分别进行排序筛选特征
    select_feature_number1 = 25
    select_feature_number2 = 10
    select_feature1 = SelectKBest(lambda X, Y: tuple(map(tuple,np.array(list(map(lambda x:pearsonr(x, Y), X.T))).T)),
                                  k=select_feature_number1
                                 ).fit(feature1, np.array(label).flatten()).get_support(indices=True)

    select_feature2 = SelectKBest(lambda X, Y: tuple(map(tuple,np.array(list(map(lambda x:pearsonr(x, Y), X.T))).T)),
                                  k=select_feature_number2
                                 ).fit(feature2, np.array(label).flatten()).get_support(indices=True)
    
    #相关系数筛选出的特征
    s1_feature1=feature1[feature1.columns.values[select_feature1]]
    s1_feature2=feature2[feature2.columns.values[select_feature2]]

    # -------------------------- PCA降维部分代码 ----------------------------
    # 选择降维维度
    pca1 = PCA(n_components=2)
    pca2 = PCA(n_components=1)
    feature_pca1 = pca1.fit_transform(s1_feature1)
    feature_pca2 = pca2.fit_transform(s1_feature2)
    
    # 获取每个主成分的⽅差变化信息
    variance_explained1 = pca1.explained_variance_ratio_
    variance_explained2 = pca2.explained_variance_ratio_
    
    # 可视化标签中不能出现负值
    pca_label = np.array(label).flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feature_pca1[:, 0], feature_pca1[:, 1], feature_pca2[:, 0],c=pca_label)
    plt.show()
    # 对 feature1 和 feature2 进⾏整合
    feature_pca1_df = pd.DataFrame(feature_pca1)
    feature_pca2_df = pd.DataFrame(feature_pca2)
    
    new_features = pd.concat([feature_pca1_df, feature_pca2_df], axis=1)
    # 返回筛选后的数据
    return new_features,label



# -------------------------- 请加载您最满意的模型 ---------------------------
# 加载模型(请加载你认为的最佳模型)
# 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
# 如果你的模型是在 results 文件夹下的 my_model.m 模型，则 model_path = 'results/my_model.m'
model_path = 'results/my_model.m'

# 加载模型
model = joblib.load(model_path)

# ---------------------------------------------------------------------------

def predict(new_features):
    """
    加载模型和模型预测
    :param  new_features : 测试数据，是 data_processing_and_feature_selecting 函数的返回值之一。
    :return y_predict : 预测结果是标签值。
    """
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    # 获取输入图片的类别
    y_predict = model.predict(new_features)
    # -------------------------------------------------------------------------

    # 返回图片的类别
    return y_predict
