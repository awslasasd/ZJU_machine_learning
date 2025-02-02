# 导入相关库
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
import joblib
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

warnings.filterwarnings('ignore')

def processing_data(data_path):

    """
    数据处理
    :param data_path: 数据集路径
    :return: feature1,feature2,label:处理后的特征数据、标签数据
    """
    feature1,feature2,label = None, None, None
    # -------------------------- 实现数据处理部分代码 ----------------------------

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

    return feature1,feature2,label


def feature_select(feature1, feature2, label):
    """
    特征选择
    :param  feature1,feature2,label: 数据处理后的输入特征数据，标签数据
    :return: new_features,label:特征选择后的特征数据、标签数据
    """
    new_features= None
    # -------------------------- 实现特征选择部分代码 ----------------------------
    
    # 统计特征值和label的皮尔孙相关系数  对两类特征分别进行排序筛选特征
    select_feature_number1 = 25
    select_feature_number2 = 10
    select_feature1 = SelectKBest(lambda X, Y: tuple(map(tuple,np.array(list(map(lambda x:pearsonr(x, Y), X.T))).T)),
                                  k=select_feature_number1
                                 ).fit(feature1, np.array(label).flatten()).get_support(indices=True)

    select_feature2 = SelectKBest(lambda X, Y: tuple(map(tuple,np.array(list(map(lambda x:pearsonr(x, Y), X.T))).T)),
                                  k=select_feature_number2
                                 ).fit(feature2, np.array(label).flatten()).get_support(indices=True)

    # 查看排序后特征
    print("select feature1 name:", select_feature1)
    print("select feature2 name:", select_feature2)
    
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
    print("主成分1的⽅差变化信息：", variance_explained1)
    print("主成分2的⽅差变化信息：", variance_explained2)
    from mpl_toolkits.mplot3d import Axes3D
    
    # 可视化标签中不能出现负值
    pca_label = np.array(label).flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feature_pca1[:, 0], feature_pca1[:, 1], feature_pca2[:, 0],c=pca_label)
    # plt.show()
    # 对 feature1 和 feature2 进⾏整合
    feature_pca1_df = pd.DataFrame(feature_pca1)
    feature_pca2_df = pd.DataFrame(feature_pca2)
    
    new_features = pd.concat([feature_pca1_df, feature_pca2_df], axis=1)
    # ------------------------------------------------------------------------
    # 返回筛选后的数据
    return new_features,label

def data_split(features,label):

    """
    数据切分
    :param  features,label: 特征选择后的输入特征数据、类标数据
    :return: X_train, X_val, X_test,y_train, y_val, y_test:数据切分后的训练数据、验证数据、测试数据
    """

    X_train, X_val, X_test,y_train, y_val, y_test=None, None,None, None, None, None
    # -------------------------- 实现数据切分部分代码 ----------------------------
    # 将 features 和 label 数据切分成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=10, stratify=label)

    # 将 X_train 和 y_train 进一步切分为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=5, stratify=y_train)

    # ------------------------------------------------------------------------

    return X_train, X_val, X_test,y_train, y_val, y_test

def plot_learning_curve(estimator, X, y, cv=None, n_jobs=1):
    """
    绘制学习曲线
    :param estimator: 训练好的模型
    :param X:绘制图像的 X 轴数据
    :param y:绘制图像的 y 轴数据
    :param cv: 交叉验证
    :param n_jobs:
    :return:
    """
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure('Learning Curve', facecolor='lightgray')
    plt.title('Learning Curve')
    plt.xlabel('train size')
    plt.ylabel('score')
    plt.grid(linestyle=":")
    plt.plot(train_sizes, train_scores_mean, label='traning score')
    plt.plot(train_sizes, test_scores_mean, label='val score')
    plt.legend()
    plt.show()

def search_model(X_train, y_train,X_val,y_val, model_save_path):
    """
    创建、训练、优化和保存深度学习模型
    :param X_train, y_train: 训练集数据
    :param X_val,y_val: 验证集数据
    :param save_model_path: 保存模型的路径和名称
    :return:|
    """
    # --------------------- 实现模型创建、训练、优化和保存等部分的代码 ---------------------
    #创建监督学习模型 以决策树为例
    clf = tree.DecisionTreeClassifier(random_state=42)

    # 创建调节的参数列表
    parameters = {'max_depth': range(3,10),
                  'min_samples_split': range(5,10)}

    # 创建一个fbeta_score打分对象 以F-score为例
    scorer = make_scorer(fbeta_score, beta=1)

    # 在分类器上使用网格搜索，使用'scorer'作为评价函数
    kfold = KFold(n_splits=10) #切割成十份

    # 同时传入交叉验证函数
    grid_obj = GridSearchCV(clf, parameters, scorer, cv=kfold)

    #绘制学习曲线
    plot_learning_curve(clf, X_train, y_train, cv=kfold, n_jobs=4)

    # 用训练数据拟合网格搜索对象并找到最佳参数
    grid_obj.fit(X_train, y_train)

    # 得到estimator并保存
    best_clf = grid_obj.best_estimator_
    joblib.dump(best_clf, model_save_path)

    # 使用没有调优的模型做预测
    predictions = (clf.fit(X_train, y_train)).predict(X_val)
    best_predictions = best_clf.predict(X_val)

    # 调优后的模型
    print ("best_clf\n------")
    print (best_clf)

    # 汇报调参前和调参后的分数
    print("\nUnoptimized model\n------")
    print("Accuracy score on validation data: {:.4f}".format(accuracy_score(y_val, predictions)))
    print("Recall score on validation data: {:.4f}".format(recall_score(y_val, predictions)))
    print("F-score on validation data: {:.4f}".format(fbeta_score(y_val, predictions, beta = 1)))
    print("\nOptimized Model\n------")
    print("Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_val, best_predictions)))
    print("Recall score on validation data: {:.4f}".format(recall_score(y_val, best_predictions)))
    print("Final F-score on the validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta = 1)))

    # 保存模型（请写好保存模型的路径及名称）
    # -------------------------------------------------------------------------


def load_and_model_prediction(X_test,y_test,save_model_path):
    """
    加载模型和评估模型
    可以实现，比如: 模型优化过程中的参数选择，测试集数据的准确率、召回率、F-score 等评价指标！
    主要步骤:
        1.加载模型(请填写你训练好的最佳模型),
        2.对自己训练的模型进行评估

    :param X_test,y_test: 测试集数据
    :param save_model_path: 加载模型的路径和名称,请填写你认为最好的模型
    :return:
    """
    # ----------------------- 实现模型加载和评估等部分的代码 -----------------------
    #加载模型
    my_model=joblib.load(save_model_path)

    #对测试数据进行预测
    copy_test = [value for value in X_test]
    copy_predicts = my_model.predict(X_test)

    print ("Accuracy on test data: {:.4f}".format(accuracy_score(y_test, copy_predicts)))
    print ("Recall on test data: {:.4f}".format(recall_score(y_test, copy_predicts)))
    print ("F-score on test data: {:.4f}".format(fbeta_score(y_test, copy_predicts, beta = 1)))
    # ---------------------------------------------------------------------------



def main():
    """
    监督学习模型训练流程, 包含数据处理、特征选择、训练优化模型、模型保存、评价模型等。
    如果对训练出来的模型不满意, 你可以通过修改数据处理方法、特征选择方法、调整模型类型和参数等方法重新训练模型, 直至训练出你满意的模型。
    如果你对自己训练出来的模型非常满意, 则可以进行测试提交!
    :return:
    """
    data_path = "DataSet.xlsx"  # 数据集路径

    save_model_path = './results/my_test_model.m'  # 保存模型路径和名称

    # 获取数据 预处理
    feature1,feature2,label = processing_data(data_path)

    #特征选择
    new_features,label = feature_select(feature1, feature2, label)

    #数据划分
    X_train, X_val, X_test,y_train, y_val, y_test = data_split(new_features,label)

    # 创建、训练和保存模型
    search_model(X_train, y_train,X_val,y_val, save_model_path)

    # 评估模型
    load_and_model_prediction(X_test,y_test,save_model_path)


if __name__ == '__main__':
    main()      