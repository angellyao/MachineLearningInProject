import pandas as pd
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from numpy import set_printoptions


# 导入数据，数据预处理的方法
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler


# 选择特征
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA


# 导入模型
from sklearn.linear_model import LogisticRegression                  # 线性模型
from sklearn.linear_model import Ridge                               # 线性模型
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # 线性模型
from sklearn.svm import SVC                        # 非线性模型
from sklearn.tree import DecisionTreeClassifier    # 非线性模型
from sklearn.neighbors import KNeighborsClassifier # 非线性模型
from sklearn.naive_bayes import GaussianNB         # 非线性模型


# 模型选择用的包：分离数据集、交叉验证、网格搜索超参数
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# 集成算法的导入
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier


# 自动流程过程
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion


# 模型评估：混淆矩阵，数据报告
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# 超参数调参
from scipy.stats import uniform


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 导入数据
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# 使用Pandas导入CSV数据
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

"""
from numpy import loadtxt
# 使用numpy导入CSV数据
filename = 'pima_data.csv'
with open(filename, 'rt') as raw_data:
    data = loadtxt(raw_data, delimiter=',')
    print(data.shape)
    
    
    
from csv import reader
import numpy as np
# 使用标准的Python类库导入CSV数据
filename = 'pima_data.csv'
with open(filename, 'rt') as raw_data:
    readers = reader(raw_data, delimiter=',')
    x = list(readers)
    data = np.array(x).astype('float')
    print(data.shape)
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 查看数据
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 显示数据最初10行
peek = data.head(10)
print(peek)

# 显示数据的行和列数据
print(data.shape)

# 显示数据的行和列数据
print(data.dtypes)

# 描述性统计
set_option('display.width', 100)
set_option('precision', 4)  # 设置数据的精确度
print(data.describe())

# 数据分类分布统计
print(data.groupby('class').size())

# 显示数据的相关性
set_option('display.width', 100)
set_option('precision', 2) # 设置数据的精确度
print(data.corr(method='pearson'))

# 计算数据的高斯偏离
print(data.skew())


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 可视化查看数据
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 直方图
data.hist()
plt.show()

# 密度图
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
plt.show()

# 箱线图
data.plot(kind = 'box', subplots = True, layout = (3,3), sharex = False)
plt.show()

# 相关矩阵图
correlations = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, 9, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

# 散点矩阵图
scatter_matrix(data)
plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 数据预处理
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 调整数据尺度
# 将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]

transform = Binarizer(threshold=0.0).fit(X)
# 数据转换
newX = transform.transform(X)
# 设定数据的打印格式
set_printoptions(precision=3)
print(newX)


# 正态化数据
scaler = StandardScaler().fit(X)
# 数据转换
rescaledX = scaler.transform(X)
# 设定数据的打印格式
set_printoptions(precision=3)
print(rescaledX)


# 标准化数据
scaler = Normalizer().fit(X)
# 数据转换
rescaledX = scaler.transform(X)
# 设定数据的打印格式
set_printoptions(precision=3)
print(rescaledX)


# 二值数据
scaler = MinMaxScaler(feature_range=(0, 1))
# 数据转换
rescaledX = scaler.fit_transform(X)
# 设定数据的打印格式
set_printoptions(precision=3)
print(rescaledX)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 数据特征选定
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 单变量特征选择
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
print(features)

# 递归特征消除
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("特征个数：")
print(fit.n_features_)
print("被选定的特征：")
print(fit.support_)
print("特征排名：")
print(fit.ranking_)

# 通过主要成分分析选定数据特征
pca = PCA(n_components=3)
fit = pca.fit(X)
print("解释方差：%s" % fit.explained_variance_ratio_)
print(fit.components_)

# 通过决策树计算特征的重要性
model = ExtraTreesClassifier()
fit = model.fit(X, Y)
print(fit.feature_importances_)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 评估算法
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 分离训练数据集和评估数据集
test_size = 0.33
seed = 4
X_train, X_test, Y_traing, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_traing)
result = model.score(X_test, Y_test)
print("算法评估结果：%.3f%%" % (result * 100))

# K折交叉验证
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
model = LogisticRegression()
result = cross_val_score(model, X, Y, cv=kfold)
print("算法评估结果：%.3f%% (%.3f%%)" % (result.mean() * 100, result.std() * 100))

# 弃一交叉验证
loocv = LeaveOneOut()
model = LogisticRegression()
result = cross_val_score(model, X, Y, cv=loocv)
print("算法评估结果：%.3f%% (%.3f%%)" % (result.mean() * 100, result.std() * 100))

# 重复随机分离评估数据集与训练数据集
n_splits = 10
test_size = 0.33
seed = 7
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
model = LogisticRegression()
result = cross_val_score(model, X, Y, cv=kfold)
print("算法评估结果：%.3f%% (%.3f%%)" % (result.mean() * 100, result.std() * 100))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 算法评估矩阵
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 分类准确度
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
model = LogisticRegression()
result = cross_val_score(model, X, Y, cv=kfold)
print("算法评估结果准确度：%.3f%% (%.3f%%)" % (result.mean(), result.std()))

# 对数损失函数
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
model = LogisticRegression()
scoring = 'neg_log_loss'
result = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print('Logloss %.3f (%.3f)' % (result.mean(), result.std()))

# AUC图
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
model = LogisticRegression()
scoring = 'roc_auc'
result = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print('AUC %.3f (%.3f)' % (result.mean(), result.std()))

# 混淆矩阵
test_size = 0.33
seed = 4
X_train, X_test, Y_traing, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_traing)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
classes = ['0', '1']
dataframe = pd.DataFrame(data=matrix, index=classes, columns=classes)
print(dataframe)

# 分类报告
test_size = 0.33
seed = 4
X_train, X_test, Y_traing, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_traing)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 算法审查
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 2大类，6种算法用于分类
# 线性算法：逻辑回归、线性判别分析
# 非线性算法：K近邻算法、贝叶斯分类器、分类与回归树、支持向量机
# 逻辑回归
model = LogisticRegression()
result = cross_val_score(model, X, Y, cv=kfold)
print(result.mean())

# 线性判别分析
model = LinearDiscriminantAnalysis()
result = cross_val_score(model, X, Y, cv=kfold)
print(result.mean())

# K近邻算法
model = KNeighborsClassifier()
result = cross_val_score(model, X, Y, cv=kfold)
print(result.mean())

# 贝叶斯分类器
model = GaussianNB()
result = cross_val_score(model, X, Y, cv=kfold)
print(result.mean())

# 分类与回归树
model = DecisionTreeClassifier()
result = cross_val_score(model, X, Y, cv=kfold)
print(result.mean())

# 支持向量机
model = SVC()
result = cross_val_score(model, X, Y, cv=kfold)
print(result.mean())



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 算法比较
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 上述算法的汇总与对比
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['SVM'] = SVC()
models['NB'] = GaussianNB()

results = []
for name in models:
    result = cross_val_score(models[name], X, Y, cv=kfold)
    results.append(result)
    msg = '%s: %.3f (%.3f)' % (name, result.mean(), result.std())
    print(msg)

# 图表显示
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 自动流程
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Pipeline过程
# 数据准备与生成模型
steps = []  # 生成 Pipeline
steps.append(('Standardize', StandardScaler()))
steps.append(('lda', LinearDiscriminantAnalysis()))
model = Pipeline(steps)
result = cross_val_score(model, X, Y, cv=kfold)
print(result.mean())



# 特征选择与生成模型
# 生成 feature union
features = []
features.append(('pca', PCA()))
features.append(('select_best', SelectKBest(k=6)))
steps = []  # 生成 Pipeline
steps.append(('feature_union', FeatureUnion(features)))
steps.append(('logistic', LogisticRegression()))
model = Pipeline(steps)
result = cross_val_score(model, X, Y, cv=kfold)
print(result.mean())


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 集成算法
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 集成算法的方法有
# 装袋算法：装袋决策树、随机森林、极端随机树
# 提升算法：AdaBoost、随机梯度提升
# 投票算法：投票算法

# 装袋决策树
kfold = KFold(n_splits=num_folds, random_state=seed)
cart = DecisionTreeClassifier()
num_tree = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_tree, random_state=seed)
result = cross_val_score(model, X, Y, cv=kfold)
print(result.mean())


# 随机森林
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
num_tree = 100
max_features = 3
model = RandomForestClassifier(n_estimators=num_tree, random_state=seed, max_features=max_features)
result = cross_val_score(model, X, Y, cv=kfold)
print(result.mean())


# 极端随机树
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
num_tree = 100
max_features = 7
model = ExtraTreesClassifier(n_estimators=num_tree, random_state=seed, max_features=max_features)
result = cross_val_score(model, X, Y, cv=kfold)
print(result.mean())


# AdaBoost
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
num_tree = 30
model = AdaBoostClassifier(n_estimators=num_tree, random_state=seed)
result = cross_val_score(model, X, Y, cv=kfold)
print(result.mean())


# 随机梯度提升
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
num_tree = 100
model = GradientBoostingClassifier(n_estimators=num_tree, random_state=seed)
result = cross_val_score(model, X, Y, cv=kfold)
print(result.mean())


# 投票算法
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
cart = DecisionTreeClassifier()
models = []
model_logistic = LogisticRegression()
models.append(('logistic', model_logistic))
model_cart = DecisionTreeClassifier()
models.append(('cart', model_cart))
model_svc = SVC()
models.append(('svm', model_svc))
ensemble_model = VotingClassifier(estimators=models)
result = cross_val_score(ensemble_model, X, Y, cv=kfold)
print(result.mean())



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 机器学习算法调参
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 1.网格搜索优化参数
# 算法实例化
model = Ridge()
# 设置要遍历的参数
param_grid = {'alpha': [1, 0.1, 0.01, 0.001, 0]}
# 通过网格搜索查询最优参数
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)
# 搜索结果
print('最高得分：%.3f' % grid.best_score_)
print('最优参数：%s' % grid.best_estimator_.alpha)


# 2.随机搜索优化参数
# 算法实例化
model = Ridge()
# 设置要遍历的参数
param_grid = {'alpha': uniform()}
# 通过网格搜索查询最优参数
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, random_state=7)
grid.fit(X, Y)
# 搜索结果
print('最高得分：%.3f' % grid.best_score_)
print('最优参数：%s' % grid.best_estimator_.alpha)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 结果部署
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
