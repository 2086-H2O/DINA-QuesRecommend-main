import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error


class PMF:
    def __init__(self, R, k, alpha=0.01, beta=0.02, iterations=1000):
        self.R = R  # 得分矩阵
        self.n_users, self.n_items = R.shape  # 用户和物品数量
        self.k = k  # 隐因子数量
        self.alpha = alpha  # 学习率
        self.beta = beta  # 正则化参数
        self.iterations = iterations  # 迭代次数

        # 初始化用户和物品的特征矩阵
        self.M = np.random.normal(scale=1./self.k, size=(self.n_users, self.k))
        self.N = np.random.normal(scale=1./self.k, size=(self.n_items, self.k))

    def train(self):
        for it in range(self.iterations):
            for u in range(self.n_users):
                for v in range(self.n_items):
                    if self.R[u, v] > 0:  # 只更新已评分的项
                        # 预测得分
                        prediction = self.M[u, :].dot(self.N[v, :].T)
                        # 计算误差
                        e = self.R[u, v] - prediction
                        # 更新特征矩阵
                        self.M[u, :] += self.alpha * (e * self.N[v, :] - self.beta * self.M[u, :])
                        self.N[v, :] += self.alpha * (e * self.M[u, :] - self.beta * self.N[v, :])

    def predict(self):
        return self.M.dot(self.N.T)  # 预测得分矩阵


# 生成示例评分矩阵
def generate_dense_ratings(num_users, num_items, density=0.9):
    R = np.zeros((num_users, num_items))
    for i in range(num_users):
        for j in range(num_items):
            if np.random.rand() < density:  # 根据密度生成评分
                R[i, j] = np.random.randint(1, 6)  # 评分范围 1-5
    return R

# 创建密集评分矩阵
data_dir = "./data/"
df_frac20X = pd.read_csv(data_dir + "frac20X.csv")
df_frac20Q = pd.read_csv(data_dir + "frac20Q.csv")
X = df_frac20X.values
Q = df_frac20Q.values.T

print("原始评分矩阵：\n", X)

# 将数据集随机分为训练集和测试集
def train_test_split(ratings, test_ratio=0.2):
    train = np.copy(ratings)
    test = np.zeros_like(ratings)

    for u in range(ratings.shape[0]):
        for i in range(ratings.shape[1]):
            if np.random.rand() < test_ratio:
                test[u, i] = ratings[u, i]  # 将该评分移动到测试集中
                train[u, i] = 0  # 从训练集中删除该评分

    return train, test

train, test = train_test_split(X)
print("\n训练集：\n", train)
print("\n测试集：\n", test)

# 创建PMF实例并训练
pmf = PMF(train, k=2, alpha=0.01, beta=0.02, iterations=1000)
pmf.train()

# 预测得分
predicted_scores = pmf.predict()
print("\n预测得分矩阵：\n", predicted_scores)

# 计算RMSE
def calculate_rmse(predicted, actual):
    # 仅考虑测试集中存在的评分
    mask = actual > 0
    rmse = np.sqrt(mean_squared_error(actual[mask], predicted[mask]))
    return rmse

rmse = calculate_rmse(predicted_scores, test)
print(f"\n测试集的RMSE: {rmse:.2f}")
