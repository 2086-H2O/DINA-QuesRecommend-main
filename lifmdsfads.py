import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error

from CD import CD


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






# 创建密集评分矩阵
data_dir = "./data/"
df_frac20X = pd.read_csv('/usr/local/xiaoqi2022/pythonProject1/data/frac20X.csv')
df_frac20Q = pd.read_csv('/usr/local/xiaoqi2022/pythonProject1/data/frac20Q.csv')

# df_frac20X = pd.read_csv(data_dir + "frac20X.csv")
# df_frac20Q = pd.read_csv(data_dir + "frac20Q.csv")
X = df_frac20X.values
Q = df_frac20Q.values.T

n_stu = X.shape[0]
if X.shape[1]==Q.shape[1] :
    n_que = X.shape[1]
else:
    print("format error!")
n_kno = Q.shape[0]

print(n_stu)
print(n_que)
print(n_kno)

#总体均值
mean_value = np.mean(X)


# 创建PMF实例并训练
pmf = PMF(X, k=2, alpha=0.01, beta=0.02, iterations=1000)
pmf.train()

# 预测得分（PMF）
pmf_scores = pmf.predict()
print("\n预测得分矩阵（PMF）：\n", pmf_scores)

# 创建CD实例并训练
cd =CD(X,Q)
# 预测得分（CD）
A_real=cd.calculate_A_real_values()
print("\n预测得分矩阵（CD）：\n", A_real)

#比重参数
rho=0.5

eta=np.zeros((n_stu,n_que))

for u in range(n_stu):
    # 对A_real第u行求和 求平均
    row_index = u  # 行索引从0开始
    row_u_sum = np.sum(A_real[row_index, :])
    bu = row_u_sum / n_que
    for v in range(n_que):
        # 对A_real第求v列和 求平均
        row_index = v  # 行索引从0开始
        row_v_sum = np.sum(A_real[:, row_index ])
        bv= row_v_sum / n_stu

        buv = bu + bv
        eta[u][v] = mean_value + rho*buv + (1-rho)*pmf_scores[u][v]

print (eta)




#题目分数矩阵（实际输入）

#推荐分数矩阵（概率或分数在范围内就推荐）

#返回推荐题号
