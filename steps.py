from itertools import product

import numpy as np
import pandas as pd


def compute_eta(Q, A):
    print("test")
    print(Q)
    print(A)
    # 计算各个试题需要的知识点数量
    kowns = np.sum(Q * Q, axis=0)
    # 计算各个知识点组合与试题交叉的知识点数量
    cross = np.dot(A, Q)
    # 初始化理想情况下答题结果矩阵
    eta = np.ones(shape=(A.shape[0], Q.shape[1]))
    # 如果知识点组合与试题交叉的知识点数量小于试题所需的知识点数量，则无法回答正确，否则能回答正确
    eta[cross < kowns] = 0
    # 返回理想情况下答题结果矩阵
    return eta


# 计算加入试题猜对率和失误率后，各个试题答对的概率
def compute_propa(eta, s, g):
    # eta[i,j] = 0时，试题答对的概率等于试题猜对率g_j
    # eta[i,j] = 1时，试题答对的概率等于1减去试题失误率s_j
    propa = (g ** (1 - eta)) * ((1 - s) ** eta)
    propa[propa <= 0] = 1e-10
    propa[propa >= 1] = 1 - 1e-10

    # 返回加入试题猜对率和失误率后，各个试题答对的概率
    return propa


# 计算给定答题情况和参数的情况下，学生掌握各个知识点组合的概率
def compute_gamma(X, pi, propa):
    # 这儿使用一个技巧 x = exp(log(x)),将连乘转换成求和，然后用矩阵进行运算
    log_pj = np.log(propa)
    log_qj = np.log(1 - propa)
    log_pi = np.log(pi)

    # 计算各个学生掌握这种知识点组合的置信
    gamma = np.exp(np.dot(X, log_pj.T) + np.dot((1 - X), log_qj.T) + log_pi)
    # 计算各个学生的总置信
    gamma_sum = np.sum(gamma, axis=1)
    # 进行归一化，计算各个学生掌握各种知识点组合的概率
    gamma = (gamma.T / gamma_sum).T
    # 返回学生掌握各个知识点组合的概率
    return gamma


# 评估各个参数的值
def compute_theta(X, gamma, eta):
    # 获取不足以答题试题的知识状态
    print("compure theta part")
    I0 = np.dot(gamma, 1 - eta)
    print(I0)
    # 获取能答对试题的知识状态
    I1 = np.dot(gamma, eta)
    print(I1)
    # 计算不足以答对试题时却答对的期望
    R0 = I0 * X
    print(R0)
    # 计算足以答对试题时，答对的期望
    R1 = I1 * X
    print(R1)

    print("compure theta part")
    I0 = np.sum(I0, axis=0)
    I1 = np.sum(I1, axis=0)
    R0 = np.sum(R0, axis=0)
    R1 = np.sum(R1, axis=0)
    print(I0)
    print(I1)
    print(R0)
    print(R1)

    I0[I0 <= 0] = 1e-15
    I1[I1 <= 0] = 1e-15

    # 更新猜对率和失误率
    g = R0 / I0
    s = (I1 - R1) / I1

    # 更新知识状态分布概率
    pi = np.sum(gamma, axis=0) / gamma.shape[0]
    pi[pi <= 0] = 1e-15
    pi[pi >= 1] = 1 - 1e-15

    return pi, s, g


def em(X, Q, maxIter=1000, tol=1e-3, prior=None):
    n_stu = X.shape[0]
    n_qus = X.shape[1]
    n_kno = Q.shape[0]

    # g = np.random.random(n_qus) * 0.25
    # s = np.random.random(n_qus) * 0.25

    g_list=[0.1, 0.1, 0.1, 0.1, 0.09, 0.1, 0.1]
    s_list=[0.2, 0.1, 0.2, 0.05, 0.2, 0.1, 0.2]
    g = np.array(g_list)
    s = np.array(s_list)
    print(g)
    print(s)
    # 获取所有可能的知识点组合
    A_all = np.array(list(product([0, 1], repeat=n_kno)))

    # 如果传入知识状态先验概率，则使用用户传入的先验概率
    # 否则使用动态估计的知识状态概率
    if prior is None:
        pi = np.ones(shape=(A_all.shape[0])) / A_all.shape[0]
    else:
        pi = prior

    # 循环迭代，进行E步和M步
    for t in range(maxIter):
        print("for step")
        print(t)
        # 计算理想情况下，各种知识状态的答题情况
        eta = compute_eta(Q, A_all)
        print("eta:")
        print(eta)
        # 加入猜对率和失误率后，各种知识状态的答题情况
        propa = compute_propa(eta, s, g)
        print("propa:")
        print(propa)
        # E步计算期望
        gamma = compute_gamma(X, pi, propa)
        print("gamma:")
        print(gamma)
        # M步更新模型参数
        pi_t, s_t, g_t = compute_theta(X, gamma, eta)
        print("pi_t:")
        print(pi_t)
        print("s_t:")
        print(s_t)
        print("g_t:")
        print(g_t)
        # 计算各个参数更新的绝对大小
        update = max(np.max(np.abs(pi_t - pi)), np.max(np.abs(g_t - g)), np.max(np.abs(s_t - s)))
        if update < tol:
            return pi_t, g_t, s_t, gamma

        # 更新参数准备下一轮迭代
        if prior is None:
            pi = pi_t
        g = g_t
        s = s_t

    #     print("step %d update %.8f" % (t, update))

    return pi, g, s, gamma


# data_dir = "./data/"
# df_frac20X = pd.read_csv(data_dir + "frac20X.csv")
# df_frac20Q = pd.read_csv(data_dir + "frac20Q.csv")
# X = df_frac20X.values
# Q = df_frac20Q.values.T

data_dir = "./Math1_1/"
df_X = pd.read_csv(data_dir + "data.csv")
df_Q = pd.read_csv(data_dir + "q.csv")
X = df_X.values
Q = df_Q.values.T




em(X,Q)