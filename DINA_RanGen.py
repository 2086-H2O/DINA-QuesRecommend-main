import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error

plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）


# 依据知识点组合和试题知识点分布，计算理想情况下，各知识点组合的答题情况
def compute_eta(Q, A):
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
    I0 = np.dot(gamma, 1 - eta)
    # 获取能答对试题的知识状态
    I1 = np.dot(gamma, eta)

    # 计算不足以答对试题时却答对的期望
    R0 = I0 * X
    # 计算足以答对试题时，答对的期望
    R1 = I1 * X

    I0 = np.sum(I0, axis=0)
    I1 = np.sum(I1, axis=0)
    R0 = np.sum(R0, axis=0)
    R1 = np.sum(R1, axis=0)

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


# 使用EM算法对模型中各个参数进行评估
# 若未传入知识状态的先验概率，则动态估计知识状态概率
def em(X, Q, maxIter=1000, tol=1e-3, prior=None):
    n_stu = X.shape[0]
    n_qus = X.shape[1]
    n_kno = Q.shape[0]

    g = np.random.random(n_qus) * 0.25
    s = np.random.random(n_qus) * 0.25

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

        # 计算理想情况下，各种知识状态的答题情况
        eta = compute_eta(Q, A_all)
        # 加入猜对率和失误率后，各种知识状态的答题情况
        propa = compute_propa(eta, s, g)
        # E步计算期望
        gamma = compute_gamma(X, pi, propa)
        # M步更新模型参数
        pi_t, s_t, g_t = compute_theta(X, gamma, eta)

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


# 评估学生的知识状态
def solve(gamma, n_kownlege):
    A_all = np.array(list(product([0, 1], repeat=n_kownlege)))
    A_idx = np.argmax(gamma, axis=1)
    return A_all[A_idx], A_idx


# # 计算全数据的联合概率对数似然
# def joint_loglike(X, Q, s, g, pi):
#     A_all = np.array(list(product([0, 1], repeat=Q.shape[0])))
#     eta = compute_eta(Q, A_all)
#     propa = compute_propa(eta, s, g)
#     log_pj = np.log(propa)
#     log_qj = np.log(1 - propa)
#     log_pi = np.log(pi)
#     # log[(P(x_i |a_u)P(a_u)]
#     L = np.dot(X, log_pj.T) + np.dot((1 - X), log_qj.T) + log_pi
#     # P(a_u |x_i)log[(P(x_i |a_u)P(a_u)]
#     L = L * gamma
#     return np.sum(L)


# 计算得分数据(边缘概率)对数似然
def marginal_loglike(X, Q, s, g, pi):
    A_all = np.array(list(product([0, 1], repeat=Q.shape[0])))
    eta = compute_eta(Q, A_all)
    propa = compute_propa(eta, s, g)
    gamma = compute_gamma(X, pi, propa)
    log_pj = np.log(propa)
    log_qj = np.log(1 - propa)
    log_pi = np.log(pi)

    # P(X_i|alpha_u)p(alpha_u)
    L = np.exp(np.dot(X, log_pj.T) + np.dot((1 - X), log_qj.T) + log_pi)
    # Nx1 sum_u P(X_i|alpha_u)p(alpha_u)
    L = np.sum(L, axis=1)
    # sum_i{log[ sum_u P(X_i|alpha_u)p(alpha_u)]}
    L = np.log(L)
    return np.sum(L)


# 生成均匀分布的被试认知状态
def create_A_uniform(n_stu, A_all):
    # 使用均匀分布进行随机抽样，生成制定数量的被试
    A_index = (np.random.uniform(0, A_all.shape[0], size=(n_stu))).astype(int)
    # 依据被试认知状态的索引，获取对应的认知状态
    A = A_all[A_index]
    # 返回被试的知识掌握矩阵，被试的认知状态索引
    return A, A_index


# 生成多项式分布的被试认知状态
# 如果传入的p_know为浮点类型，则将所有的知识点掌握概率都设置为该值，即所有知识点的掌握概率相同
# 如果传入的p_know为数组，则数组的长度需与知识点的梳理对应，代表每个知识点被掌握的概率
def create_A_multinomial(n_stu, A_all, p_know=0.7):
    # 进行二项式采样，每个知识点采样为1的概率为0.7，采样为0的概率为0.3，生成被试知识状态矩阵
    A = np.random.binomial(1, p_know, (n_stu, A_all.shape[1]))

    # 计算每个被试对应那种知识状态
    A_index = np.zeros(n_stu)
    for i in range(n_stu):
        for l in range(A_all.shape[0]):
            if np.sum(np.abs(A[i] - A_all[l])) == 0:
                A_index[i] = l

    # 返回被试的知识掌握矩阵，被试的认知状态索引和该多项式分布的先验概率
    return A, A_index


# 对模型参数进行估计
def evaluate(X, Q, priors):
    n = len(priors)
    results = []

    for i in range(n):
        pi, g, s, gamma = em(X, Q, maxIter=100, tol=1e-6, prior=priors[i])
        A, A_idx = solve(gamma, Q.shape[0])
        results.append((pi, g, s, gamma, A, A_idx))

    return results


# 计算各项指标的得分
def score(results, A_test, A_test_idx, s, g):
    pmrs = []
    mmrs = []
    sr2s = []
    gr2s = []
    smaes = []
    gmaes = []
    for i in range(len(results)):
        pmrs.append(accuracy_score(results[i][5], A_test_idx))
        mmrs.append(accuracy_score(results[i][4].flatten(), A_test.flatten()))
        sr2s.append(r2_score(results[i][2], s))
        gr2s.append(r2_score(results[i][1], g))
        smaes.append(mean_absolute_error(results[i][2], s))
        gmaes.append(mean_absolute_error(results[i][1], g))

    df_result = pd.DataFrame({"pmr": pmrs, "mmr": mmrs, "s_r2": sr2s, "g_r2": gr2s, "s_mae": smaes, "g_mae": gmaes})
    return df_result


# 获取三种认知分布的先验分布
# 1. 平均分布
# 2. 多项式分布(各个知识点掌握的概率相同)
# 3. 多项式分布(各个知识点掌握的概率不同)
def get_priors(A_all, p_know, p_know_list):
    # 计算平均分布的先验分布
    prior1 = np.ones(A_all.shape[0]) / A_all.shape[0]
    # 计算多项式分布的先验概率
    # 各知识点掌握概率相同的
    prior2 = np.ones(A_all.shape[0])
    prior3 = np.ones(A_all.shape[0])

    for l in range(A_all.shape[0]):
        for k in range(A_all.shape[1]):
            p = p_know_list[k]
            prior2[l] *= (p_know ** A_all[l, k] * (1 - p_know) ** (1 - A_all[l, k]))
            prior3[l] *= (p ** A_all[l, k] * (1 - p) ** (1 - A_all[l, k]))

    return [prior1, prior2, prior3, None]


# 实际数据输入 未实现########################################################################################

def test_single(n_stu, n_qus, n_kno, priors, distribution, A_all, randomState=1):
    # 设置随机数种子，使得实验结果可重现
    np.random.seed(randomState)

    # 生成Q矩阵(二项分布，每个地方尝试一次，成功失败概率都是0.5)
    Q = np.random.binomial(1, 0.5, (n_kno, n_qus))

    if distribution == 1:
        # 获取平均分布的学生认知矩阵、学生认知状态索引和平均分布的先验概率
        A, A_idx = create_A_uniform(n_stu, A_all)
    elif distribution == 2:
        # 获取多项式分布(各知识点掌握概率相同)的学生认知矩阵、学生认知状态索引和多项式分布的先验概率
        A, A_idx = create_A_multinomial(n_stu, A_all)
    else:
        # 获取多项式分布(各知识点掌握概率不同)的学生认知矩阵、学生认知状态索引和多项式分布的先验概率
        A, A_idx = create_A_multinomial(n_stu, A_all, p_know=[0.4, 0.5, 0.6, 0.7, 0.8])

    # 随机产生猜对率和失误率
    g = np.random.random(n_qus) * 0.5
    s = np.random.random(n_qus) * 0.2

    # 计算平均分布下，学生能否答对各道试题
    eta = compute_eta(Q, A)
    # 计算平均分布下，学生答对各道试题的概率
    propa = compute_propa(eta, s, g)
    # 依据答对概率进行采样，得出学生试题得分矩阵
    X = np.random.binomial(1, propa)

    results = evaluate(X, Q, priors)

    df_result = score(results, A, A_idx, s, g)

    return df_result


def test1(n_stu, n_qus, n_kno, distribution=1, times=100):
    pmrs = np.zeros(shape=(4, times))
    mmrs = np.zeros(shape=(4, times))
    sr2s = np.zeros(shape=(4, times))
    gr2s = np.zeros(shape=(4, times))
    smaes = np.zeros(shape=(4, times))
    gmaes = np.zeros(shape=(4, times))

    # 获取所有可能的认知状态
    A_all = np.array(list(product([0, 1], repeat=n_kno)))
    priors = get_priors(A_all, p_know=0.7, p_know_list=[0.4, 0.5, 0.6, 0.7, 0.8])

    for i in tqdm(range(times)):
        df_single = test_single(n_stu, n_qus, n_kno, priors, distribution, A_all, randomState=i)
        pmrs[:, i] = df_single.pmr
        mmrs[:, i] = df_single.mmr
        sr2s[:, i] = df_single.s_r2
        gr2s[:, i] = df_single.g_r2
        smaes[:, i] = df_single.s_mae
        gmaes[:, i] = df_single.g_mae

    df_result = pd.DataFrame({"pmr": pmrs.mean(axis=1), "mmr": mmrs.mean(axis=1), "s_r2": sr2s.mean(axis=1),
                              "g_r2": gr2s.mean(axis=1), "s_mae": smaes.mean(axis=1), "g_mae": gmaes.mean(axis=1)})
    return df_result

# # 模式数据为平均分布时，估计效果对比
# print(test1(n_stu=1920, n_qus=30, n_kno=5, distribution=1))
# print(test1(n_stu=1920, n_qus=30, n_kno=5, distribution=2))
# print(test1(n_stu=1920, n_qus=30, n_kno=5, distribution=3))






