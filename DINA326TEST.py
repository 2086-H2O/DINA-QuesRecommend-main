import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# DINA 模型相关函数（完全保持不变）
def compute_eta(Q, A):
    kowns = np.sum(Q * Q, axis=0)
    cross = np.dot(A, Q)
    eta = np.ones(shape=(A.shape[0], Q.shape[1]))
    eta[cross < kowns] = 0
    return eta


def compute_propa(eta, s, g):
    propa = (g ** (1 - eta)) * ((1 - s) ** eta)
    propa[propa <= 0] = 1e-10
    propa[propa >= 1] = 1 - 1e-10
    return propa


def compute_gamma(X, pi, propa):
    log_pj = np.log(propa)
    log_qj = np.log(1 - propa)
    log_pi = np.log(pi)
    gamma = np.exp(np.dot(X, log_pj.T) + np.dot((1 - X), log_qj.T) + log_pi)
    gamma_sum = np.sum(gamma, axis=1)
    gamma = (gamma.T / gamma_sum).T
    return gamma


def compute_theta(X, gamma, eta):
    I0 = np.dot(gamma, 1 - eta)
    I1 = np.dot(gamma, eta)
    R0 = I0 * X
    R1 = I1 * X
    I0 = np.sum(I0, axis=0)
    I1 = np.sum(I1, axis=0)
    R0 = np.sum(R0, axis=0)
    R1 = np.sum(R1, axis=0)
    I0[I0 <= 0] = 1e-15
    I1[I1 <= 0] = 1e-15
    g = R0 / I0
    s = (I1 - R1) / I1
    pi = np.sum(gamma, axis=0) / gamma.shape[0]
    pi[pi <= 0] = 1e-15
    pi[pi >= 1] = 1 - 1e-15
    return pi, s, g


def em(X, Q, maxIter=1000, tol=1e-3, prior=None):
    n_stu = X.shape[0]
    n_qus = X.shape[1]
    n_kno = Q.shape[0]
    g = np.random.random(n_qus) * 0.25
    s = np.random.random(n_qus) * 0.25
    A_all = np.array(list(product([0, 1], repeat=n_kno)))
    if prior is None:
        pi = np.ones(shape=(A_all.shape[0])) / A_all.shape[0]
    else:
        pi = prior
    for t in range(maxIter):
        eta = compute_eta(Q, A_all)
        propa = compute_propa(eta, s, g)
        gamma = compute_gamma(X, pi, propa)
        pi_t, s_t, g_t = compute_theta(X, gamma, eta)
        update = max(np.max(np.abs(pi_t - pi)), np.max(np.abs(g_t - g)), np.max(np.abs(s_t - s)))
        if update < tol:
            return pi_t, g_t, s_t, gamma
        if prior is None:
            pi = pi_t
        g = g_t
        s = s_t
    return pi, g, s, gamma


def solve(gamma, n_kownlege):
    A_all = np.array(list(product([0, 1], repeat=n_kownlege)))
    A_idx = np.argmax(gamma, axis=1)
    return A_all[A_idx], A_idx


def joint_loglike(X, Q, s, g, pi):
    A_all = np.array(list(product([0, 1], repeat=Q.shape[0])))
    eta = compute_eta(Q, A_all)
    propa = compute_propa(eta, s, g)
    log_pj = np.log(propa)
    log_qj = np.log(1 - propa)
    log_pi = np.log(pi)
    L = np.dot(X, log_pj.T) + np.dot((1 - X), log_qj.T) + log_pi
    L = L * gamma
    return np.sum(L)


def evaluate(X, Q, priors):
    n = len(priors)
    results = []
    for i in range(n):
        pi, g, s, gamma = em(X, Q, maxIter=100, tol=1e-6, prior=priors[i])
        A, A_idx = solve(gamma, Q.shape[0])
        results.append((pi, g, s, gamma, A, A_idx))
    return results


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


def get_priors(A_all, p_know, p_know_list):
    prior1 = np.ones(A_all.shape[0]) / A_all.shape[0]
    prior2 = np.ones(A_all.shape[0])
    prior3 = np.ones(A_all.shape[0])
    n_kno = A_all.shape[1]
    # 扩展 p_know_list 以匹配 n_kno
    if len(p_know_list) < n_kno:
        p_know_list = p_know_list * (n_kno // len(p_know_list) + 1)
    p_know_list = p_know_list[:n_kno]  # 裁剪到 n_kno
    for l in range(A_all.shape[0]):
        for k in range(A_all.shape[1]):
            p = p_know_list[k]
            prior2[l] *= (p_know ** A_all[l, k] * (1 - p_know) ** (1 - A_all[l, k]))
            prior3[l] *= (p ** A_all[l, k] * (1 - p) ** (1 - A_all[l, k]))
    return [prior1, prior2, prior3, None]

# 加载原始数据和分组结果
data = pd.read_csv(r"C:\Users\BNT\转换\cleaned_data_20250326_0931.csv")
group_data = pd.read_csv('optimal_student_groups_leiden.csv')

# 检查分组数据的列名
print("分组数据的列名:", group_data.columns.tolist())

# 获取所有组
groups = group_data['group'].unique()
print(f"总共有 {len(groups)} 个组需要分析")

# 计算每个组的学生数
group_counts = group_data['group'].value_counts()
print("每个组的学生数分布:")
print(group_counts)

# 过滤掉学生数为 1 的组
valid_groups = group_counts[group_counts > 1].index
print(f"学生数大于 1 的组数: {len(valid_groups)}")

# 存储所有组的结果
all_results = []

# 读取 output_matrix.xlsx 文件
output_matrix = pd.read_excel('output_matrix.xlsx', index_col=0)  # 假设第一列是知识点名称（如 Q1, Q2, ...）
print("output_matrix 的列名（题目 ID）:", output_matrix.columns.tolist())
print("output_matrix 的行名（知识点）:", output_matrix.index.tolist())

# 获取知识点数量（行数）和所有可能的题目 ID（列名）
n_kno_total = output_matrix.shape[0]  # 知识点数量（例如 18）
all_qs_ids = output_matrix.columns  # 所有题目 ID

# 逐组分析（只处理学生数大于 1 的组）
for group_id in tqdm(valid_groups):
    print(f"\n=== 处理组 {group_id} ===")

    # 提取该组的学生 ID
    group_students = group_data[group_data['group'] == group_id]['student_id'].values
    n_students = len(group_students)

    # 从原始数据中提取该组的做题矩阵 X
    group_data_subset = data[data['student_id'].isin(group_students)].drop_duplicates(subset=['student_id', 'qs_id'])
    X_group_df = group_data_subset.pivot(index='student_id', columns='qs_id', values='qs_validity').fillna(0)
    X_group = X_group_df.values
    n_questions = X_group.shape[1]
    print(f"组 {group_id} - 学生数: {X_group.shape[0]}, 题目数: {n_questions}")

    # 打印学生做题矩阵 X（前 5 行，前 10 列）
    print(f"组 {group_id} - 学生做题矩阵 X (前 5 行，前 10 列):")
    print(X_group_df.iloc[:min(5, n_students), :min(10, n_questions)])

    # ============= 修改点：从 output_matrix.xlsx 构建 Q 矩阵 =============
    max_knowledge = 8  # 设置最大知识点数量
    n_kno = min(max(1, n_questions // 5), max_knowledge, n_kno_total)  # 知识点数量限制
    group_qs_ids = X_group_df.columns  # 组内题目 ID

    # 初始化 Q 矩阵
    Q = np.zeros((n_kno, n_questions), dtype=int)

    # 遍历组内每个题目 ID，查找其在 output_matrix 中的对应列
    for j, qs_id in enumerate(group_qs_ids):
        if qs_id in all_qs_ids:
            # 找到该题目在 output_matrix 中的列
            qs_col = output_matrix[qs_id].values
            # 取前 n_kno 个知识点（如果知识点数量超过 n_kno）
            Q[:, j] = qs_col[:n_kno]
        else:
            # 如果题目 ID 不在 output_matrix 中，保持全 0
            print(f"警告: 组 {group_id} 中的题目 ID {qs_id} 未在 output_matrix.xlsx 中找到，Q 矩阵对应列将填充为 0")

    print(f"组 {group_id} - 生成的 Q 矩阵形状: {Q.shape} (知识点数限制为≤{max_knowledge})")

    # 打印 Q 矩阵（前 5 行，前 10 列）
    print(f"组 {group_id} - Q 矩阵 (前 5 行，前 10 列):")
    print(pd.DataFrame(Q, index=[f"K{i + 1}" for i in range(n_kno)],
                       columns=X_group_df.columns[:n_questions]).iloc[:, :min(10, n_questions)])

    # 初始化先验
    A_all = np.array(list(product([0, 1], repeat=n_kno)))
    priors = get_priors(A_all, p_know=0.7, p_know_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    d = [2 * Q.shape[1], 2 * Q.shape[1], 2 * Q.shape[1], 2 * Q.shape[1] + 2 ** Q.shape[0]]

    # 存储该组的结果
    ss = []
    gs = []
    NLL = []
    AIC = []
    results = []
    X_sim = []

    # 测试四种先验
    for i in range(4):
        print(f"组 {group_id} - 测试先验模式 {i}")
        pi_t, g_t, s_t, gamma = em(X_group, Q, maxIter=1000, tol=1e-6, prior=priors[i])
        ss.append(s_t)
        gs.append(g_t)
        A, A_idx = solve(gamma, Q.shape[0])
        results.append((pi_t, g_t, s_t, gamma, A, A_idx))

        # 生成模拟数据
        eta = compute_eta(Q, A)
        propa = compute_propa(eta, s_t, g_t)
        X1 = np.random.binomial(1, propa)
        X_sim.append(X1)

        # 评估
        eval_results = evaluate(X1, Q, priors)
        df_result = score(eval_results, A, A_idx, s_t, g_t)
        print(f"组 {group_id} - 先验 {i} 的评估结果:")
        print(df_result)

        # 计算联合对数似然和 AIC
        LL = joint_loglike(X_group, Q, s_t, g_t, pi_t)
        NLL.append(-LL)
        AIC.append(-2 * LL + d[i])
        print(f"组 {group_id} - 先验 {i} - NLL: {NLL[i]}, AIC: {AIC[i]}")

    # 可视化 NLL 和 AIC
    plt.figure(figsize=(12, 6))
    labels = ["先验分布D1", "先验分布D2", "先验分布D3", "改进后的模型"]
    for i in range(4):
        plt.bar(np.arange(0, 4, 2) + i * 0.3, [NLL[i], AIC[i]], width=0.25, label=labels[i])
    plt.legend(fontsize=14)
    plt.xticks([])
    plt.xlabel("NLL                                             AIC", fontsize=16)
    plt.title(f"组 {group_id} - NLL 和 AIC 比较")
    plt.savefig(f'group_{group_id}_nll_aic_comparison.png')
    plt.close()

    # 可视化失误率
    plt.figure(figsize=(12, 6))
    x = np.arange(1, n_questions + 1)
    plt.plot(x, ss[0], label="先验分布D1", linestyle="--")
    plt.plot(x, ss[1], label="先验分布D2", linestyle="-.")
    plt.plot(x, ss[2], label="先验分布D3", linestyle=":")
    plt.plot(x, ss[3], label="改进后的模型")
    plt.legend(fontsize=14)
    plt.xticks(x)
    plt.xlabel("试题编号", fontsize=14)
    plt.ylabel("失误率", fontsize=14)
    plt.title(f"组 {group_id} - 失误率")
    plt.savefig(f'group_{group_id}_slip_rate.png')
    plt.close()

    # 可视化猜对率
    plt.figure(figsize=(12, 6))
    plt.plot(x, gs[0], label="先验分布D1", linestyle="--")
    plt.plot(x, gs[1], label="先验分布D2", linestyle="-.")
    plt.plot(x, gs[2], label="先验分布D3", linestyle=":")
    plt.plot(x, gs[3], label="改进后的模型")
    plt.legend(fontsize=14)
    plt.xticks(x)
    plt.xlabel("试题编号", fontsize=14)
    plt.ylabel("猜对率", fontsize=14)
    plt.title(f"组 {group_id} - 猜对率")
    plt.savefig(f'group_{group_id}_guess_rate.png')
    plt.close()

    # 保存该组的评估结果
    all_results.append({
        'group': group_id,
        'X': X_group,
        'Q': Q,
        'results': results,
        'NLL': NLL,
        'AIC': AIC
    })

# 保存所有组的评估结果到 CSV
results_df = pd.DataFrame([
    {'group': res['group'], 'NLL_D1': res['NLL'][0], 'NLL_D2': res['NLL'][1],
     'NLL_D3': res['NLL'][2], 'NLL_D4': res['NLL'][3],
     'AIC_D1': res['AIC'][0], 'AIC_D2': res['AIC'][1],
     'AIC_D3': res['AIC'][2], 'AIC_D4': res['AIC'][3]}
    for res in all_results
])
results_df.to_csv('all_groups_dina_results.csv', index=False)
print("\n所有组的分析结果已保存到 'all_groups_dina_results.csv'")