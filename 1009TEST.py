import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error
import os
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['Noto Sans S Chinese']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

import matplotlib
matplotlib.use('Agg')  # 不弹出图片窗口，直接保存

# DINA 模型相关函数
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


def joint_loglike(X, Q, s, g, pi,gamma):
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


# 改进后的Q矩阵构建（内存优化版）
def build_q_matrix(output_matrix, group_qs_ids, all_qs_ids, max_knowledge):
    """
    优化版Q矩阵构建函数
    参数:
        output_matrix: 原始知识点矩阵 (题目ID为行索引)
        group_qs_ids: 当前组的题目ID列表
        all_qs_ids: 所有有效题目ID集合
        max_knowledge: 最大知识点数限制
    返回:
        Q: 优化后的Q矩阵 (知识点×题目)
    """
    # --- 步骤1: 筛选有效知识点 ---
    # 统计每个知识点在output_matrix中的总覆盖率（排除当前组不存在的题目）
    knowledge_coverage = np.zeros(output_matrix.shape[1])
    for j, qs_id in enumerate(output_matrix.index):
        if qs_id in all_qs_ids:
            knowledge_coverage += output_matrix.loc[qs_id].values

    # 选择覆盖题目最多的前max_knowledge个知识点
    top_k_indices = np.argsort(-knowledge_coverage)[:max_knowledge]
    top_k_indices = sorted(top_k_indices)  # 保持原始顺序

    # --- 步骤2: 构建Q矩阵（保留原有题目ID匹配逻辑）---
    Q = np.zeros((len(top_k_indices), len(group_qs_ids)), dtype=int)

    missing_qs = 0
    for j, qs_id in enumerate(group_qs_ids):
        if qs_id in all_qs_ids:
            # 只选取top_k_indices对应的知识点列
            Q[:, j] = output_matrix.loc[qs_id].values[top_k_indices]
        else:
            missing_qs += 1
            Q[:, j] = 0  # 保持原有逻辑

    if missing_qs > 0:
        print(f"警告: 共 {missing_qs} 个题目未在output_matrix中找到，已填充为0")

    # --- 步骤3: 后处理验证 ---
    # 移除全零知识点（如果因题目缺失导致）
    non_zero_knowledge = np.where(Q.sum(axis=1) > 0)[0]
    Q = Q[non_zero_knowledge, :]

    # 确保每个知识点至少覆盖3题（否则移除）
    valid_knowledge = np.where(Q.sum(axis=1) >= 3)[0]
    if len(valid_knowledge) < Q.shape[0]:
        print(f"优化: 移除 {Q.shape[0] - len(valid_knowledge)} 个低覆盖知识点")
        Q = Q[valid_knowledge, :]

    print(f"最终Q矩阵形状: {Q.shape} (知识点×题目)")
    print(f"知识点覆盖统计: 平均每题 {Q.sum(axis=0).mean():.2f} 个知识点")

    return Q


def run_dina_self_consistency_test(
    group_id, X_group, Q, prior, d=None,
    maxIter=1000, tol=1e-6, verbose=True
):
    """
    针对某一组学生数据，运行 DINA 模型的自洽性检验流程
    返回结构化指标和解释文本3
    """

    if verbose:
        print(f"\n===== Group {group_id} - DINA 自洽性检验 =====")

    # Step 1: 拟合原始数据
    print("1. EM 拟合原始数据...")
    pi, g, s, gamma = em(X_group, Q, maxIter=maxIter, tol=tol, prior=prior)
    A, A_idx = solve(gamma, Q.shape[0])

    # Step 2: 生成模拟数据
    eta = compute_eta(Q, A) 
    propa = compute_propa(eta, s, g)
    X_sim = np.random.binomial(1, propa)

    # Step 3: 用模拟数据重新拟合（回归）
    pi_sim, g_sim, s_sim, gamma_sim = em(X_sim, Q, maxIter=maxIter, tol=tol, prior=prior)

    # Step 4: 评估参数误差
    slip_error = np.mean(np.abs(s - s_sim))
    guess_error = np.mean(np.abs(g - g_sim))
    pi_error = np.mean(np.abs(pi - pi_sim))

    # Step 5: 评估模拟答题矩阵与原始数据的一致性
    acc = accuracy_score(X_group.flatten(), X_sim.flatten())
    bce = log_loss(X_group.flatten(), propa.flatten(), labels=[0,1])

    # Step 6: 联合似然与 AIC
    LL = joint_loglike(X_group, Q, s, g, pi,gamma)
    NLL = -LL
    AIC_value = -2 * LL + d if d is not None else None



    # Step 7: 可视化参数对比，保存到文件夹

    # plt.figure(figsize=(10, 3))
    # plt.subplot(1, 2, 1)
    # plt.plot(s, label='原始 slip'); plt.plot(s_sim, label='回归 slip')
    # plt.legend(); plt.title(f'Group {group_id} - Slip对比')

    # plt.subplot(1, 2, 2)
    # plt.plot(g, label='原始 guess'); plt.plot(g_sim, label='回归 guess')
    # plt.legend(); plt.title(f'Group {group_id} - Guess对比')
    # plt.tight_layout()
    # plt.show()

    # 创建带时间戳的结果文件夹 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%MS")
    results_dir = f"test_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"所有结果将保存在文件夹: '{results_dir}'")

    # --- 修改后的绘图与保存逻辑 ---

    # 1. 找出最佳模型
    best_model_idx = np.argmin(AIC)
    best_aic = AIC[best_model_idx]
    labels = ["先验分布D1", "先验分布D2", "先验分布D3", "无先验模型"]
    
    # 2. 绘制并保存 NLL 和 AIC 比较图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(np.arange(4), AIC, width=0.4, label="AIC")
    bars[best_model_idx].set_color('salmon') # 高亮最佳模型
    plt.xticks(np.arange(4), labels)
    plt.ylabel("AIC 值 (越低越好)", fontsize=14)
    # 在标题中加入指标
    plt.title(f"组 {group_id} - AIC 比较 (最佳模型: {labels[best_model_idx]}, AIC={best_aic:.2f})", fontsize=16)
    plt.legend()
    # 构造保存路径并保存
    save_path = os.path.join(results_dir, f'group_{group_id}_aic_comparison.png')
    plt.savefig(save_path)
    plt.close() # 关闭图形，释放内存

    # 3. 绘制并保存失误率图
    avg_slip = np.mean(ss[best_model_idx]) # 计算最佳模型的平均失误率
    plt.figure(figsize=(12, 6))
    x = np.arange(1, n_questions + 1)
    for i in range(4):
        plt.plot(x, ss[i], label=f"{labels[i]}", linestyle="--", alpha=0.6)
    # 高亮最佳模型的曲线
    plt.plot(x, ss[best_model_idx], label=f"最佳模型: {labels[best_model_idx]}", linewidth=2.5, color='salmon')
    plt.legend(fontsize=12)
    plt.xticks(x)
    plt.xlabel("试题编号", fontsize=14)
    plt.ylabel("失误率", fontsize=14)
    # 在标题中加入指标
    plt.title(f"组 {group_id} - 失误率 (最佳模型平均值: {avg_slip:.4f})", fontsize=16)
    save_path = os.path.join(results_dir, f'group_{group_id}_slip_rate.png')
    plt.savefig(save_path)
    plt.close()

    # 4. 绘制并保存猜对率图
    avg_guess = np.mean(gs[best_model_idx]) # 计算最佳模型的平均猜测率
    plt.figure(figsize=(12, 6))
    for i in range(4):
        plt.plot(x, gs[i], label=f"{labels[i]}", linestyle="--", alpha=0.6)
    # 高亮最佳模型的曲线
    plt.plot(x, gs[best_model_idx], label=f"最佳模型: {labels[best_model_idx]}", linewidth=2.5, color='salmon')
    plt.legend(fontsize=12)
    plt.xticks(x)
    plt.xlabel("试题编号", fontsize=14)
    plt.ylabel("猜对率", fontsize=14)
    # 在标题中加入指标
    plt.title(f"组 {group_id} - 猜对率 (最佳模型平均值: {avg_guess:.4f})", fontsize=16)
    save_path = os.path.join(results_dir, f'group_{group_id}_guess_rate.png')
    plt.savefig(save_path)
    plt.close()




    # Step 8: 解释性评估输出
    slip_eval = "偏差较大，模型对 slip 敏感" if slip_error > 0.2 else "偏差较小，slip 拟合较好"
    guess_eval = "估计不稳定，可能与先验有关" if guess_error > 0.2 else "较稳定"
    pi_eval = "掌握模式分布估计稳定" if pi_error < 0.05 else "可能存在偏差"
    acc_eval = "中等，模型能部分还原答题行为" if acc > 0.4 else "较弱，模拟能力有限"

    if verbose:
        print(f"[Group {group_id}] 参数误差：")
        print(f"  - slip 平均误差：{slip_error:.4f}（{slip_eval}）")
        print(f"  - guess 平均误差：{guess_error:.4f}（{guess_eval}）")
        print(f"  - π 平均误差：{pi_error:.4f}（{pi_eval}）")
        print(f"[Group {group_id}] 模拟数据拟合能力：")
        print(f"  - 准确率：{acc:.4f}（{acc_eval}）")
        print(f"  - BCE Loss：{bce:.4f}")
        print(f"  - NLL = {NLL:.2f}")
        if AIC_value is not None:
            print(f"  - AIC = {AIC_value:.2f}")

    return {
        "group_id": group_id,
        "slip_error": slip_error,
        "guess_error": guess_error,
        "pi_error": pi_error,
        "X_accuracy": acc,
        "bce_loss": bce,
        "NLL": NLL,
        "AIC": AIC_value,
        "slip_eval": slip_eval,
        "guess_eval": guess_eval,
        "pi_eval": pi_eval,
        "acc_eval": acc_eval,
        "params": (pi, g, s, gamma),
        "params_sim": (pi_sim, g_sim, s_sim, gamma_sim),
        "X_sim": X_sim
    }
def generate_random_q_matrix(n_knowledge, n_questions, sparsity=0.1, min_knowledge_per_q=1, min_q_per_knowledge=1):
    """
    生成随机 Q 矩阵
    Args:
        n_knowledge: 知识点数
        n_questions: 题目数
        sparsity: Q 矩阵中 1 的比例（默认 10%）
        min_knowledge_per_q: 每题至少关联的知识点数
        min_q_per_knowledge: 每个知识点至少关联的题数
    """
    Q = np.zeros((n_knowledge, n_questions), dtype=int)

    # 确保每题至少关联 min_knowledge_per_q 个知识点
    for q in range(n_questions):
        k_indices = np.random.choice(n_knowledge, min_knowledge_per_q, replace=False)
        Q[k_indices, q] = 1

    # 随机填充剩余位置，控制稀疏度
    target_ones = int(sparsity * n_knowledge * n_questions) - min_knowledge_per_q * n_questions
    if target_ones > 0:
        flat_indices = np.random.choice(
            np.where(Q.ravel() == 0)[0],  # 仅选择未被初始化的位置
            size=target_ones,
            replace=False
        )
        Q.ravel()[flat_indices] = 1

    # 确保每个知识点至少关联 min_q_per_knowledge 道题
    for k in range(n_knowledge):
        if np.sum(Q[k, :]) < min_q_per_knowledge:
            q_indices = np.random.choice(n_questions, min_q_per_knowledge, replace=False)
            Q[k, q_indices] = 1

    return Q

# 使用方式

# 加载原始数据和分组结果
# data = pd.read_csv(r"C:\Users\BNT\转换\cleaned_data_20250326_0931.csv")
data = pd.read_csv(r"cleaned_data_20250326_0931.csv")
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
output_matrix = pd.read_excel('516matrix.xlsx', index_col=0)
# output_matrix.index = output_matrix.index.str.extract(r'(\d+)')[0].astype(int)
print("output_matrix :", output_matrix)
print("output_matrix 的列名（题目 ID）:", output_matrix.columns.tolist())
print("output_matrix 的行名（题目ID）:", output_matrix.index.tolist())

# 获取知识点数量（行数）和所有可能的题目 ID（列名）
n_kno_total = output_matrix.shape[0]  # 知识点数量（例如 18）
all_qs_ids = output_matrix.index  # 所有题目 ID

print(all_qs_ids)

# 找到 group_id=2 在 valid_groups 中的位置
start_index = np.where(valid_groups == 2)[0][0]  # 获取 group_id=2 的索引位置

# 逐组分析（只处理学生数大于 1 的组，并从 group_id=2 开始）
for group_id in tqdm(valid_groups):
    print(f"\n=== 处理组 {group_id} ===")

# # 逐组分析（只处理学生数大于 1 的组）
# for group_id in tqdm(valid_groups):
#     print(f"\n=== 处理组 {group_id} ===")

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
    max_knowledge = 10
    n_kno = min(max(1, n_questions // 5), max_knowledge, n_kno_total)  # 知识点数量限制
    group_qs_ids = X_group_df.columns  # 组内题目 ID

    Q = build_q_matrix(
        output_matrix=output_matrix,
        group_qs_ids=X_group_df.columns.tolist(),
        all_qs_ids=set(output_matrix.index),  # 转为集合加速查询
        max_knowledge=max_knowledge
)
    # 替换原 Q 矩阵生成代码


    # sparsity = 0.30  # 目标稀疏度（15% 的 1）
    #
    # Q = generate_random_q_matrix(
    #     n_knowledge=n_kno,
    #     n_questions=n_questions,
    #     sparsity=sparsity,
    #     min_knowledge_per_q=1,
    #     min_q_per_knowledge=5  # 每个知识点至少关联 5 题
    # )

    # 检查生成结果
    print("Q 矩阵形状:", Q.shape)
    print("1 的比例:", np.mean(Q))
    print("每知识点关联题数:", np.sum(Q, axis=1))
    print("每题关联知识点数:", np.sum(Q, axis=0))

    print(f"组 {group_id} - 生成的 Q 矩阵形状: {Q.shape} (知识点数限制为≤{max_knowledge})")

    # 打印 Q 矩阵（前 5 行，前 10 列）
    print(f"组 {group_id} - Q 矩阵 (前 5 行，前 10 列):")
    print(pd.DataFrame(Q, index=[f"K{i + 1}" for i in range(Q.shape[0])],
                       columns=X_group_df.columns[:n_questions]).iloc[:, :min(10, n_questions)])
    # 统计 Q 矩阵中 1 和 0 的数量
    n_ones = np.sum(Q == 1)
    n_zeros = np.sum(Q == 0)
    total_elements = Q.size
    print(f"\nQ 矩阵统计信息:")
    print(f"- 1 的数量: {n_ones} (占比: {n_ones / total_elements:.2%})")
    print(f"- 0 的数量: {n_zeros} (占比: {n_zeros / total_elements:.2%})")
    print(f"- 矩阵形状: {Q.shape} (知识点数: {Q.shape[0]}, 题目数: {Q.shape[1]})")

    # 初始化先验
    A_all = np.array(list(product([0, 1], repeat=Q.shape[0])))
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

        result = run_dina_self_consistency_test(group_id=group_id,
            X_group=X_group,
            Q=Q,
            prior=priors[i],
            d=2 * Q.shape[0] + 2 ** Q.shape[1],  # 模型自由度（大致），可选
            verbose=True)
        print(result["slip_error"], result["X_accuracy"], result["AIC"])

