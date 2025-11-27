"""
测试第三个LLM生成的Q矩阵
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from sklearn.metrics import accuracy_score, f1_score, log_loss
import os
import tracemalloc

# ==========================================
#               1. 实验配置区
# ==========================================

# 在这里配置要跑哪些组 (索引)
# [0] 代表只跑 valid_groups 中的第1个组 (通常是 Group 0)
# [0, 1, 2] 代表跑前3个组
TARGET_GROUP_INDICES = [0,1,2] 

# 新 Q 矩阵的文件名
NEW_Q_MATRIX_FILE = r"C:\Users\User\Documents\Developer 2086\DINA-QuesRecommend-main\LLM_Q_Generate\outputs\4+10_2_results\DINA_Q_Matrix_4+10_2.xlsx"  # 请确保文件在此路径
# 如果是 Excel 文件，请改为 pd.read_excel

# ==========================================
#             2. DINA 模型核心函数
#           (复用之前逻辑，保持不变)
# ==========================================

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
    gamma_sum[gamma_sum == 0] = 1e-15
    gamma = (gamma.T / gamma_sum).T
    return gamma

def compute_theta(X, gamma, eta):
    I0 = np.dot(gamma, 1 - eta)
    I1 = np.dot(gamma, eta)
    R0 = I0 * X
    R1 = I1 * X
    I0_sum = np.sum(I0, axis=0)
    I1_sum = np.sum(I1, axis=0)
    R0_sum = np.sum(R0, axis=0)
    R1_sum = np.sum(R1, axis=0)
    I0_sum[I0_sum <= 0] = 1e-15
    I1_sum[I1_sum <= 0] = 1e-15
    g = R0_sum / I0_sum
    s = (I1_sum - R1_sum) / I1_sum
    pi = np.sum(gamma, axis=0) / gamma.shape[0]
    pi[pi <= 0] = 1e-15
    pi[pi >= 1] = 1 - 1e-15
    g[g < 0] = 0
    g[g > 1] = 1
    s[s < 0] = 0
    s[s > 1] = 1
    return pi, s, g

def em(X, Q, maxIter=1000, tol=1e-3, prior=None):
    n_stu, n_qus = X.shape
    n_kno = Q.shape[0]
    
    if n_kno == 0:
        return (np.array([1.0]), np.full(n_qus, 0.5), np.full(n_qus, 0.5), np.ones((n_stu, 1)))

    g = np.random.random(n_qus) * 0.25 + 0.1
    s = np.random.random(n_qus) * 0.25 + 0.1
    A_all = np.array(list(product([0, 1], repeat=n_kno)))
    
    if prior is None:
        pi = np.ones(A_all.shape[0]) / A_all.shape[0]
    else:
        pi = prior if len(prior) == A_all.shape[0] else np.ones(A_all.shape[0]) / A_all.shape[0]
            
    for t in range(maxIter):
        eta = compute_eta(Q, A_all)
        propa = compute_propa(eta, s, g)
        gamma = compute_gamma(X, pi, propa)
        pi_t, s_t, g_t = compute_theta(X, gamma, eta)
        
        update = max(np.max(np.abs(pi_t - pi)), np.max(np.abs(g_t - g)), np.max(np.abs(s_t - s)))
        if update < tol:
            return pi_t, g_t, s_t, gamma
        if prior is None: pi = pi_t
        g, s = g_t, s_t
        
    return pi, g, s, gamma

def solve(gamma, n_kownlege):
    if n_kownlege == 0: return np.array([]), np.array([])
    A_all = np.array(list(product([0, 1], repeat=n_kownlege)))
    A_idx = np.argmax(gamma, axis=1)
    return A_all[A_idx], A_idx

def joint_loglike(X, Q, s, g, pi, gamma):
    if Q.shape[0] == 0: return 0
    A_all = np.array(list(product([0, 1], repeat=Q.shape[0])))
    eta = compute_eta(Q, A_all)
    propa = compute_propa(eta, s, g)
    # 避免 log(0)
    propa[propa <= 0] = 1e-15
    propa[propa >= 1] = 1 - 1e-15
    
    log_pj = np.log(propa)
    log_qj = np.log(1 - propa)
    log_pi = np.log(pi)
    
    L = np.dot(X, log_pj.T) + np.dot((1 - X), log_qj.T) + log_pi
    L = L * gamma
    return np.sum(L)

def get_priors(A_all, p_know, p_know_list):
    n_patterns = A_all.shape[0]
    prior2 = np.ones(n_patterns)
    n_kno = A_all.shape[1]
    
    if len(p_know_list) < n_kno:
        p_know_list = p_know_list * (n_kno // len(p_know_list) + 1)
    p_know_list = p_know_list[:n_kno]
    
    for l in range(n_patterns): 
        pattern = A_all[l, :] 
        for k in range(n_kno): 
            p = p_know_list[k]
            prior2[l] *= (p_know ** pattern[k] * (1 - p_know) ** (1 - pattern[k]))
    prior2 /= np.sum(prior2)
    return prior2  # 只返回 Prior 2 (独立同分布)

# ==========================================
#             3. 核心工具函数
# ==========================================

def preprocess_llm_q_matrix(file_path):
    """
    读取并预处理 LLM 生成的 Q 矩阵
    假设格式: id, qs_title, section_id, section_name, K1_xxx, K2_xxx ...
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # 1. 确保 id 是索引
        if 'id' in df.columns:
            df['id'] = df['id'].astype(str)
            df = df.set_index('id')
        else:
            print("警告: 新矩阵中没找到 'id' 列，尝试使用第一列作为索引...")
            df.index = df.iloc[:, 0].astype(str)

        # 2. 自动提取知识点列 (以 'K' 开头的列)
        k_columns = [col for col in df.columns if col.strip().upper().startswith('K')]
        
        if not k_columns:
            raise ValueError("未在新矩阵中找到以 'K' 开头的知识点列 (例如 K1_仪器操作)")
            
        print(f"检测到 {len(k_columns)} 个知识点列: {k_columns}")
        
        # 3. 提取 Q 矩阵部分 (只保留 0/1)
        q_df = df[k_columns].fillna(0).astype(int)
        
        return q_df, k_columns
        
    except Exception as e:
        print(f"读取新 Q 矩阵失败: {e}")
        return None, None

def build_fixed_q_matrix(full_q_df, group_qs_ids):
    """
    根据组内题目 (group_qs_ids) 从总 Q 矩阵 (full_q_df) 中切片
    """
    # 确保索引类型一致: 统统转为字符串
    group_qs_ids_str = [str(q) for q in group_qs_ids]
    # 确保 Q 矩阵索引也是字符串
    full_q_df.index = full_q_df.index.astype(str)
    available_qs = set(full_q_df.index)
    
    # 找出交集题目 (Q矩阵里有的，且学生做过的)
    valid_qs = [q for q in group_qs_ids_str if q in available_qs]
    
    if not valid_qs:
        return None, []
    
    # 切片: 行=题目(valid_qs), 列=知识点
    subset_df = full_q_df.loc[valid_qs]
    
    # 转置: 行=知识点, 列=题目
    Q_matrix = subset_df.values.T 
    
    print(f"组内题目数: {len(group_qs_ids)} -> Q矩阵匹配到的题目数: {len(valid_qs)}")
    
    # === 修改点：返回 Q 矩阵 和 题目ID列表 ===
    return Q_matrix, valid_qs

# ==========================================
#               4. 主程序
# ==========================================

def main():
    print(">>> 开始 1124TEST: 新 LLM Q 矩阵测试")
    
    # 1. 加载学生数据
    try:
        data = pd.read_csv("cleaned_data_20250326_0931.csv")
        data['qs_id'] = data['qs_id'].astype(str)
        group_data = pd.read_csv("optimal_student_groups_leiden.csv")
        print("学生答题数据加载完毕.")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 2. 加载新 Q 矩阵
    full_q_df, k_names = preprocess_llm_q_matrix(NEW_Q_MATRIX_FILE)
    if full_q_df is None:
        return
    
    n_kno_total = len(k_names)
    print(f"新 Q 矩阵加载完毕. 总题目数: {len(full_q_df)}, 知识点数: {n_kno_total}")
    
    # 3. 准备组信息
    group_counts = group_data['group'].value_counts()
    # 过滤只有1个学生的组
    valid_groups = group_counts[group_counts > 1].index.tolist()
    
    # 4. 循环测试指定的组
    print(f"\n即将测试的组索引: {TARGET_GROUP_INDICES}")
    
    for idx in TARGET_GROUP_INDICES:
        if idx >= len(valid_groups):
            print(f"跳过索引 {idx}: 超出有效组数量 ({len(valid_groups)})")
            continue
            
        group_id = valid_groups[idx]
        print(f"\n========================================")
        print(f"正在测试 Group ID: {group_id} (索引 {idx})")
        print(f"学生数量: {group_counts[group_id]}")
        print(f"========================================")
        
        # 4.1 提取组数据
        group_students = group_data[group_data['group'] == group_id]['student_id'].values
        group_data_subset = data[data['student_id'].isin(group_students)].drop_duplicates(subset=['student_id', 'qs_id'])
        
        # 4.2 构建 X 矩阵 (学生 x 题目)
        X_df = group_data_subset.pivot(index='student_id', columns='qs_id', values='qs_validity').fillna(0)
        X = X_df.values
        group_qs_ids = X_df.columns.tolist()
        
        # 4.3 构建适配该组的 Q 矩阵
        Q, final_qs_list = build_fixed_q_matrix(full_q_df, group_qs_ids)
        
        if Q is None or len(final_qs_list) == 0:
            print("该组题目与 Q 矩阵无交集，跳过。")
            continue
            
        # Q 形状: (n_kno, n_qs_match)
        # X 形状需对齐: (n_stu, n_qs_match)
        
        # === 关键修改：直接使用 Q 矩阵确定的题目列表来切片 X ===
        # 确保 X_df 的列名也是字符串，以匹配 final_qs_list
        X_df.columns = X_df.columns.astype(str)
        
        # 只保留 Q 矩阵中存在的题目
        X_aligned = X_df[final_qs_list].values
        
        print(f"数据对齐后: 学生 N={X_aligned.shape[0]}, 题目 J={X_aligned.shape[1]}, 知识点 K={Q.shape[0]}")
        
        # 二次检查维度 (防止报错)
        if Q.shape[1] != X_aligned.shape[1]:
            print(f"❌ 严重错误: 维度依然不匹配! Q={Q.shape}, X={X_aligned.shape}")
            continue

        # 4.4 准备先验 (独立同分布 P2)
        n_kno = Q.shape[0]
        
        # 安全检查: 如果 K > 14 (比如 LLM 生成了太多), 可能会爆内存
        # 您的新矩阵是 14 个点， 2^14 = 16384，这在任何机器上都非常快且安全
        if n_kno > 20:
            print(f"警告: K={n_kno} 过大，跳过...")
            continue
            
        A_all = np.array(list(product([0, 1], repeat=n_kno)))
        prior = get_priors(A_all, p_know=0.6, p_know_list=[0.6]*n_kno) # 默认 0.6
        
        # 4.5 运行 DINA 自洽性检验
        # Step 1: 拟合
        print("1. EM 拟合原始数据...")
        pi, g, s, gamma = em(X_aligned, Q, prior=prior)
        A, A_idx = solve(gamma, n_kno)
        
        # Step 2: 模拟
        print("2. 生成模拟数据并回归...")
        eta = compute_eta(Q, A)
        propa = compute_propa(eta, s, g)
        X_sim = np.random.binomial(1, propa)
        
        pi_sim, g_sim, s_sim, gamma_sim = em(X_sim, Q, prior=prior)
        
        # Step 3: 计算指标
        acc = accuracy_score(X_aligned.flatten(), X_sim.flatten())
        
        # 避免 log 错误
        propa_safe = propa.copy()
        propa_safe[propa_safe <= 0] = 1e-15
        propa_safe[propa_safe >= 1] = 1 - 1e-15
        bce = log_loss(X_aligned.flatten(), propa_safe.flatten(), labels=[0,1])
        
        LL = joint_loglike(X_aligned, Q, s, g, pi, gamma)
        AIC = -2 * LL + 2 * X_aligned.shape[1] # standard AIC definition
        
        slip_err = np.mean(np.abs(s - s_sim))
        guess_err = np.mean(np.abs(g - g_sim))
        
        # 4.6 打印结果
        print(f"\n>>> [Group {group_id}] 测试结果 <<<")
        print(f"{'指标':<20} {'数值':<15}")
        print("-" * 35)
        print(f"{'Accuracy':<20} {acc:.5f}")
        print(f"{'AIC':<20} {AIC:.2f}")
        print(f"{'BCE Loss':<20} {bce:.5f}")
        print(f"{'Slip Error':<20} {slip_err:.5f}")
        print(f"{'Guess Error':<20} {guess_err:.5f}")
        print("-" * 35)
        
        # 简单评价
        if acc > 0.79:
            print("评价: 优秀！拟合度接近或超过 Baseline (0.80)。")
        elif acc > 0.75:
            print("评价: 良好。新矩阵表现稳定。")
        else:
            print("评价: 一般。可能新知识点定义与学生实际答题行为有偏差。")

if __name__ == "__main__":
    main()