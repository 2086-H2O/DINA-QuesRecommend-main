import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error
import os
from datetime import datetime
import tracemalloc # 导入内存跟踪库

# --- Matplotlib 配置 ---
try:
    # 尝试设置中文字体
    plt.rcParams['font.sans-serif'] = ['Noto Sans S Chinese', 'SimHei', 'Heiti TC']
    print("中文字体设置成功。")
except Exception as e:
    print(f"警告：设置中文字体失败 ({e})。绘图中文可能显示为方框。")
    
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

import matplotlib
matplotlib.use('Agg')  # 不弹出图片窗口，直接保存

# ==================================================================
#                       DINA 模型相关函数
# ==================================================================

def compute_eta(Q, A):
    """ 计算给定Q矩阵和所有属性模式A的eta矩阵 """
    kowns = np.sum(Q * Q, axis=0)
    cross = np.dot(A, Q)
    eta = np.ones(shape=(A.shape[0], Q.shape[1]))
    eta[cross < kowns] = 0
    return eta


def compute_propa(eta, s, g):
    """ 计算给定eta, slip, guess的答对概率 """
    propa = (g ** (1 - eta)) * ((1 - s) ** eta)
    propa[propa <= 0] = 1e-10
    propa[propa >= 1] = 1 - 1e-10
    return propa


def compute_gamma(X, pi, propa):
    """ E-step: 计算后验概率 gamma (学生i 掌握模式l 的概率) """
    log_pj = np.log(propa)
    log_qj = np.log(1 - propa)
    log_pi = np.log(pi)
    
    gamma = np.exp(np.dot(X, log_pj.T) + np.dot((1 - X), log_qj.T) + log_pi)
    gamma_sum = np.sum(gamma, axis=1)
    gamma_sum[gamma_sum == 0] = 1e-15
    gamma = (gamma.T / gamma_sum).T
    return gamma


def compute_theta(X, gamma, eta):
    """ M-step: 更新参数 pi, s, g """
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
    """ DINA 模型的 EM 估计算法 """
    n_stu = X.shape[0]
    n_qus = X.shape[1]
    n_kno = Q.shape[0]
    
    if n_kno == 0:
        print("EM 错误: 知识点数量为 0, 无法执行 EM。")
        return (np.array([1.0]), 
                np.full(n_qus, 0.5), 
                np.full(n_qus, 0.5), 
                np.ones((n_stu, 1)) / 1.0) 

    g = np.random.random(n_qus) * 0.25 + 0.1
    s = np.random.random(n_qus) * 0.25 + 0.1
    
    A_all = np.array(list(product([0, 1], repeat=n_kno)))
    
    if prior is None:
        pi = np.ones(shape=(A_all.shape[0])) / A_all.shape[0]
    else:
        if len(prior) != A_all.shape[0]:
            print(f"警告: 先验 'prior' 长度 ({len(prior)}) 与 A_all 长度 ({A_all.shape[0]}) 不匹配。使用均匀分布代替。")
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
    """ 根据后验概率 gamma 确定每个学生最可能的掌握模式 A """
    if n_kownlege == 0:
        return np.array([]), np.array([])
        
    A_all = np.array(list(product([0, 1], repeat=n_kownlege)))
    A_idx = np.argmax(gamma, axis=1)
    
    if not A_all.any():
        return np.array([]), A_idx
        
    return A_all[A_idx], A_idx


def joint_loglike(X, Q, s, g, pi,gamma):
    """ 计算联合对数似然 (用于 AIC) """
    if Q.shape[0] == 0:
        return 0
        
    A_all = np.array(list(product([0, 1], repeat=Q.shape[0])))
    eta = compute_eta(Q, A_all)
    propa = compute_propa(eta, s, g)
    log_pj = np.log(propa)
    log_qj = np.log(1 - propa)
    log_pi = np.log(pi)
    
    if len(log_pi) != A_all.shape[0]:
         print(f"Loglike 警告: log_pi 长度 {len(log_pi)} 与 A_all 长度 {A_all.shape[0]} 不匹配")
         return 0
    
    L = np.dot(X, log_pj.T) + np.dot((1 - X), log_qj.T) + log_pi
    L = L * gamma
    
    return np.sum(L)


def get_priors(A_all, p_know, p_know_list):
    """ 生成三种不同的先验分布 """
    n_patterns = A_all.shape[0]
    prior1 = np.ones(n_patterns) / n_patterns # P1: 均匀分布
    prior2 = np.ones(n_patterns) # P2: 独立同分布
    prior3 = np.ones(n_patterns) # P3: 独立不同分布
    
    n_kno = A_all.shape[1]
    
    if n_kno == 0:
        return [np.array([1.0]), np.array([1.0]), np.array([1.0]), None]

    if len(p_know_list) < n_kno:
        p_know_list = p_know_list * (n_kno // len(p_know_list) + 1)
    p_know_list = p_know_list[:n_kno]
    
    for l in range(n_patterns): 
        pattern = A_all[l, :] 
        for k in range(n_kno): 
            p = p_know_list[k]
            prior2[l] *= (p_know ** pattern[k] * (1 - p_know) ** (1 - pattern[k]))
            prior3[l] *= (p ** pattern[k] * (1 - p) ** (1 - pattern[k]))
            
    prior2 /= np.sum(prior2)
    prior3 /= np.sum(prior3)
    
    return [prior1, prior2, prior3, None] 


# ==================================================================
#                       Q 矩阵构建
# ==================================================================

def build_q_matrix(output_matrix, group_qs_ids, all_qs_ids, max_knowledge):
    """
    优化版Q矩阵构建函数
    返回:
        Q: 优化后的Q矩阵 (知识点×题目)
        selected_names: Q矩阵中行对应的知识点名称 (pd.Index)
    """
    
    # --- 类型转换 ---
    try:
        if output_matrix.index.dtype == 'int64':
            all_qs_ids_typed = {int(q) for q in all_qs_ids}
            group_qs_ids_typed = [int(q) for q in group_qs_ids]
            output_matrix_index_set = set(output_matrix.index)
        elif output_matrix.index.dtype == 'object':
             all_qs_ids_typed = {str(q) for q in all_qs_ids}
             group_qs_ids_typed = [str(q) for q in group_qs_ids]
             output_matrix_index_set = set(output_matrix.index.astype(str))
        else:
             all_qs_ids_typed = set(all_qs_ids)
             group_qs_ids_typed = list(group_qs_ids)
             output_matrix_index_set = set(output_matrix.index)
    except Exception as e:
        print(f"类型转换警告: {e}。回退到原始类型。")
        all_qs_ids_typed = set(all_qs_ids)
        group_qs_ids_typed = list(group_qs_ids)
        output_matrix_index_set = set(output_matrix.index)

    # --- 步骤1: 筛选有效知识点 (基于全局覆盖率) ---
    valid_indices_in_output = [idx for idx in output_matrix.index if idx in all_qs_ids_typed]
    
    if not valid_indices_in_output:
        print(f"警告: output_matrix 中没有找到任何有效的题目 ID。返回空 Q 矩阵 (max_k={max_knowledge})。")
        return np.empty((0, len(group_qs_ids)), dtype=int), pd.Index([])
        
    # 注意：这里的 .sum() 是在传入的 (可能已被过滤的) output_matrix 上计算的
    knowledge_coverage = output_matrix.loc[valid_indices_in_output].sum(axis=0).values

    top_k_indices = np.argsort(-knowledge_coverage)[:max_knowledge]
    top_k_indices = sorted(top_k_indices)  # 保持原始列索引顺序

    selected_names = output_matrix.columns[top_k_indices]

    # --- 步骤2: 构建Q矩阵 (知识点 x 题目) ---
    Q = np.zeros((len(top_k_indices), len(group_qs_ids_typed)), dtype=int)
    missing_qs = 0
    
    for j, qs_id in enumerate(group_qs_ids_typed): 
        if qs_id in output_matrix_index_set: 
            # 确保 loc 使用原始索引类型
            Q[:, j] = output_matrix.loc[qs_id].values[top_k_indices]
        else:
            missing_qs += 1
            Q[:, j] = 0  

    if missing_qs > 0:
        print(f"(k={max_knowledge}) 警告: {missing_qs} 个题目未在output_matrix中找到。")

    # --- 步骤3: 后处理验证 ---
    if Q.size == 0:
        print(f"(k={max_knowledge}) Q矩阵为空。")
        return Q, pd.Index([])
        
    non_zero_knowledge_rows = np.where(Q.sum(axis=1) > 0)[0]
    Q = Q[non_zero_knowledge_rows, :]
    selected_names = selected_names[non_zero_knowledge_rows]

    if Q.shape[0] == 0:
        print(f"(k={max_knowledge}) 后处理: 0 个有效知识点。")
        return Q, selected_names 

    valid_knowledge_rows = np.where(Q.sum(axis=1) >= 3)[0]
    if len(valid_knowledge_rows) < Q.shape[0]:
        print(f"(k={max_knowledge}) 优化: 移除 {Q.shape[0] - len(valid_knowledge_rows)} 个低覆盖(<3)知识点。")
        Q = Q[valid_knowledge_rows, :]
        selected_names = selected_names[valid_knowledge_rows]

    print(f"(k={max_knowledge}) 最终Q矩阵形状: {Q.shape} (知识点×题目)")
    
    return Q, selected_names

# ==================================================================
#                     自洽性检验函数
# ==================================================================

def run_dina_self_consistency_test(
    group_id, X_group, Q, prior, d=None,
    maxIter=1000, tol=1e-6, verbose=True
):
    """
    针对某一组学生数据，运行 DINA 模型的自洽性检验流程
    返回结构化指标
    """

    if verbose:
        print(f"\n===== Group {group_id} - DINA 自洽性检验 =====")

    n_kno = Q.shape[0]
    if n_kno == 0:
        print("错误: Q 矩阵知识点为 0, 中止自洽性检验。")
        return {
            "group_id": group_id, "slip_error": np.nan, "guess_error": np.nan,
            "pi_error": np.nan, "X_accuracy": np.nan, "bce_loss": np.nan,
            "NLL": np.nan, "AIC": np.nan
        }

    # Step 1: 拟合原始数据
    if verbose: print("1. EM 拟合原始数据...")
    pi, g, s, gamma = em(X_group, Q, maxIter=maxIter, tol=tol, prior=prior)
    A, A_idx = solve(gamma, n_kno)
    
    if A.size == 0:
        print("错误: 原始数据拟合失败 (A 矩阵为空)。")
        return { "group_id": group_id, "slip_error": np.nan, "guess_error": np.nan,
                 "pi_error": np.nan, "X_accuracy": np.nan, "bce_loss": np.nan,
                 "NLL": np.nan, "AIC": np.nan }

    # Step 2: 生成模拟数据
    eta = compute_eta(Q, A) 
    propa = compute_propa(eta, s, g)
    X_sim = np.random.binomial(1, propa) 

    # Step 3: 用模拟数据重新拟合（回归）
    if verbose: print("2. EM 拟合模拟数据 (回归)...")
    pi_sim, g_sim, s_sim, gamma_sim = em(X_sim, Q, maxIter=maxIter, tol=tol, prior=prior)

    # Step 4: 评估参数误差
    slip_error = np.mean(np.abs(s - s_sim))
    guess_error = np.mean(np.abs(g - g_sim))
    pi_error = np.mean(np.abs(pi - pi_sim))

    # Step 5: 评估模拟答题矩阵与原始数据的一致性
    acc = accuracy_score(X_group.flatten(), X_sim.flatten())
    bce = log_loss(X_group.flatten(), propa.flatten(), labels=[0,1])

    # Step 6: 联合似然与 AIC
    LL = joint_loglike(X_group, Q, s, g, pi, gamma)
    NLL = -LL
    AIC_value = -2 * LL + d if d is not None else None

    # Step 7: 解释性评估输出
    if verbose:
        print(f"[Group {group_id}] 参数误差：")
        print(f"  - slip 平均误差：{slip_error:.4f}")
        print(f"  - guess 平均误差：{guess_error:.4f}")
        print(f"  - π 平均误差：{pi_error:.4f}")
        print(f"[Group {group_id}] 模拟数据拟合能力：")
        print(f"  - 准确率 (X_orig vs X_sim)：{acc:.4f}")
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
        "AIC": AIC_value
    }


# ==================================================================
#                       主实验脚本 (修改版)
# ==================================================================

def main():
    try:
        # --- 1. 加载数据 ---
        print("加载数据...")
        data = pd.read_csv(r"cleaned_data_20250326_0931.csv")
        group_data = pd.read_csv('optimal_student_groups_leiden.csv')
        
        output_matrix = pd.read_excel('516matrix.xlsx', index_col=0)
        
        print(f"Q矩阵 (output_matrix) 加载完毕，形状: {output_matrix.shape}")

        # --- 2. 准备实验组 (学生数据) ---
        
        try:
            output_matrix_original_index = output_matrix.index.astype(str)
            output_matrix.index = output_matrix_original_index
            data['qs_id'] = data['qs_id'].astype(str)
            print("统一索引和qs_id为 str 类型")
        except Exception as e:
            print(f"类型转换失败: {e}。后续匹配可能出错。")
            
        # 注意：all_qs_ids 应始终基于 *原始* Q矩阵的 *行索引*
        all_qs_ids = set(output_matrix.index) 
        
        group_counts = group_data['group'].value_counts()
        valid_groups = group_counts[group_counts > 1].index
        
        if len(valid_groups) == 0:
            print("错误：没有找到学生数大于1的组，无法继续实验。")
            return

        # 锁定第一组学生
        first_group_id = valid_groups[0]
        print(f"\n=== 锁定实验组: {first_group_id} (学生数: {group_counts[first_group_id]}) ===")

        group_students = group_data[group_data['group'] == first_group_id]['student_id'].values
        group_data_subset = data[data['student_id'].isin(group_students)].drop_duplicates(subset=['student_id', 'qs_id'])
        
        X_group_df = group_data_subset.pivot(index='student_id', columns='qs_id', values='qs_validity').fillna(0)
        X_group = X_group_df.values
        group_qs_ids = X_group_df.columns.tolist() 
        
        n_stu = X_group.shape[0]
        n_qus = X_group.shape[1]
        print(f"组 {first_group_id} - 学生数 (N): {n_stu}, 题目数 (J): {n_qus}")


        # --- 3. K 值对比实验 (k=15) ---
        
        MAX_K_EXPERIMENT = 15
        
        # 定义实验组要删除的知识点列表
        drop_lists = [
            [], # 组 0: 对照组
            ["输出", "连接"], # 组 1: 轻度筛选
            ["输出", "连接", "电源供电", "电压信号"] # 组 2: 中度筛选
        ]

        print(f"\n开始 K={MAX_K_EXPERIMENT} 的对比实验...")

        for i, drop_list in enumerate(drop_lists):
            print(f"\n========================================================")
            print(f"                实验组 {i} (max_k = {MAX_K_EXPERIMENT})")
            print(f"========================================================")
            
            # 1. 准备 Q 矩阵 (根据实验组定义)
            
            # 从原始 Q 矩阵中删除指定的列（知识点）
            # errors='ignore' 确保即使知识点不存在（或已被删除）也不会报错
            try:
                filtered_output_matrix = output_matrix.drop(columns=drop_list, errors='ignore')
                print(f"实验组 {i}: 移除了 {len(drop_list)} 个知识点。")
                if drop_list:
                    print(f"   > {', '.join(drop_list)}")
            except Exception as e:
                print(f"移除知识点时出错: {e}。使用原始 Q 矩阵。")
                filtered_output_matrix = output_matrix

            # 2. 构建 Q 矩阵 (使用被过滤的 output_matrix)
            Q, selected_kp_names = build_q_matrix(
                output_matrix=filtered_output_matrix, # 使用过滤后的矩阵
                group_qs_ids=group_qs_ids,
                all_qs_ids=all_qs_ids, # all_qs_ids 保持不变 (它基于行)
                max_knowledge=MAX_K_EXPERIMENT
            )
            
            n_kno = Q.shape[0] # 实际使用的知识点数

            # 3. 打印本轮实验的详细信息 (按您要求)
            print("\n--- [组 {i}] 实验参数 ---")
            print(f"学生数量 (N): {n_stu}")
            print(f"题目数量 (J): {n_qus}")
            print(f"知识点数量 (K): {n_kno}")
            print(f"知识点详细组成 ({len(selected_kp_names)} 个):")
            if not selected_kp_names.empty:
                kp_print_series = pd.Series(selected_kp_names.values, index=range(1, len(selected_kp_names) + 1))
                print(kp_print_series.to_string())
            else:
                print("(无有效知识点)")
            print("-" * 40)
            
            # 4. 运行 EM 算法
            if n_kno == 0:
                print(f"警告: 组 {i} (max_k={MAX_K_EXPERIMENT}) 没有有效的知识点被保留。跳过此实验组。")
                continue
            if n_kno > 20: 
                 print(f"警告: n_kno = {n_kno}。状态空间过大，跳过此实验组。")
                 continue
            
            # 初始化先验 (P2)
            A_all = np.array(list(product([0, 1], repeat=n_kno)))
            priors = get_priors(A_all, p_know=0.7, p_know_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            prior_to_use = priors[1]
            d_model = 2 * Q.shape[1] 

            # 运行自洽性检验
            result = run_dina_self_consistency_test(
                group_id=first_group_id,
                X_group=X_group,
                Q=Q,
                prior=prior_to_use,
                d=d_model,
                verbose=False # 减少冗余输出
            )
            
            # 5. 打印本轮实验的结果指标 (按您要求)
            print(f"\n--- [组 {i}] 拟合结果 ---")
            print(f"Slip Error (s - s_sim):   {result.get('slip_error', np.nan):.6f}")
            print(f"Guess Error (g - g_sim):  {result.get('guess_error', np.nan):.6f}")
            print(f"Pi Error (π - π_sim):     {result.get('pi_error', np.nan):.6f}")
            print(f"Accuracy (X_orig vs X_sim): {result.get('X_accuracy', np.nan):.6f}")
            print(f"BCE Loss:                 {result.get('bce_loss', np.nan):.6f}")
            print(f"NLL (Negative Log-Like):  {result.get('NLL', np.nan):.2f}")
            print(f"AIC:                      {result.get('AIC', np.nan):.2f}")
            print(f"========================================================\n")

        print("\n=== 所有实验组运行完毕 ===")
        
        # (移除了原有的绘图和结果汇总代码)

    except FileNotFoundError as e:
        print(f"\n!!! 文件未找到错误: {e}")
        print("!!! 请确保 'cleaned_data_20250326_0931.csv', 'optimal_student_groups_leiden.csv', 和 '516matrix.xlsx' 文件位于同一目录中。")
    except Exception as e:
        print(f"\n!!! 发生意外错误: {e}")
        import traceback
        traceback.print_exc()

# 运行主函数
if __name__ == "__main__":
    main()