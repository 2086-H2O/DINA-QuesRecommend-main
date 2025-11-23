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
    参数:
        ...
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
        # *** 修改点 1: 返回空名称 ***
        return np.empty((0, len(group_qs_ids)), dtype=int), pd.Index([])
        
    knowledge_coverage = output_matrix.loc[valid_indices_in_output].sum(axis=0).values

    top_k_indices = np.argsort(-knowledge_coverage)[:max_knowledge]
    top_k_indices = sorted(top_k_indices)  # 保持原始顺序

    # *** 修改点 2: 在过滤前获取知识点名称 ***
    selected_names = output_matrix.columns[top_k_indices]

    # --- 步骤2: 构建Q矩阵 (知识点 x 题目) ---
    Q = np.zeros((len(top_k_indices), len(group_qs_ids_typed)), dtype=int)
    missing_qs = 0
    
    for j, qs_id in enumerate(group_qs_ids_typed): 
        if qs_id in output_matrix_index_set: 
            Q[:, j] = output_matrix.loc[qs_id].values[top_k_indices]
        else:
            missing_qs += 1
            Q[:, j] = 0  

    if missing_qs > 0:
        print(f"(k={max_knowledge}) 警告: {missing_qs} 个题目未在output_matrix中找到。")

    # --- 步骤3: 后处理验证 ---
    if Q.size == 0:
        print(f"(k={max_knowledge}) Q矩阵为空。")
        # *** 修改点 3: 返回空名称 ***
        return Q, pd.Index([])
        
    non_zero_knowledge_rows = np.where(Q.sum(axis=1) > 0)[0]
    Q = Q[non_zero_knowledge_rows, :]
    # *** 修改点 4: 同步过滤名称 ***
    selected_names = selected_names[non_zero_knowledge_rows]

    if Q.shape[0] == 0:
        print(f"(k={max_knowledge}) 后处理: 0 个有效知识点。")
        # *** 修改点 5: 返回空的名称 ***
        return Q, selected_names # selected_names 在这里是空的

    valid_knowledge_rows = np.where(Q.sum(axis=1) >= 3)[0]
    if len(valid_knowledge_rows) < Q.shape[0]:
        print(f"(k={max_knowledge}) 优化: 移除 {Q.shape[0] - len(valid_knowledge_rows)} 个低覆盖(<3)知识点。")
        Q = Q[valid_knowledge_rows, :]
        # *** 修改点 6: 再次同步过滤名称 ***
        selected_names = selected_names[valid_knowledge_rows]

    print(f"(k={max_knowledge}) 最终Q矩阵形状: {Q.shape} (知识点×题目)")
    
    # *** 修改点 7: 返回最终的 Q 和 名称 ***
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
#                       主实验脚本
# ==================================================================

def main():
    try:
        # --- 1. 加载数据 ---
        print("加载数据...")
        data = pd.read_csv(r"cleaned_data_20250326_0931.csv")
        group_data = pd.read_csv('optimal_student_groups_leiden.csv')
        
        # *** 修改点：使用 read_excel ***
        output_matrix = pd.read_excel('516matrix.xlsx', index_col=0)
        
        print(f"Q矩阵 (output_matrix) 加载完毕，形状: {output_matrix.shape}")

        # --- 2. 准备实验组 ---
        
        try:
            output_matrix.index = output_matrix.index.astype(str)
            data['qs_id'] = data['qs_id'].astype(str)
            print("统一索引和qs_id为 str 类型")
        except Exception as e:
            print(f"类型转换失败: {e}。后续匹配可能出错。")
            
        all_qs_ids = set(output_matrix.index) # Q矩阵中的所有题目ID
        n_kno_total = output_matrix.shape[1] # 总知识点数
        print(f"总知识点数 (K_total): {n_kno_total}")
        
        # *** (移除了之前的 Top-K 打印代码) ***
        
        group_counts = group_data['group'].value_counts()
        valid_groups = group_counts[group_counts > 1].index
        
        if len(valid_groups) == 0:
            print("错误：没有找到学生数大于1的组，无法继续实验。")
            return

        # 只使用第一组
        first_group_id = valid_groups[0]
        print(f"\n=== 锁定实验组: {first_group_id} (学生数: {group_counts[first_group_id]}) ===")

        group_students = group_data[group_data['group'] == first_group_id]['student_id'].values
        group_data_subset = data[data['student_id'].isin(group_students)].drop_duplicates(subset=['student_id', 'qs_id'])
        
        X_group_df = group_data_subset.pivot(index='student_id', columns='qs_id', values='qs_validity').fillna(0)
        X_group = X_group_df.values
        group_qs_ids = X_group_df.columns.tolist() 
        
        print(f"组 {first_group_id} - 学生数: {X_group.shape[0]}, 题目数: {X_group.shape[1]}")

        # --- 3. K 值循环实验 ---

        plot_x_axis = [] 
        actual_k_used = [] 
        accuracies = []
        slip_errors = []
        guess_errors = []
        pi_errors = []
        aics = []
        memory_peaks = [] 
        
        k_range = range(8, 32)
        
        print(f"开始循环 max_knowledge 从 {k_range.start} 到 {k_range.stop - 1}")
        
        tracemalloc.start() # 开始内存跟踪

        for max_k in tqdm(k_range):
            
            # 1. 构建 Q 矩阵
            # *** 修改点 8: 接收返回的知识点名称 ***
            Q, selected_kp_names = build_q_matrix(
                output_matrix=output_matrix,
                group_qs_ids=group_qs_ids,
                all_qs_ids=all_qs_ids,
                max_knowledge=max_k
            )
            
            # *** 修改点 9: 在循环内打印本轮使用的知识点 ***
            print(f"\n--- [max_k = {max_k}] 正在使用的 {len(selected_kp_names)} 个知识点 (过滤后 n_kno = {Q.shape[0]}) ---")
            if not selected_kp_names.empty:
                # 使用 1-based 索引打印
                kp_print_series = pd.Series(selected_kp_names.values, index=range(1, len(selected_kp_names) + 1))
                print(kp_print_series.to_string())
            else:
                print("(无有效知识点)")
            print("-" * (50 + len(str(max_k))))
            # *** 打印代码结束 ***

            n_kno = Q.shape[0] # 实际使用的知识点数
            
            # # 如果 Q 矩阵过滤后没有知识点，或知识点过多导致无法计算 (2^K)
            # if n_kno == 0:
            #     print(f"警告: max_k={max_k} 时，没有有效的知识点被保留。跳过此迭代。")
            #     continue
            # if n_kno > 20: 
            #      print(f"警告: n_kno = {n_kno} (当 max_k={max_k})。状态空间 (2^{n_kno}) 过大，跳过此迭代。")
            #      continue
            
            # # 2. 初始化先验
            # A_all = np.array(list(product([0, 1], repeat=n_kno)))
            # priors = get_priors(A_all, p_know=0.7, p_know_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            
            # # 只使用第二种先验 (priors[1])
            # prior_to_use = priors[1]
            # d_model = 2 * Q.shape[1] 

            # 3. 运行自洽性检验
            # result = run_dina_self_consistency_test(
            #     group_id=first_group_id,
            #     X_group=X_group,
            #     Q=Q,
            #     prior=prior_to_use,
            #     d=d_model,
            #     verbose=False # 循环中减少输出
            # )
            
            # print(f"max_k={max_k} (n_kno={n_kno}) 完成。Acc: {result.get('X_accuracy', np.nan):.4f}, AIC: {result.get('AIC', np.nan):.2f}")

            # # 4. 收集指标
            # plot_x_axis.append(max_k) 
            # actual_k_used.append(n_kno) 
            # accuracies.append(result.get('X_accuracy', np.nan))
            # slip_errors.append(result.get('slip_error', np.nan))
            # guess_errors.append(result.get('guess_error', np.nan))
            # pi_errors.append(result.get('pi_error', np.nan))
            # aics.append(result.get('AIC', np.nan))
            
            # current, peak = tracemalloc.get_traced_memory()
            # memory_peaks.append(peak / (1024 * 1024)) # 转换为 MB
            

        # tracemalloc.stop() # 停止内存跟踪
        # print("\n=== 实验循环完成 ===")
        
        # if not plot_x_axis:
        #     print("错误：没有一次迭代成功运行，无法生成图表。")
        #     return

        # # --- 4. 创建结果图表 ---
        
        # print("正在生成结果图表...")
        
        # (如果您添加了 OUTPUT_DIR，请确保这里的 savefig 路径正确)
        # 示例: save_path = os.path.join(OUTPUT_DIR, "accuracy_vs_max_k.png")
        
        # x_ticks = plot_x_axis
        # x_labels = [f"{max_k}\n(k={act_k})" for max_k, act_k in zip(plot_x_axis, actual_k_used)]

        # 图 1: 准确率 (X_accuracy)
        # plt.figure(figsize=(12, 7))
        # plt.plot(plot_x_axis, accuracies, marker='o', linestyle='-')
        # plt.xlabel("最大知识点数 (max_knowledge) \n (括号内为过滤后实际知识点 k)")
        # plt.ylabel("模拟准确率 (X_accuracy)")
        # plt.title(f"组 {first_group_id} - 模拟准确率 vs Max Knowledge (先验 P2)")
        # plt.grid(True, linestyle='--', alpha=0.6)
        # plt.xticks(ticks=x_ticks, labels=x_labels, rotation=45, ha='right')
        # plt.tight_layout()
        # plt.savefig("accuracy_vs_max_k.png")
        # plt.close()
        # print("已保存: accuracy_vs_max_k.png")

        # # 图 2: 参数误差 (slip_error, guess_error, pi_error)
        # plt.figure(figsize=(12, 7))
        # plt.plot(plot_x_axis, slip_errors, marker='s', linestyle='--', label='Slip Error (s - s_sim)')
        # plt.plot(plot_x_axis, guess_errors, marker='^', linestyle='--', label='Guess Error (g - g_sim)')
        # plt.plot(plot_x_axis, pi_errors, marker='x', linestyle='--', label='Pi Error (π - π_sim)')
        # plt.xlabel("最大知识点数 (max_knowledge) \n (括号内为过滤后实际知识点 k)")
        # plt.ylabel("平均参数绝对误差")
        # plt.title(f"组 {first_group_id} - 参数回归误差 vs Max Knowledge (先验 P2)")
        # plt.legend()
        # plt.grid(True, linestyle='--', alpha=0.6)
        # plt.xticks(ticks=x_ticks, labels=x_labels, rotation=45, ha='right')
        # plt.tight_layout()
        # plt.savefig("params_error_vs_max_k.png")
        # plt.close()
        # print("已保存: params_error_vs_max_k.png")

        # # 图 3: 内存占用 (Peak Memory) 和 AIC
        # fig, ax1 = plt.subplots(figsize=(12, 7))

        # color = 'tab:red'
        # ax1.set_xlabel("最大知识点数 (max_knowledge) \n (括号内为过滤后实际知识点 k)")
        # ax1.set_ylabel("峰值内存占用 (MB)", color=color)
        # ax1.plot(plot_x_axis, memory_peaks, marker='D', linestyle='-', color=color, label='内存峰值 (MB)')
        # ax1.tick_params(axis='y', labelcolor=color)

        # ax2 = ax1.twinx() 
        # color = 'tab:blue'
        # ax2.set_ylabel("AIC 值 (越低越好)", color=color)
        # ax2.plot(plot_x_axis, aics, marker='o', linestyle='--', color=color, label='AIC')
        # ax2.tick_params(axis='y', labelcolor=color)

        # plt.title(f"组 {first_group_id} - 内存/AIC vs Max Knowledge (先验 P2)")
        # fig.tight_layout() 
        # plt.xticks(ticks=x_ticks, labels=x_labels, rotation=45, ha='right')
        # lines, labels = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        
        # plt.grid(True, linestyle='--', alpha=0.6)
        # plt.savefig("memory_aic_vs_max_k.png")
        # plt.close()
        # print("已保存: memory_aic_vs_max_k.png")

        # print("\n所有图表已保存至当前目录。")
        
    #     results_df = pd.DataFrame({
    #         'max_k': plot_x_axis,
    #         'actual_k': actual_k_used,
    #         'accuracy': accuracies,
    #         'slip_error': slip_errors,
    #         'guess_error': guess_errors,
    #         'pi_error': pi_errors,
    #         'AIC': aics,
    #         'memory_MB': memory_peaks
    #     })
    #     print("\n--- 实验结果概览 ---")
    #     print(results_df.to_string())


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