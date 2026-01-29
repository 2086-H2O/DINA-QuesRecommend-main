"""
不分组的半遮面
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from itertools import product
import os
import warnings

# 忽略数值计算中的一些除零警告
warnings.filterwarnings('ignore')

# ==========================================
#               1. 实验配置区 (User Config)
# ==========================================

# --- 核心参数配置 ---
EXPERIMENT_NAME = "全量数据测试 (All Students)"
R_TRAIN_STUDENTS = 0.8  # r: 80% 的学生用于训练题目参数 (s, g)
K_OBSERVED_ITEMS = 0.5  # k: 测试集学生 50% 的题目已知，用于预测剩下 50%
RANDOM_SEED = 42        # 固定随机种子
MAX_KNOWLEDGE = 15      # 智能降维阈值

# --- 待测 Q 矩阵列表 ---
Q_MATRIX_LIST = {
    "1. LLM策略2": r"LLM_Q_Generate\outputs\4+10_1_改_results\DINA_Q_Matrix_4+10_改.xlsx",
    "2. LLM策略3": r"LLM_Q_Generate\outputs\4+10_2_8_results\DINA_Q_Matrix_4+10_2_8.xlsx",
    "3. LLM策略4": r"LLM_Q_Generate\outputs\4+10_3_results\DINA_Q_Matrix_3.xlsx",
    "4. 专家手工降维 (14维)": r"Artificial_Q_process\Q矩阵_手工标注合并14个.xlsx",
    "5. 老版本 V1 (10维-词频)": r"516matrix.xlsx"
}


DATA_PATH = "cleaned_data_20250326_0931.csv"
# 注意：我们现在不再依赖 optimal_student_groups_leiden.csv 来切分，而是直接读取 data 里的所有学生

# 兼容路径前缀
if not os.path.exists(DATA_PATH) and os.path.exists(f"DINA-QuesRecommend-main/{DATA_PATH}"):
    DATA_PATH = f"DINA-QuesRecommend-main/{DATA_PATH}"

# ==========================================
#           2. DINA 核心算子
# ==========================================

def compute_eta(Q, A):
    kowns = np.sum(Q * Q, axis=0)
    cross = np.dot(A, Q)
    eta = np.ones(shape=(A.shape[0], Q.shape[1]))
    eta[cross < kowns] = 0
    return eta

def compute_propa(eta, s, g):
    propa = (g ** (1 - eta)) * ((1 - s) ** eta)
    return np.clip(propa, 1e-10, 1 - 1e-10)

def compute_gamma(X, pi, propa):
    log_pj = np.log(propa)
    log_qj = np.log(1 - propa)
    log_pi = np.log(pi)
    gamma = np.exp(np.dot(X, log_pj.T) + np.dot((1 - X), log_qj.T) + log_pi)
    gamma_sum = np.sum(gamma, axis=1, keepdims=True)
    gamma_sum[gamma_sum == 0] = 1e-15
    return gamma / gamma_sum

def compute_theta(X, gamma, eta):
    I0 = np.dot(gamma, 1 - eta)
    I1 = np.dot(gamma, eta)
    R0 = I0 * X
    R1 = I1 * X
    
    I0_sum, I1_sum = np.sum(I0, axis=0), np.sum(I1, axis=0)
    R0_sum, R1_sum = np.sum(R0, axis=0), np.sum(R1, axis=0)
    
    I0_sum[I0_sum <= 0] = 1e-15
    I1_sum[I1_sum <= 0] = 1e-15
    
    g = R0_sum / I0_sum
    s = (I1_sum - R1_sum) / I1_sum
    pi = np.sum(gamma, axis=0) / gamma.shape[0]
    
    return np.clip(pi, 1e-15, 1-1e-15), np.clip(s, 0.001, 0.999), np.clip(g, 0.001, 0.999)

def train_dina(X, Q, max_iter=30, tol=1e-3):
    n_items = X.shape[1]
    n_kno = Q.shape[0]
    
    s = np.random.uniform(0.1, 0.3, n_items)
    g = np.random.uniform(0.1, 0.3, n_items)
    A_all = np.array(list(product([0, 1], repeat=n_kno)))
    pi = np.ones(A_all.shape[0]) / A_all.shape[0]
    
    for t in range(max_iter):
        eta = compute_eta(Q, A_all)
        propa = compute_propa(eta, s, g)
        gamma = compute_gamma(X, pi, propa)
        pi_new, s_new, g_new = compute_theta(X, gamma, eta)
        
        diff = max(np.max(np.abs(pi_new - pi)), np.max(np.abs(s_new - s)), np.max(np.abs(g_new - g)))
        pi, s, g = pi_new, s_new, g_new
        if diff < tol: break
            
    return {"s": s, "g": g, "pi": pi, "A_all": A_all}

# ==========================================
#      3. 智能 Q 矩阵构建
# ==========================================

def build_smart_q_matrix(file_path, group_qs_ids, max_k=15):
    try:
        # 兼容路径查找
        if not os.path.exists(file_path):
            alt_path = f"DINA-QuesRecommend-main/{file_path}"
            if os.path.exists(alt_path):
                file_path = alt_path
            else:
                base_name = os.path.basename(file_path)
                for root, dirs, files in os.walk("."):
                    if base_name in files:
                        file_path = os.path.join(root, base_name)
                        break
        
        if not os.path.exists(file_path):
            print(f"   [Error] 文件未找到: {file_path}")
            return None, None

        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
            
        # 索引处理
        cols_lower = [c.lower() for c in df.columns]
        if 'id' in cols_lower: df = df.set_index(df.columns[cols_lower.index('id')])
        elif 'qs_id' in cols_lower: df = df.set_index(df.columns[cols_lower.index('qs_id')])
        else: df = df.set_index(df.columns[0])
        df.index = df.index.astype(str)
        
        # 提取知识点
        numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
        
        valid_qs = [q for q in group_qs_ids if q in df.index]
        if not valid_qs: return None, None
        
        subset_df = numeric_df.loc[valid_qs]
        
        # 智能降维
        if subset_df.shape[1] > max_k:
            print(f"   [优化] 知识点 {subset_df.shape[1]} > {max_k}，执行 Top-K 降维...")
            top_cols = subset_df.sum(axis=0).nlargest(max_k).index
            subset_df = subset_df[top_cols]
            
        Q = (subset_df.values > 0).astype(int).T
        valid_k = np.where(Q.sum(axis=1) > 0)[0]
        Q = Q[valid_k, :]
        
        return Q, valid_qs
    except Exception as e:
        print(f"   [Load Error] {e}")
        return None, None

# ==========================================
#               4. 实验主程序
# ==========================================

def run_experiment():
    print(f"🚀 开始全量数据测试 | 训练r={R_TRAIN_STUDENTS} | 已知k={K_OBSERVED_ITEMS}")
    print("=" * 70)
    
    # 1. 加载所有数据
    try:
        data = pd.read_csv(DATA_PATH)
        data['qs_id'] = data['qs_id'].astype(str)
        
        # 直接透视所有学生数据，不分 Group
        print("正在构建全量学生答题矩阵 (这可能需要几秒钟)...")
        X_df = data.pivot_table(
            index='student_id', columns='qs_id', values='qs_validity', fill_value=0
        )
        print(f"✅ 全量矩阵构建完成: {X_df.shape[0]} 学生 x {X_df.shape[1]} 题目")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    results = []

    # 2. 遍历 Q 矩阵
    for q_name, q_path in Q_MATRIX_LIST.items():
        print(f"\n📂 正在评测: [{q_name}]")
        print("-" * 50)
        
        # 构建 Q 矩阵
        Q, valid_qs = build_smart_q_matrix(q_path, X_df.columns.astype(str).tolist(), MAX_KNOWLEDGE)
        if Q is None: continue
        
        X = X_df[valid_qs].values
        n_items = X.shape[1]
        n_kno = Q.shape[0]
        
        # 3. 划分学生 (Train / Test)
        # 80% 的学生用于训练题目参数
        X_train_stu, X_test_stu = train_test_split(X, train_size=R_TRAIN_STUDENTS, random_state=RANDOM_SEED)
        
        # --- Phase 1: 训练 (Learning) ---
        print(f"   全量训练 (N={len(X_train_stu)})...")
        model = train_dina(X_train_stu, Q, max_iter=30)
        s_learned, g_learned, pi_learned = model['s'], model['g'], model['pi']
        A_all = model['A_all']
        
        # --- Phase 2: 测试 (Split-Item Prediction) ---
        # 准备遮挡掩码
        all_indices = np.arange(n_items)
        np.random.shuffle(all_indices)
        n_obs = int(n_items * K_OBSERVED_ITEMS)
        
        idx_obs = all_indices[:n_obs] # 已知题目
        idx_tar = all_indices[n_obs:] # 待测题目
        
        # 准备数据
        X_obs = X_test_stu[:, idx_obs]
        X_tar = X_test_stu[:, idx_tar]
        
        Q_obs = Q[:, idx_obs]
        s_obs, g_obs = s_learned[idx_obs], g_learned[idx_obs]
        
        Q_tar = Q[:, idx_tar]
        s_tar, g_tar = s_learned[idx_tar], g_learned[idx_tar]
        
        # Step A: 推断能力
        eta_obs = compute_eta(Q_obs, A_all)
        propa_obs = compute_propa(eta_obs, s_obs, g_obs)
        gamma_test = compute_gamma(X_obs, pi_learned, propa_obs)
        
        # Step B: 预测未知题目
        eta_tar = compute_eta(Q_tar, A_all)
        propa_tar = compute_propa(eta_tar, s_tar, g_tar)
        
        X_pred_prob = np.dot(gamma_test, propa_tar)
        X_pred_bin = (X_pred_prob >= 0.5).astype(int)
        
        # Step C: 评估
        acc = accuracy_score(X_tar.flatten(), X_pred_bin.flatten())
        loss = log_loss(X_tar.flatten(), X_pred_prob.flatten(), labels=[0,1])
        
        print(f"   [All Students] (K={n_kno}): Acc = {acc:.4f} | Loss = {loss:.4f} | (Test N={len(X_test_stu)})")
        
        results.append({
            "Matrix": q_name,
            "Knowledge_Dim": n_kno,
            "Test_Students": len(X_test_stu),
            "Split_Accuracy": acc,
            "Split_LogLoss": loss
        })

    # 汇总输出
    print("\n" + "="*70)
    print("🏆 全量数据实验结果汇总 (按 Accuracy 排序)")
    print("="*70)
    if results:
        res_df = pd.DataFrame(results)
        res_df = res_df[["Matrix", "Knowledge_Dim", "Split_Accuracy", "Split_LogLoss"]]
        print(res_df.sort_values(by="Split_Accuracy", ascending=False).to_string(index=False))
        res_df.to_csv("all_students_experiment_results.csv", index=False)
    else:
        print("无有效结果")

if __name__ == "__main__":
    run_experiment()