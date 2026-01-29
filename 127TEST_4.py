"""
全量学生、全量题目的自洽性检验，不分组
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from itertools import product
import os
import warnings

# 忽略数值计算警告
warnings.filterwarnings('ignore')

# ================= 配置区 =================
# 实验名称
EXPERIMENT_NAME = "全量数据自洽性检验 (Self-Consistency Check)"
MAX_KNOWLEDGE = 15      # 智能降维阈值

# 待测 Q 矩阵列表
Q_MATRIX_LIST = {
    "1. LLM策略2": r"LLM_Q_Generate\outputs\4+10_1_改_results\DINA_Q_Matrix_4+10_改.xlsx",
    "2. LLM策略3": r"LLM_Q_Generate\outputs\4+10_2_8_results\DINA_Q_Matrix_4+10_2_8.xlsx",
    "3. LLM策略4": r"LLM_Q_Generate\outputs\4+10_3_results\DINA_Q_Matrix_3.xlsx",
    "4. 专家手工降维 (14维)": r"Artificial_Q_process\Q矩阵_手工标注合并14个.xlsx",
    "5. 老版本 V1 (10维-词频)": r"516matrix.xlsx"
}

DATA_PATH = "cleaned_data_20250326_0931.csv"
if not os.path.exists(DATA_PATH) and os.path.exists(f"DINA-QuesRecommend-main/{DATA_PATH}"):
    DATA_PATH = f"DINA-QuesRecommend-main/{DATA_PATH}"

# ================= DINA 核心算子 =================
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
    
    g = np.sum(R0, axis=0) / np.maximum(np.sum(I0, axis=0), 1e-15)
    s = (np.sum(I1, axis=0) - np.sum(R1, axis=0)) / np.maximum(np.sum(I1, axis=0), 1e-15)
    pi = np.sum(gamma, axis=0) / gamma.shape[0]
    
    return np.clip(pi, 1e-15, 1-1e-15), np.clip(s, 0.001, 0.999), np.clip(g, 0.001, 0.999)

def train_dina_full(X, Q, max_iter=200, tol=1e-3):
    """全量训练，返回最终的拟合参数"""
    n_items = X.shape[1]
    n_kno = Q.shape[0]
    
    # 初始化
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
            
    # 计算重构矩阵 (Self-Consistency Check)
    # 用最终参数生成每个学生的“理论答题概率”
    # P(X_ij=1) = Sum_over_k ( Gamma_ik * P(X_ij=1|alpha_k) )
    eta_final = compute_eta(Q, A_all)
    propa_final = compute_propa(eta_final, s, g)
    X_reconstruct_prob = np.dot(gamma, propa_final)
    
    return X_reconstruct_prob, s, g

# ================= 智能 Q 矩阵加载 =================
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
            
        cols_lower = [c.lower() for c in df.columns]
        if 'id' in cols_lower: df = df.set_index(df.columns[cols_lower.index('id')])
        elif 'qs_id' in cols_lower: df = df.set_index(df.columns[cols_lower.index('qs_id')])
        else: df = df.set_index(df.columns[0])
        df.index = df.index.astype(str)
        
        numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
        valid_qs = [q for q in group_qs_ids if q in df.index]
        
        if not valid_qs: return None, None
        
        subset_df = numeric_df.loc[valid_qs]
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

# ================= 主程序 =================
def run_consistency_check():
    print(f"🚀 开始全量自洽性检验 (Training = 100% Data, Testing = 100% Data)")
    print("=" * 70)
    
    try:
        data = pd.read_csv(DATA_PATH)
        data['qs_id'] = data['qs_id'].astype(str)
        print("正在构建全量答题矩阵...")
        X_df = data.pivot_table(index='student_id', columns='qs_id', values='qs_validity', fill_value=0)
        print(f"✅ 全量矩阵: {X_df.shape[0]} 学生 x {X_df.shape[1]} 题目")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    results = []

    for q_name, q_path in Q_MATRIX_LIST.items():
        print(f"\n📂 正在检验: [{q_name}]")
        print("-" * 50)
        
        Q, valid_qs = build_smart_q_matrix(q_path, X_df.columns.astype(str).tolist(), MAX_KNOWLEDGE)
        if Q is None: continue
        
        X = X_df[valid_qs].values
        n_items = X.shape[1]
        n_kno = Q.shape[0]
        
        # 核心：全量训练 + 全量回测
        print(f"   执行 EM 算法 (N={len(X)})...")
        X_recon_prob, s, g = train_dina_full(X, Q, max_iter=30)
        
        # 评估 (Compare Original vs Reconstructed)
        X_recon_bin = (X_recon_prob >= 0.5).astype(int)
        
        acc = accuracy_score(X.flatten(), X_recon_bin.flatten())
        loss = log_loss(X.flatten(), X_recon_prob.flatten(), labels=[0,1])
        
        # 统计平均 s 和 g (反映题目质量)
        avg_s = np.mean(s)
        avg_g = np.mean(g)
        
        print(f"   [Result] Acc = {acc:.4f} | Loss = {loss:.4f} | Avg Slip={avg_s:.3f}, Avg Guess={avg_g:.3f}")
        
        results.append({
            "Matrix": q_name,
            "Knowledge_Dim": n_kno,
            "Consistency_Acc": acc,
            "Consistency_LogLoss": loss,
            "Avg_Slip": avg_s,
            "Avg_Guess": avg_g
        })

    print("\n" + "="*70)
    print("🏆 自洽性检验报告 (拟合度排名)")
    print("="*70)
    if results:
        res_df = pd.DataFrame(results)
        print(res_df.sort_values(by="Consistency_Acc", ascending=False).to_string(index=False))
        res_df.to_csv("consistency_check_results.csv", index=False)
    else:
        print("无结果")

if __name__ == "__main__":
    run_consistency_check()