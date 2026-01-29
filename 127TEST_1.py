import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from itertools import product
import os
import warnings

# 忽略一些除零警告
warnings.filterwarnings('ignore')

# ==========================================
#               1. 实验配置区 (User Config)
# ==========================================

# 1. 在这里配置所有你想测试的 Q 矩阵文件路径
# 格式：{"显示名称": "文件路径"}
Q_MATRIX_LIST = {
    "1. LLM策略2": r"LLM_Q_Generate\outputs\4+10_1_改_results\DINA_Q_Matrix_4+10_改.xlsx",
    "2. LLM策略3": r"LLM_Q_Generate\outputs\4+10_2_8_results\DINA_Q_Matrix_4+10_2_8.xlsx",
    "3. LLM策略4": r"LLM_Q_Generate\outputs\4+10_3_results\DINA_Q_Matrix_3.xlsx",
    "4. 专家手工降维 (14维)": r"Artificial_Q_process\Q矩阵_手工标注合并14个.xlsx",
    "5. 老版本 V1 (10维-词频)": r"516matrix.xlsx" 
}

# 2. 数据文件路径
DATA_PATH = "cleaned_data_20250326_0931.csv"
GROUP_PATH = "optimal_student_groups_leiden.csv"

# 3. 实验超参数
TRAIN_RATIO = 0.5       # 80% 训练，20% 测试
RANDOM_SEED = 42        # 固定随机种子，保证结果可复现
MAX_KNOWLEDGE = 15      # 【关键】如果知识点超过这个数，自动截断（防止死机）

# ==========================================
#           2. DINA 核心算子 (Core)
# ==========================================

def compute_eta(Q, A):
    kowns = np.sum(Q * Q, axis=0)
    cross = np.dot(A, Q)
    eta = np.ones(shape=(A.shape[0], Q.shape[1]))
    eta[cross < kowns] = 0
    return eta

def compute_propa(eta, s, g):
    propa = (g ** (1 - eta)) * ((1 - s) ** eta)
    propa = np.clip(propa, 1e-10, 1 - 1e-10) # 数值稳定
    return propa

def compute_gamma(X, pi, propa):
    log_pj = np.log(propa)
    log_qj = np.log(1 - propa)
    log_pi = np.log(pi)
    # 关键：利用矩阵乘法加速计算后验
    gamma = np.exp(np.dot(X, log_pj.T) + np.dot((1 - X), log_qj.T) + log_pi)
    gamma_sum = np.sum(gamma, axis=1, keepdims=True)
    gamma_sum[gamma_sum == 0] = 1e-15
    gamma = gamma / gamma_sum
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
    
    # 防止分母为0
    I0_sum[I0_sum <= 0] = 1e-15
    I1_sum[I1_sum <= 0] = 1e-15
    
    g = R0_sum / I0_sum
    s = (I1_sum - R1_sum) / I1_sum
    pi = np.sum(gamma, axis=0) / gamma.shape[0]
    
    return np.clip(pi, 1e-15, 1-1e-15), np.clip(s, 0.001, 0.999), np.clip(g, 0.001, 0.999)

# --- 训练函数 (Full EM) ---
def train_dina(X, Q, max_iter=50, tol=1e-3):
    n_stu, n_items = X.shape
    n_kno = Q.shape[0]
    
    # 初始化
    s = np.random.uniform(0.1, 0.3, n_items)
    g = np.random.uniform(0.1, 0.3, n_items)
    
    # 生成所有可能的掌握模式 (2^K)
    # 注意：如果 K > 20 这里会爆内存，但我们前面的 MAX_KNOWLEDGE 会防住它
    A_all = np.array(list(product([0, 1], repeat=n_kno)))
    pi = np.ones(A_all.shape[0]) / A_all.shape[0] # 均匀分布初始化
    
    for t in range(max_iter):
        eta = compute_eta(Q, A_all)
        propa = compute_propa(eta, s, g)
        gamma = compute_gamma(X, pi, propa)
        pi_new, s_new, g_new = compute_theta(X, gamma, eta)
        
        # 检查收敛
        diff = max(np.max(np.abs(pi_new - pi)), np.max(np.abs(s_new - s)), np.max(np.abs(g_new - g)))
        pi, s, g = pi_new, s_new, g_new
        if diff < tol:
            break
            
    return {"s": s, "g": g, "pi": pi, "A_all": A_all}

# --- 预测函数 (Inference Only) ---
def predict_dina(X_test, Q, model_params):
    s, g, pi, A_all = model_params["s"], model_params["g"], model_params["pi"], model_params["A_all"]
    
    # 1. 计算理论答题概率
    eta = compute_eta(Q, A_all)
    propa = compute_propa(eta, s, g)
    
    # 2. E-Step: 推断测试集学生的能力分布 (Gamma)
    gamma_test = compute_gamma(X_test, pi, propa)
    
    # 3. 预测答题行为 (概率矩阵)
    # 学生的预测答题概率 = sum(该学生属于模式k的概率 * 模式k答对该题的概率)
    X_pred_prob = np.dot(gamma_test, propa)
    
    return X_pred_prob

# ==========================================
#      3. 智能 Q 矩阵构建 (Smart Builder)
# ==========================================

def build_smart_q_matrix(file_path, group_qs_ids, max_k=15):
    """
    智能加载函数：
    1. 读取 Q 矩阵文件
    2. 自动匹配本组题目
    3. 【关键】如果知识点过多，自动筛选 Top-K 高频知识点，防止爆炸
    """
    try:
        # 1. 读取文件
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
            
        # 尝试将第一列或名为 id/qs_id 的列设为索引
        cols_lower = [c.lower() for c in df.columns]
        if 'id' in cols_lower:
            df = df.set_index(df.columns[cols_lower.index('id')])
        elif 'qs_id' in cols_lower:
            df = df.set_index(df.columns[cols_lower.index('qs_id')])
        elif '题目id' in cols_lower:
             df = df.set_index(df.columns[cols_lower.index('题目id')])
        else:
            # 默认第一列是 ID
            df = df.set_index(df.columns[0])
            
        # 统一索引为字符串
        df.index = df.index.astype(str)
        
        # 2. 识别知识点列 (数字列，且不是全是0或全是1之外的乱七八糟的数)
        # 简单逻辑：选取所有数值类型的列作为候选
        numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
        
        # 3. 筛选出本组涉及的题目
        valid_qs = [q for q in group_qs_ids if q in df.index]
        if not valid_qs:
            print(f"   [Error] 该 Q 矩阵未包含本组任何题目！")
            return None, None
            
        subset_df = numeric_df.loc[valid_qs]
        
        # 4. 【核心优化】知识点降维逻辑
        current_k = subset_df.shape[1]
        
        if current_k > max_k:
            print(f"   [优化] 检测到知识点维度 K={current_k} > {max_k}，正在执行智能降维...")
            # 计算每个知识点的覆盖率（在本组题目中）
            coverage = subset_df.sum(axis=0)
            # 选出 Top-K
            top_cols = coverage.nlargest(max_k).index
            final_df = subset_df[top_cols]
            # 再次转为 0/1 (防止 excel 里写了 2, 3 这种权重)
            Q_matrix = (final_df.values > 0).astype(int).T # 转置为 (K, Items)
            print(f"   [成功] 已降维至 Top {max_k} 知识点")
        else:
            Q_matrix = (subset_df.values > 0).astype(int).T
            
        # 移除全零行（有些知识点可能在本组题目里根本没考）
        valid_k_idx = np.where(Q_matrix.sum(axis=1) > 0)[0]
        Q_matrix = Q_matrix[valid_k_idx, :]
        
        return Q_matrix, valid_qs
        
    except Exception as e:
        print(f"   [加载失败] {file_path}: {e}")
        return None, None

# ==========================================
#               4. 主实验流程
# ==========================================

def main():
    print(f"🚀 开始多轮实验 | 训练集: {TRAIN_RATIO*100}% | MAX_K: {MAX_KNOWLEDGE}")
    print("=" * 65)
    
    # 1. 加载基础数据
    try:
        data_df = pd.read_csv(DATA_PATH)
        data_df['qs_id'] = data_df['qs_id'].astype(str)
        group_df = pd.read_csv(GROUP_PATH)
        print("✅ 基础数据加载成功")
    except Exception as e:
        print(f"❌ 数据文件缺失: {e}")
        return

    results = []

    # 2. 遍历 Q 矩阵列表
    for q_name, q_path in Q_MATRIX_LIST.items():
        if not os.path.exists(q_path):
            print(f"\n⚠️ 跳过 {q_name}: 文件不存在")
            continue
            
        print(f"\n📂 正在评测: [{q_name}]")
        print("-" * 40)
        
        # 3. 遍历组别 (Group 0, 1, 2)
        target_groups = [0, 1, 2]
        
        for grp_id in target_groups:
            # 准备该组的学生答题数据 X
            stu_ids = group_df[group_df['group'] == grp_id]['student_id'].values
            if len(stu_ids) < 10: continue # 忽略小样本
            
            grp_records = data_df[data_df['student_id'].isin(stu_ids)]
            # 转为矩阵形式: 行=学生, 列=题目
            X_df = grp_records.pivot_table(index='student_id', columns='qs_id', values='qs_validity', fill_value=0)
            group_qs_ids = X_df.columns.astype(str).tolist()
            
            # --- 智能构建 Q 矩阵 ---
            Q, valid_qs = build_smart_q_matrix(q_path, group_qs_ids, max_k=MAX_KNOWLEDGE)
            
            if Q is None or Q.shape[0] == 0:
                print(f"   Group {grp_id}: Q 矩阵构建失败或无匹配题目")
                continue
                
            # 对齐数据：只取 Q 矩阵中存在的题目
            X_aligned = X_df[valid_qs].values
            
            # --- 划分训练/测试集 ---
            try:
                X_train, X_test = train_test_split(X_aligned, train_size=TRAIN_RATIO, random_state=RANDOM_SEED)
            except ValueError:
                print(f"   Group {grp_id}: 样本不足无法划分，跳过")
                continue
            
            # --- 阶段 1: 训练 (Learning) ---
            # 使用训练集学习 s, g, pi
            model = train_dina(X_train, Q, max_iter=30)
            
            # --- 阶段 2: 测试 (Inference) ---
            # 使用学习到的参数预测测试集
            X_test_pred_prob = predict_dina(X_test, Q, model)
            
            # --- 评估 ---
            X_test_pred_bin = (X_test_pred_prob >= 0.5).astype(int)
            acc = accuracy_score(X_test.flatten(), X_test_pred_bin.flatten())
            loss = log_loss(X_test.flatten(), X_test_pred_prob.flatten(), labels=[0,1])
            
            print(f"   Group {grp_id} (K={Q.shape[0]}): Test Acc = {acc:.4f} | LogLoss = {loss:.4f}")
            
            results.append({
                "Matrix": q_name,
                "Group": grp_id,
                "Knowledge_Dim": Q.shape[0],
                "Test_Accuracy": acc,
                "Test_LogLoss": loss
            })

    # 4. 最终汇总输出
    print("\n" + "="*65)
    print("🏆 最终实验报告 (按测试集准确率排序)")
    print("="*65)
    if results:
        res_df = pd.DataFrame(results)
        # 调整列顺序
        res_df = res_df[["Matrix", "Group", "Knowledge_Dim", "Test_Accuracy", "Test_LogLoss"]]
        print(res_df.sort_values(by="Test_Accuracy", ascending=False).to_string(index=False))
        
        # 保存结果到文件
        res_df.to_csv("final_experiment_results.csv", index=False)
        print("\n✅ 详细结果已保存至 final_experiment_results.csv")
    else:
        print("未产生有效结果，请检查文件路径配置。")

if __name__ == "__main__":
    main()