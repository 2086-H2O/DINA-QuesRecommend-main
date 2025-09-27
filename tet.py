import numpy as np
import pandas as pd

# 读取 output_matrix.xlsx 文件
output_matrix = pd.read_excel('516matrix.xlsx', index_col=0)
print("output_matrix 的列名（题目 ID）:", output_matrix.columns.tolist())
print("output_matrix 的行名（知识点）:", output_matrix.index.tolist())


# 假设 output_matrix 是题目×知识点的 DataFrame（行是题目，列是知识点）
# 如果需要转置为知识点×题目，取消下面一行的注释
# output_matrix = output_matrix.T

# 直接提取所有数据作为 Q 矩阵
Q = output_matrix.values.T  # 转置为形状 (知识点数, 题目数)
Q = Q.astype(int)  # 确保数据类型为整数

# 检查全0题目（没有任何知识点关联的题目）
zero_question_mask = (Q.sum(axis=0) == 0)  # 列和为0的题目
zero_question_ids = output_matrix.index[zero_question_mask].tolist()  # 获取题目ID
zero_count = len(zero_question_ids)  # 全0题目总数
total_questions = Q.shape[1]  # 总题目数

print(f"生成的 Q 矩阵形状: {Q.shape} (知识点×题目)")
print("知识点数量:", Q.shape[0])
print("题目数量:", Q.shape[1])



print("Q 矩阵形状:", Q.shape)
print("1 的比例:", np.mean(Q))
print("每知识点关联题数:", np.sum(Q, axis=1))
print("每题关联知识点数:", np.sum(Q, axis=0))
print("\n全0题目（未关联任何知识点）的ID:", zero_question_ids)
print(f"全0题目总数: {zero_count} (占比: {zero_count/total_questions:.2%})")