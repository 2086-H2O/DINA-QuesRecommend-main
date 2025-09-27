# 读取 output_matrix.xlsx 文件
import pandas as pd

output_matrix = pd.read_excel('knowledge_point_matrix.xlsx', index_col=0)
print("output_matrix :", output_matrix)
print("output_matrix 的列名（题目 ID）:", output_matrix.columns.tolist())
print("output_matrix 的行名（知识点）:", output_matrix.index.tolist())

# 确保output_matrix的索引是题目ID
output_matrix.index = output_matrix.index.astype(str)  # 统一为字符串类型

# 获取组内题目ID（来自X_group_df）
group_qs_ids = X_group_df.columns.astype(str).tolist()  # 转为字符串列表
all_qs_ids = output_matrix.index.tolist()  # 所有题目ID（行名）

# 初始化Q矩阵（知识点数×题目数）
n_knowledge = len(output_matrix.columns)  # 知识点总数（列数）
n_questions = len(group_qs_ids)          # 组内题目数
Q = np.zeros((n_knowledge, n_questions), dtype=int)

# 构建知识点名称列表（用于调试）
knowledge_names = output_matrix.columns.tolist()
print(f"知识点列表: {knowledge_names}")

# 按题目ID匹配构建Q矩阵
matched = 0
for j, q_id in enumerate(group_qs_ids):
    if q_id in all_qs_ids:
        # 获取该题目对应的知识点向量（所有列的值）
        Q[:, j] = output_matrix.loc[q_id].values
        matched += 1
    else:
        print(f"警告: 题目ID {q_id} 不存在于Q矩阵中，已填充为0")

print(f"\nQ矩阵构建完成 - 匹配率: {matched}/{n_questions} ({matched/n_questions:.1%})")

# 打印Q矩阵摘要
print("\nQ矩阵摘要:")
print(f"行（知识点）: {len(knowledge_names)}个 | 列（题目）: {n_questions}个")
print("示例知识点-题目关系:")
for i, kn in enumerate(knowledge_names[:3]):  # 打印前3个知识点
    related_qs = np.where(Q[i, :] == 1)[0]
    print(f"  {kn}: 关联题目{len(related_qs)}个 (示例: {group_qs_ids[related_qs[0]] if len(related_qs)>0 else '无'})")