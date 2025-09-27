import pandas as pd

# 读取 output_matrix.xlsx 文件
output_matrix = pd.read_excel('515matrix.xlsx', index_col=0)  # 假设第一列是知识点名称（如 Q1, Q2, ...）
# output_matrix.index = output_matrix.index.str.extract(r'(\d+)')[0].astype(int)
print("output_matrix :", output_matrix)
print("output_matrix 的列名（题目 ID）:", output_matrix.columns.tolist())
print("output_matrix 的行名（知识点）:", output_matrix.index.tolist())

# 获取知识点数量（行数）和所有可能的题目 ID（列名）
n_kno_total = output_matrix.shape[0]  # 知识点数量（例如 18）
all_qs_ids = output_matrix.columns  # 所有题目 ID


print(all_qs_ids)