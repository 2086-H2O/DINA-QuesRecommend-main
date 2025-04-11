# 载入 Excel 文件
import pandas as pd
import openpyxl


file_path = 'output_matrix.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path, index_col=0)  # 使用第一列作为行索引

def extract_knowledge_matrix(data, question_sequence):
    # 提取这些题目的数据
    selected_data = data.loc[question_sequence]  # 使用题目编号作为索引

    # 转置矩阵，使行表示知识点，列表示题目
    knowledge_matrix = selected_data.T

    return knowledge_matrix


question_sequence = ['Q10', 'Q14', 'Q15', 'Q16']
knowledge_matrix = extract_knowledge_matrix(data, question_sequence)


print(knowledge_matrix)