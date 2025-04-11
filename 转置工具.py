import numpy as np
import pandas as pd

# 读取 CSV 文件到 DataFrame
df = pd.read_csv('Math1_1/q.csv')

# 转置 DataFrame
df_transposed = df.transpose()

# 将转置后的 DataFrame 写回新的 CSV 文件
df_transposed.to_csv('Math1_1/q.csv', index=False, header=False)

print("转置完成！")


Alpha= pd.read_csv('Math1_1/alpha2.csv').values
Alpha = np.array(Alpha)

print(Alpha)