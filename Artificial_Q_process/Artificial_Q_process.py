import pandas as pd
import os

# ================= 配置区 =================
# 老师提供的文件路径 (支持 .xlsx 或 .csv)
FILE_PATH = r"/Users/gantaotao/Documents/Develop2086/DINA-QuesRecommend/DINA-QuesRecommend-main/Artificial_Q_process/知识点人工(1).xlsx" 

# 知识点列的前缀 (根据您的描述是 "知识点")
KP_COL_PREFIX = "知识点"
# 槽位数量 (知识点1 - 知识点8)
SLOT_COUNT = 8
# =========================================

def main():
    if not os.path.exists(FILE_PATH):
        print(f"❌ 找不到文件: {FILE_PATH}")
        return

    print(f"正在读取文件: {FILE_PATH} ...")
    try:
        if FILE_PATH.endswith('.csv'):
            df = pd.read_csv(FILE_PATH)
        else:
            df = pd.read_excel(FILE_PATH)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 1. 构造列名列表 ['知识点1', '知识点2', ..., '知识点8']
    target_cols = [f"{KP_COL_PREFIX}{i}" for i in range(1, SLOT_COUNT + 1)]
    
    # 检查这些列是否存在
    existing_cols = [c for c in target_cols if c in df.columns]
    if not existing_cols:
        print(f"❌ 未找到类似 '{KP_COL_PREFIX}X' 的列，请检查表头。")
        print(f"当前表头: {df.columns.tolist()}")
        return

    print(f"✅ 检测到知识点列: {existing_cols}")

    # 2. 提取所有知识点
    all_kps = []
    
    # 遍历每一列，把非空的内容加到大列表里
    for col in existing_cols:
        # 取出这一列数据
        series = df[col].dropna().astype(str)
        # 去除首尾空格 (非常重要，防止 ' K1' 和 'K1' 算两个)
        series = series.str.strip()
        # 排除空字符串或纯空格
        series = series[series != '']
        
        all_kps.extend(series.tolist())

    # 3. 统计唯一值和频次
    kp_counts = pd.Series(all_kps).value_counts().sort_index()
    
    unique_kps = kp_counts.index.tolist()

    print("-" * 30)
    print(f"🎉 提取完成！共发现 {len(unique_kps)} 个唯一知识点。")
    print("-" * 30)
    
    # 4. 打印结果
    print(f"{'知识点名称':<30} | {'出现频次':<10}")
    print("-" * 45)
    for kp, count in kp_counts.items():
        print(f"{kp:<35} | {count}")

    # 5. (可选) 保存到文件方便查看
    output_file = "unique_kps_check.csv"
    kp_counts.to_frame(name='count').to_csv(output_file, encoding='utf-8-sig')
    print("-" * 45)
    print(f"📝 结果已保存至: {output_file} (可用Excel打开检查是否有同义词)")

if __name__ == "__main__":
    main()