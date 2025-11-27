import pandas as pd
import json
import time
import requests
import os

# 🌟 从同级目录导入提示词
from prompt2_5 import prompt

# ================= 配置区 =================
# DeepSeek API 配置
API_KEY = "sk-2636e69fcc0744fa8b975e2b82eaa345"
API_URL = "https://api.deepseek.com/chat/completions"
MODEL_NAME = "deepseek-chat"

# 文件路径配置
INPUT_FILE = "所有题目.xlsx"
NOTE = "4+10_2_7"
OUTPUT_DIR = f"./outputs/{NOTE}_results"  # 📂 指定输出文件夹路径
BATCH_SIZE = 20

# 🧪 测试模式配置
TEST_MODE = False
TEST_K = 40  
RANDOM_SEED = 42  

# 💾 临时存档 (自动保存在输出目录下)
TEMP_FILENAME = f"temp_saved_tags_{NOTE}.csv"

# 📚 知识点定义 (用于生成列名和人工对照文本)
KNOWLEDGE_MAP = {
    1: "K1_仪器操作", 2: "K2_电路构建", 3: "K3_故障排查", 4: "K4_数据处理",
    5: "K5_直流分析", 6: "K6_暂态过程", 7: "K7_交流稳态", 8: "K8_频率响应",
    9: "K9_谐振理论", 10: "K10_半导体", 11: "K11_放大电路", 12: "K12_运放应用",
    13: "K13_振荡反馈", 14: "K14_变压器三相"
}
# ========================================

def call_deepseek_api(batch_df):
    """
    调用 DeepSeek API 进行打标
    """
    questions_text = ""
    for _, row in batch_df.iterrows():
        questions_text += f"ID: {row['id']}\n题目: {row['qs_title']}\n章节: {row['section_name']}\n---\n"

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"请对以下题目进行打标，严格按照 JSON 格式返回:\n\n{questions_text}"}
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"}
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    retries = 3
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result_json = response.json()
            content = result_json['choices'][0]['message']['content']
            content = content.replace("```json", "").replace("```", "").strip()
            
            data = json.loads(content)
            
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        return value
                print(f"警告: 返回的 JSON 结构不符合预期: {data.keys()}")
                return []
            elif isinstance(data, list):
                return data
                
        except Exception as e:
            print(f"⚠️ API 请求出错 (尝试 {attempt+1}/{retries}): {e}")
            time.sleep(2)
            
    return []

def main():
    # 0. 准备输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📂 已创建输出目录: {OUTPUT_DIR}")
    
    temp_save_path = os.path.join(OUTPUT_DIR, TEMP_FILENAME)

    # 1. 读取数据
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到文件 {INPUT_FILE}")
        return

    print(f"正在读取 {INPUT_FILE} ...")
    df = pd.read_excel(INPUT_FILE)
    df['id'] = df['id'].astype(str)

    # 🧪 测试模式：随机抽样
    if TEST_MODE:
        print(f"\n🎲 测试模式已开启！使用种子 {RANDOM_SEED} 随机抽取 {TEST_K} 条数据...\n")
        # 如果数据量不够抽，就取全部
        n_sample = min(TEST_K, len(df))
        df = df.sample(n=n_sample, random_state=RANDOM_SEED).sort_index() # 抽样后按原序排列，方便查看
    
    # 2. 准备结果容器
    labeled_results = []
    total_batches = (len(df) // BATCH_SIZE) + 1
    
    print(f"共 {len(df)} 道题目，将分为 {total_batches} 个批次处理。")
    print("-" * 50)

    # 清理旧的临时文件 (仅非测试模式)
    if os.path.exists(temp_save_path) and not TEST_MODE:
        print(f"提示: 清理旧临时文件 {temp_save_path}")
        os.remove(temp_save_path)

    # 3. 分批处理
    start_time = time.time()
    
    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        print(f"🚀 [批次 {batch_num}/{total_batches}] 处理题目 {i} - {min(i+BATCH_SIZE, len(df))} ... ", end="")
        
        tags = call_deepseek_api(batch)
        
        if tags:
            labeled_results.extend(tags)
            print(f"✅ 成功 ({len(tags)}条)")
            
            # 💾 实时保存
            try:
                temp_df = pd.DataFrame(tags)
                temp_df.to_csv(temp_save_path, mode='a', header=not os.path.exists(temp_save_path), index=False)
            except Exception as e:
                print(f"⚠️ 临时文件写入失败: {e}")
        else:
            print("❌ 失败 (跳过)")
        
        time.sleep(0.5)

    # 4. 生成最终结果
    print("-" * 50)
    print("正在生成最终结果文件...")
    
    if not labeled_results:
        print("没有获取到任何标签数据，程序结束。")
        return

    tags_df = pd.DataFrame(labeled_results)
    tags_df['id'] = tags_df['id'].astype(str)
    
    # 文件名生成
    suffix = f"_{NOTE}" if NOTE else ""
    human_filename = os.path.join(OUTPUT_DIR, f"DINA_Mark_Results{suffix}.xlsx")
    q_matrix_filename = os.path.join(OUTPUT_DIR, f"DINA_Q_Matrix{suffix}.xlsx")

    # ==========================================
    # 版本 A: 人工对照版 (Human Review) - 保持不变
    # ==========================================
    def convert_tags_to_names(tag_list):
        if not isinstance(tag_list, list): return ""
        names = [KNOWLEDGE_MAP.get(tag_id, f"未知ID_{tag_id}") for tag_id in tag_list]
        return ", ".join(names)

    human_tags_df = tags_df.copy()
    human_tags_df['knowledge_names'] = human_tags_df['tags'].apply(convert_tags_to_names)
    human_tags_df['tags_raw'] = human_tags_df['tags'].apply(lambda x: str(x))
    
    human_final_df = pd.merge(df, human_tags_df[['id', 'knowledge_names', 'tags_raw']], on='id', how='left')
    human_final_df.to_excel(human_filename, index=False)
    print(f"✅ [1/2] 人工对照表已保存: {human_filename}")

    # ==========================================
    # 版本 B: Q-Matrix 增强版 (多 Sheet)
    # ==========================================
    knowledge_columns = list(KNOWLEDGE_MAP.values())
    
    # 1. 准备基础 Q 矩阵数据
    for col in knowledge_columns:
        tags_df[col] = 0
        
    for index, row in tags_df.iterrows():
        tag_list = row['tags']
        if isinstance(tag_list, list):
            for tag_id in tag_list:
                if tag_id in KNOWLEDGE_MAP:
                    col_name = KNOWLEDGE_MAP[tag_id]
                    tags_df.at[index, col_name] = 1

    q_cols_to_merge = ['id'] + knowledge_columns
    q_matrix_df = pd.merge(df, tags_df[q_cols_to_merge], on='id', how='left')
    q_matrix_df[knowledge_columns] = q_matrix_df[knowledge_columns].fillna(0).astype(int)

    # 2. 准备“覆盖情况”数据 (Sheet 2)
    coverage_data = []
    for col in knowledge_columns:
        # 筛选出当前知识点为 1 的行
        covered_rows = q_matrix_df[q_matrix_df[col] == 1]
        count = len(covered_rows)
        # 获取题目 ID 列表，用逗号连接 (为了防止 ID 太长 Excel 显示不全，这里只存前 50 个 ID 作为示例，或者全部存)
        # 这里我存全部 ID
        ids_str = ",".join(covered_rows['id'].astype(str).tolist())
        
        coverage_data.append({
            "知识点": col,
            "覆盖题目数": count,
            "题目ID列表": ids_str
        })
    df_coverage = pd.DataFrame(coverage_data)

    # 3. 准备“题目关联数量分布”数据 (Sheet 4)
    # 计算每行有多少个 1
    row_sums = q_matrix_df[knowledge_columns].sum(axis=1)
    # 统计分布
    dist_counts = row_sums.value_counts().sort_index()
    df_dist = dist_counts.reset_index()
    df_dist.columns = ['关联知识点数量', '题目数']

    # 4. 准备“统计信息”数据 (Sheet 3)
    stats_data = {
        "指标": [
            "总题目数", 
            "总知识点数", 
            "平均每题关联知识点数", 
            "未匹配题目数 (关联数为0)", 
            "知识点覆盖数中位数", 
            "知识点最大覆盖数",
            "知识点最小覆盖数"
        ],
        "数值": [
            len(q_matrix_df),
            len(knowledge_columns),
            round(row_sums.mean(), 2),
            (row_sums == 0).sum(),
            df_coverage["覆盖题目数"].median(),
            df_coverage["覆盖题目数"].max(),
            df_coverage["覆盖题目数"].min()
        ]
    }
    df_stats = pd.DataFrame(stats_data)

    # 5. 写入 Excel (使用 ExcelWriter 写入多 Sheet)
    try:
        with pd.ExcelWriter(q_matrix_filename, engine='openpyxl') as writer:
            # Sheet 1: 必须是 Q 矩阵，且放在第一个，保证兼容性
            q_matrix_df.to_excel(writer, sheet_name='Q矩阵', index=False)
            
            # Sheet 2: 覆盖情况
            df_coverage.to_excel(writer, sheet_name='覆盖情况', index=False)
            
            # Sheet 3: 统计信息
            df_stats.to_excel(writer, sheet_name='统计信息', index=False)
            
            # Sheet 4: 关联分布
            df_dist.to_excel(writer, sheet_name='题目关联分布', index=False)
            
        print(f"✅ [2/2] Q-Matrix 增强版已保存: {q_matrix_filename}")
        print("   (包含页签: Q矩阵, 覆盖情况, 统计信息, 题目关联分布)")
        
    except Exception as e:
        print(f"❌ 保存 Excel 失败: {e}")

    # 完成后清理临时文件
    if os.path.exists(temp_save_path) and not TEST_MODE:
        os.remove(temp_save_path)
    
    duration = time.time() - start_time
    print("-" * 50)
    print(f"🎉 全部完成！总耗时 {duration:.2f} 秒")
if __name__ == "__main__":
    main()