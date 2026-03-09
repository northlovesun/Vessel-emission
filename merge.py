import pandas as pd


def merge_gdp_to_master(master_csv_path, gdp_data_path, output_csv_path):
    print("1. 正在严格以纯文本模式读取表格，防止 ID 变形...")

    # 核心修改点 1：使用 dtype={'outID': str} 强制把 outID 当作文本读取
    master_df = pd.read_csv(master_csv_path, dtype={'OutID': str}, encoding='gbk')

    # gdp_df 是我们刚用 Python 生成的纯净版，默认是 utf-8，如果它也报同样的错，也加上 encoding='gbk'
    # 为了保险，这里我们可以先不给 gdp_df 加，或者使用 utf-8-sig 兼容模式
    try:
        gdp_df = pd.read_csv(gdp_data_path, dtype={'OutID': str})
    except UnicodeDecodeError:
        gdp_df = pd.read_csv(gdp_data_path, dtype={'OutID': str}, encoding='gbk')

    # 核心修改点 2：如果之前保存时不幸带上了 ".0" 尾巴（比如 11000001.0），将其安全擦除
    # 确保两边的暗号格式绝对统一
    master_df['OutID'] = master_df['OutID'].str.replace(r'\.0$', '', regex=True)
    gdp_df['OutID'] = gdp_df['OutID'].str.replace(r'\.0$', '', regex=True)

    print("2. 正在根据纯文本 OutID 进行精准匹配 (Left Join)...")
    merged_df = pd.merge(master_df, gdp_df, on='OutID', how='left')

    print("3. 匹配完成，正在保存最终结果...")
    # 保存时数据将保持纯文本的原汁原味
    merged_df.to_csv(output_csv_path, index=False)

    print(f"大功告成！文件已保存为：{output_csv_path}")
    print(f"合并前目标表行数：{len(master_df)}")
    print(f"合并后最终表行数：{len(merged_df)}")


# ====================
# 执行区域
# ====================
if __name__ == "__main__":
    MASTER_CSV = "result/Final_Merged_Result(4).csv"
    GDP_DATA_CSV = "POP/result/Global_POP_2022_Cleaned.csv"
    OUTPUT_CSV = "result/Final_Merged_Result_withPOP.csv"

    merge_gdp_to_master(MASTER_CSV, GDP_DATA_CSV, OUTPUT_CSV)