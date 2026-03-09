# -*- coding: utf-8 -*-
"""
功能：
    1. 读取基础AIS和经验AIS，按经纬度范围互斥融合。
    2. 【新增】对读取的数据进行清洗（按MMSI和UnixTime去重、排序）。
    3. 对基础数据进行轨迹插值补全（生成新数据行）。
    4. 对补全后的完整数据进行统一排放计算。
    5. 支持“全量跑”或“指定船型跑”，直接在代码底部修改变量即可。
"""

import pandas as pd
import numpy as np
import time
import joblib
import os
import re

# ==============================================================================
# 1. 全局配置与常量
# ==============================================================================

# 基础路径 (请根据你的电脑修改这里)
BASE_PATH = "G:/Globalemission"

# 模型和数据路径
MODEL_PATH = f"{BASE_PATH}/modelsforAll"
MODEL_NO_TYPE_PATH = f"{BASE_PATH}/modelswithoutShipType"
STSD_FILE = f"{BASE_PATH}/STSD/finalSTSD-v2.csv"
MEAN_STSD_FILE = f"{BASE_PATH}/STSD/meanSTSD.csv"
CONTAINER_LIST_FILE = f"{BASE_PATH}/ContainerList.xlsx"

# 船舶类型映射表 (字符串 -> 数字ID)
SHIP_TYPES_TRANS = {
    'Port tender': 6, 'Pilot vessel': 6, 'Tug': 7, 'Tanker': 11,
    'Towing and length>200m or breadth>25m': 6, 'Medical transport': 6,
    'Passenger ship': 8, 'Enagged in military operations': 6,
    'Engaged in diving operations': 6, 'Sailing': 5, 'Pleasure craft': 6,
    'Spare-for assignments to local vessel': 6, 'Cargo ship': 5,
    'Fishing': 6, 'Search and rescue vessel': 6, 'WIG': 6,
    'Law enforcement vessel': 6, 'HSC': 6, 'Towing': 6,
    'Ship according to Resolution No 18(Mob-83)': 6,
    'Other type of ship': 6, 'Engaged in dredging or underwater operations': 6,
    'Vessel with anti-pollution facilities or equipment': 6,
    'Undefined': 6, 'Reserved': 6
}

# 默认需要跑的船型列表
DEFAULT_TYPE_LIST = [
    'Cargo ship', 'Tanker', 'Passenger ship', 'Tug', 'Fishing']

# 全局变量容器
MODELS = {}
STSD_DATA = {}


# ==============================================================================
# 2. 资源加载模块
# ==============================================================================

def load_resources():
    """加载模型和静态数据"""
    print("正在加载模型和静态数据...")

    # 1. 加载 GBRT 模型
    model_files = {
        'mds': 'Speedmax_gbrt_model.m', 'mcr': 'Powerkwmax_gbrt_model.m',
        'rpm': 'MainEngineRPM_gbrt_model.m', 'lng': 'FuelType_gbdt_model.m',
        'teu': 'container_gbrt_model.m', 'gt': 'GrossTonnage_gbrt_model.m'
    }
    for key, fname in model_files.items():
        MODELS[key] = joblib.load(f"{MODEL_PATH}/{fname}")

    # 2. 加载无船型模型
    model_nt_files = {
        'mds_nt': 'MDS_gbrt_model.pkl', 'mcr_nt': 'MCR_gbrt_model.pkl',
        'rpm_nt': 'RPM_gbrt_model.pkl', 'lng_nt': 'FuelType_gbdt_model.pkl',
        'teu_nt': 'TEU_gbrt_model.pkl', 'gt_nt': 'gt_gbrt_model.pkl'
    }
    for key, fname in model_nt_files.items():
        MODELS[key] = joblib.load(f"{MODEL_NO_TYPE_PATH}/{fname}")

    # 3. 加载 STSD 静态数据库
    stsd_df = pd.read_csv(STSD_FILE, index_col=0)
    STSD_DATA['IMO'] = {row['ID1']: row for _, row in stsd_df.iterrows()}
    STSD_DATA['MMSI'] = {row['ID2']: row for _, row in stsd_df.iterrows()}

    # 4. 加载均值数据
    mean_stsd_df = pd.read_csv(MEAN_STSD_FILE)
    STSD_DATA['MEAN'] = {row['ShipType']: row for _, row in mean_stsd_df.iterrows()}

    # 5. 加载集装箱船列表
    cont_df = pd.read_excel(CONTAINER_LIST_FILE, usecols=['imo', 'mmsi'])
    STSD_DATA['CONT_IMO'] = set(cont_df['imo'])
    STSD_DATA['CONT_MMSI'] = set(cont_df['mmsi'].dropna())

    print("资源加载完成。")


# ==============================================================================
# 3. 辅助计算函数
# ==============================================================================

def get_ship_params(imo, mmsi, ship_type, length, width):
    """获取船舶参数 (MCR, MDS等)"""
    # 优先查 MMSI
    row = STSD_DATA['MMSI'].get(mmsi)
    # 如果 MMSI 查不到，再查 IMO
    if row is None:
        row = STSD_DATA['IMO'].get(imo)
    if row is not None:
        lng = 2 if row['LNG'] == 2 else 1

        ae_power = str(row['AEPower']).split() if pd.notna(row['AEPower']) else [0, 0, 0, 0]
        boiler_power = str(row['BPower']).split() if pd.notna(row['BPower']) else [0, 0, 0, 0]

        return {
            'MCR': row['MCR'],
            'AEPower': ae_power,
            'BoilerPower': boiler_power,
            'MDS': row['SpeedMax'],
            'MEEF': row['MEEF'],
            'AEEF': row['AEEF'],
            'BoilerEF': 457 if lng == 2 else 970,
            'LLAF': row['LLAF']
        }

    # 均值/模型逻辑
    try:
        ship_type = int(ship_type)
    except:
        ship_type = 5  # 默认 Cargo

    mean_row = STSD_DATA['MEAN'].get(ship_type, STSD_DATA['MEAN'][5])

    ae_power_mean = str(mean_row['AEPower']).split()
    boiler_power_mean = str(mean_row['BPower']).split()

    return {
        'MCR': mean_row['MCR'],
        'AEPower': ae_power_mean,
        'BoilerPower': boiler_power_mean,
        'MDS': mean_row['SpeedMax'],
        'MEEF': mean_row['MEEF'],
        'AEEF': mean_row['AEEF'],
        'BoilerEF': 970,
        'LLAF': 1
    }


def get_activity_mode(speed, lf):
    """根据速度和负荷判断工况"""
    if speed < 1:
        return 0  # Hoteling
    elif speed < 3:
        return 1  # Anchoring
    else:
        return 2 if lf < 0.2 else 3  # Maneuvering / Cruising


def parse_date_from_filename(filename):
    """从文件名解析年月"""
    basename = os.path.basename(filename)
    match = re.search(r'(\d{4})[-_](\d{2})', basename)
    if match:
        return match.group(1), match.group(2)
    return 'Unknown', 'Unknown'


# ==============================================================================
# 4. 数据清洗模块 (新增)
# ==============================================================================

def clean_ais_data(df):
    """
    对 AIS 数据进行标准清洗：
    1. 去重：基于 MMSI 和 UnixTime，保留第一条。
    2. 排序：基于 MMSI 和 UnixTime 升序排列。
    """
    if df is None or df.empty:
        return df

    # 去重
    # subset指定列重复即视为重复，keep='first'保留第一次出现的
    df.drop_duplicates(subset=['MMSI', 'UnixTime'], keep='first', inplace=True)

    # 排序
    df.sort_values(by=['MMSI', 'UnixTime'], ascending=[True, True], inplace=True)

    # 重置索引，防止后续遍历出错
    df.reset_index(drop=True, inplace=True)

    return df


# ==============================================================================
# 5. 数据读取与融合模块
# ==============================================================================

def load_and_merge_ais(base_file, exp_file, target_ship_type):
    """
    逻辑：
    1. 读取基础AIS。
    2. 如果 target_ship_type 不是 'All'，则只保留该船型。
    3. 剔除 EXP_ZONE 范围内的基础AIS点。
    4. 读取经验AIS，保留 EXP_ZONE 范围内的点。
    5. 合并并按 MMSI, Time 排序。
    """
    is_all_mode = (target_ship_type == 'All')
    print(f"--> 读取并融合数据...")

    try:
        # 1. 读取基础数据
        use_cols = ['MMSI', 'IMO', 'UnixTime', 'Lon_d', 'Lat_d', 'Speed', 'Course', 'ShipTypeEN', 'Length', 'Width']
        df_base = pd.read_csv(base_file, encoding="gbk", usecols=use_cols)

        # [清洗] 基础数据去重排序
        df_base = clean_ais_data(df_base)

        # 筛选船型
        if not is_all_mode:
            df_base = df_base[df_base['ShipTypeEN'] == target_ship_type].copy()

        # 空间剔除 (删除经验区域内的点)
        if not df_base.empty:
            in_zone = (df_base['Lon_d'].between(EXP_ZONE['lon_min'], EXP_ZONE['lon_max'])) & \
                      (df_base['Lat_d'].between(EXP_ZONE['lat_min'], EXP_ZONE['lat_max']))
            df_base = df_base[~in_zone].copy()
            df_base['is_exp'] = 0  # 标记为基础数据

        # 2. 读取经验数据
        df_exp = pd.DataFrame()
        if os.path.exists(exp_file):
            print("   -> 读取经验数据...")
            df_exp = pd.read_csv(exp_file, encoding="gbk")

            # [清洗] 经验数据去重排序
            df_exp = clean_ais_data(df_exp)

            # 筛选船型 (经验数据可能也包含多种船型)
            if not is_all_mode:
                if 'ShipTypeEN' in df_exp.columns:
                    df_exp = df_exp[df_exp['ShipTypeEN'] == target_ship_type].copy()
                else:
                    # 如果经验数据没船型列，默认它符合要求
                    df_exp['ShipTypeEN'] = target_ship_type

            # 空间保留 (只取经验区域内的点)
            if not df_exp.empty:
                in_zone_exp = (df_exp['Lon_d'].between(EXP_ZONE['lon_min'], EXP_ZONE['lon_max'])) & \
                              (df_exp['Lat_d'].between(EXP_ZONE['lat_min'], EXP_ZONE['lat_max']))
                df_exp = df_exp[in_zone_exp].copy()
                df_exp['is_exp'] = 1  # 标记为经验数据

        # 3. 合并
        df_merged = pd.concat([df_base, df_exp], ignore_index=True)
        if df_merged.empty:
            return df_merged

        # [清洗] 合并后再次去重排序，确保轨迹绝对连续
        df_merged = clean_ais_data(df_merged)

        return df_merged

    except Exception as e:
        print(f"数据读取失败: {e}")
        return pd.DataFrame()


# ==============================================================================
# 6. 轨迹插值补全模块 (利用插值函数补全)
# ==============================================================================

def interpolate_gaps(df):
    """
    遍历 DataFrame，对于长间隔的基础数据段进行 Hermite 插值，
    生成新的点并插入。
    """
    if df.empty: return df
    print("--> 检查断点并执行插值补全...")

    # 预先计算差值列，方便判断
    df['deltaT'] = df['UnixTime'] - df['UnixTime'].shift(1)
    df['changeMMSI'] = df['MMSI'] != df['MMSI'].shift(1)

    # 结果容器
    new_rows = []

    # 使用 itertuples 进行高效遍历
    # row 是一个 namedtuple，可以通过 row.Lon_d 访问
    # 为了能访问“上一行”，我们需要缓存
    prev_row = None

    for row in df.itertuples(index=False):
        # 转换为字典方便后续修改和添加 (单行转换开销很小)
        current_dict = row._asdict()
        new_rows.append(current_dict)

        # 如果是第一行或换了船，无法插值
        if prev_row is None or row.changeMMSI:
            prev_row = row
            continue

        delta_t = row.deltaT

        # ==================================================================
        # 核心判断逻辑：
        # 1. 是基础数据 (row.is_exp == 0) -> 允许插值
        # 2. 是经验数据 (row.is_exp == 1) -> 不插值 (认为是已经处理好的)
        # 3. 时间间隔在合理范围内 (600s < dt <= 3600s)
        # ==================================================================

        # 注意：这里我们取当前行的 is_exp 状态作为判断依据
        is_candidate = (row.is_exp == 0) and (600 < delta_t <= 3600)
        speed_ok = (row.Speed > 0.5) and (prev_row.Speed > 0.5)

        if is_candidate and speed_ok:
            # 提取前后点参数
            t0, t1 = prev_row.UnixTime, row.UnixTime
            lon0, lat0, spd0, cog0 = prev_row.Lon_d, prev_row.Lat_d, prev_row.Speed, prev_row.Course
            lon1, lat1, spd1, cog1 = row.Lon_d, row.Lat_d, row.Speed, row.Course

            # 速度平滑处理
            R, K = 0.1, 0.5
            if spd0 < spd1 and spd0 / spd1 < R:
                spd0 = spd1 * K
            elif spd0 > spd1 and spd1 / spd0 < R:
                spd1 = spd0 * K

            # 计算需要插入的点数
            nums = int(np.floor(delta_t / 600))
            dt_step = delta_t / (nums + 1)

            # 经纬度转换因子
            dist_lat = 111000
            dist_lon = 111000 * np.cos(np.radians(lat1))

            # 生成中间点
            for k in range(1, nums + 1):
                ratio = k / (nums + 1)

                # 时间插值
                t_new = t0 + k * dt_step
                # 速度插值
                spd_new = spd0 + ratio * (spd1 - spd0)

                # 经纬度插值 (Hermite / 加权推算)
                time_diff_1 = ratio * delta_t
                l1_lon = lon0 + spd0 * 0.5144 * np.sin(np.radians(cog0)) * time_diff_1 / dist_lon
                l1_lat = lat0 + spd0 * 0.5144 * np.cos(np.radians(cog0)) * time_diff_1 / dist_lat

                time_diff_2 = (ratio - 1) * delta_t
                l2_lon = lon1 + spd1 * 0.5144 * np.sin(np.radians(cog1)) * time_diff_2 / dist_lon
                l2_lat = lat1 + spd1 * 0.5144 * np.cos(np.radians(cog1)) * time_diff_2 / dist_lat

                w1 = 1 - ratio
                w2 = ratio

                lon_new = w1 * l1_lon + w2 * l2_lon
                lat_new = w1 * l1_lat + w2 * l2_lat

                # 构建新点 (复制当前点的属性，修改时空信息)
                new_point = current_dict.copy()
                new_point['UnixTime'] = t_new
                new_point['Lon_d'] = lon_new
                new_point['Lat_d'] = lat_new
                new_point['Speed'] = spd_new
                new_point['Course'] = cog0  # 航向简化沿用前点
                new_point['is_exp'] = 2  # 标记：这是自动插值生成的

                new_rows.append(new_point)

        # 更新 prev_row
        prev_row = row

    # 将列表重新转回 DataFrame
    df_dense = pd.DataFrame(new_rows)

    # 再次排序，确保插入的点在正确的时间位置
    df_dense = clean_ais_data(df_dense)  # 这里也可以复用清洗函数确保顺序

    return df_dense


# ==============================================================================
# 7. 统一排放计算模块
# ==============================================================================

def calculate_emission_grid(df, grid_size=0.01):
    """
    对完整数据进行排放计算。
    此时不区分 is_exp 0,1,2，只要有 deltaT 就计算。
    """
    print("--> 计算排放矩阵...")

    # 初始化网格
    min_lat, max_lat = -90, 90
    min_lon, max_lon = -180, 180
    rows = int(np.ceil((max_lat - min_lat) / grid_size)) + 1
    cols = int(np.ceil((max_lon - min_lon) / grid_size)) + 1
    emission_grid = np.zeros((rows, cols))

    # 重新计算 deltaT (因为插入了新点，或者顺序变了)
    df['deltaT'] = df['UnixTime'] - df['UnixTime'].shift(1)
    df['changeMMSI'] = df['MMSI'] != df['MMSI'].shift(1)

    # 映射船型ID (fillna处理防止全量模式下的未知类型报错)
    df['ShipTypeID'] = df['ShipTypeEN'].map(SHIP_TYPES_TRANS).fillna(6).astype(int)

    ship_params = {}
    current_mmsi = None

    # 使用 itertuples 遍历计算
    for row in df.itertuples(index=False):
        # 换船或第一行
        if row.changeMMSI or pd.isna(row.deltaT):
            current_mmsi = row.MMSI
            imo = row.IMO
            # 集装箱船特殊判定
            stype = 3 if (current_mmsi in STSD_DATA['CONT_MMSI']) else row.ShipTypeID
            ship_params = get_ship_params(imo, current_mmsi, stype, row.Length, row.Width)
            continue

        delta_t = row.deltaT

        # 异常过滤：过大的断层(>3600)依然不计算
        if delta_t > 3600 or delta_t <= 0:
            continue

        # 空间映射
        row_i = int(rows - np.ceil((row.Lat_d - min_lat) / grid_size) - 1)
        col_j = int(np.ceil((row.Lon_d - min_lon) / grid_size))

        if 0 <= row_i < rows and 0 <= col_j < cols:
            mds = ship_params['MDS']
            speed = row.Speed

            # 物理公式
            lf = pow(speed / mds, 3) if mds > 0 else 0.8
            act_mode = get_activity_mode(speed, lf)
            factor = delta_t * 1e-6 / 3600  # 转换为吨

            e_me = ship_params['MCR'] * ship_params['MEEF'] * lf * ship_params['LLAF'] * factor
            e_bo = float(ship_params['BoilerPower'][act_mode]) * ship_params['BoilerEF'] * factor
            e_ae = float(ship_params['AEPower'][act_mode]) * ship_params['AEEF'] * factor

            emission_grid[row_i][col_j] += (e_me + e_bo + e_ae)

    return emission_grid


# ==============================================================================
# 8. 主运行入口 (在这里修改配置)
# ==============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 用户配置区域 (请在这里修改)
    # ------------------------------------------------------------------

    # 1. 输入文件路径
    input_ais_file = f"G:\Globalemission\AAA经验插值碳排放计算\经验插值_武汉.csv"
    exp_file = f"G:\Globalemission\AAA经验插值碳排放计算\经验插值_武汉.csv"  # 经验数据文件夹

    # 经验数据优先区域 (在此区域内，强制使用经验数据，剔除基础数据)
    EXP_ZONE = {
        'lon_min': 113.7, 'lon_max': 114.2,
        'lat_min': 29.9, 'lat_max': 30.4
    }

    # 2. 运行模式设置
    #    'All'          -> 全量模式 (所有船型混合在一起跑)
    #    'Cargo ship'   -> 单船型模式 (只跑 Cargo ship)
    #    None           -> 列表模式 (跑 DEFAULT_TYPE_LIST 里的所有船型)
    RUN_MODE = 'All'
    # ------------------------------------------------------------------
    # 自动执行逻辑
    # ------------------------------------------------------------------

    load_resources()

    # 确定要处理的列表
    if RUN_MODE == 'All':
        process_list = ['All']
    elif RUN_MODE:
        process_list = [RUN_MODE]
    else:
        process_list = DEFAULT_TYPE_LIST
        print("未指定 RUN_MODE，将批量运行默认船型列表。")

    print(f"开始处理，模式: {process_list}")

    for s_type in process_list:
        print(f"\n[{s_type}] 任务启动...")
        t0 = time.time()

        # 1. 读取融合
        df = load_and_merge_ais(input_ais_file, exp_file, s_type)

        if not df.empty:
            # 2. 未经验插值的数据插值补全
            df = interpolate_gaps(df)

            # 3. 排放计算
            grid = calculate_emission_grid(df)

            # 4. 保存结果
            folder_name = "All" if s_type == 'All' else s_type.replace('/', '_')
            out_file = f"Results/{folder_name}/try.csv"

            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            pd.DataFrame(grid).to_csv(out_file, sep=',', index=False, header=False)

            print(f"[{s_type}] 成功保存至: {out_file}")
        else:
            print(f"[{s_type}] 无有效数据，跳过。")

        print(f"[{s_type}] 耗时: {time.time() - t0:.2f}s")

    print("\n所有任务完成。")