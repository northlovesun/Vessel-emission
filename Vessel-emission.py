# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time
import joblib
import os
import re



BASE_PATH = "G:/Globalemission"

MODEL_PATH = f"{BASE_PATH}/modelsforAll"
MODEL_NO_TYPE_PATH = f"{BASE_PATH}/modelswithoutShipType"
STSD_FILE = f"{BASE_PATH}/STSD/finalSTSD-v2.csv"
MEAN_STSD_FILE = f"{BASE_PATH}/STSD/meanSTSD.csv"
CONTAINER_LIST_FILE = f"{BASE_PATH}/ContainerList.xlsx"

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

DEFAULT_TYPE_LIST = [
    'Cargo ship', 'Tanker', 'Passenger ship', 'Tug', 'Fishing']

MODELS = {}
STSD_DATA = {}


def load_resources():

    model_files = {
        'mds': 'Speedmax_gbrt_model.m', 'mcr': 'Powerkwmax_gbrt_model.m',
        'rpm': 'MainEngineRPM_gbrt_model.m', 'lng': 'FuelType_gbdt_model.m',
        'teu': 'container_gbrt_model.m', 'gt': 'GrossTonnage_gbrt_model.m'
    }
    for key, fname in model_files.items():
        MODELS[key] = joblib.load(f"{MODEL_PATH}/{fname}")

    model_nt_files = {
        'mds_nt': 'MDS_gbrt_model.pkl', 'mcr_nt': 'MCR_gbrt_model.pkl',
        'rpm_nt': 'RPM_gbrt_model.pkl', 'lng_nt': 'FuelType_gbdt_model.pkl',
        'teu_nt': 'TEU_gbrt_model.pkl', 'gt_nt': 'gt_gbrt_model.pkl'
    }
    for key, fname in model_nt_files.items():
        MODELS[key] = joblib.load(f"{MODEL_NO_TYPE_PATH}/{fname}")

    stsd_df = pd.read_csv(STSD_FILE, index_col=0)
    STSD_DATA['IMO'] = {row['ID1']: row for _, row in stsd_df.iterrows()}
    STSD_DATA['MMSI'] = {row['ID2']: row for _, row in stsd_df.iterrows()}

    mean_stsd_df = pd.read_csv(MEAN_STSD_FILE)
    STSD_DATA['MEAN'] = {row['ShipType']: row for _, row in mean_stsd_df.iterrows()}

    cont_df = pd.read_excel(CONTAINER_LIST_FILE, usecols=['imo', 'mmsi'])
    STSD_DATA['CONT_IMO'] = set(cont_df['imo'])
    STSD_DATA['CONT_MMSI'] = set(cont_df['mmsi'].dropna())




def get_ship_params(imo, mmsi, ship_type, length, width):

    row = STSD_DATA['MMSI'].get(mmsi)

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

    try:
        ship_type = int(ship_type)
    except:
        ship_type = 5

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
    if speed < 1:
        return 0
    elif speed < 3:
        return 1
    else:
        return 2 if lf < 0.2 else 3


def parse_date_from_filename(filename):

    basename = os.path.basename(filename)
    match = re.search(r'(\d{4})[-_](\d{2})', basename)
    if match:
        return match.group(1), match.group(2)
    return 'Unknown', 'Unknown'


def clean_ais_data(df):
    if df is None or df.empty:
        return df

    df.drop_duplicates(subset=['MMSI', 'UnixTime'], keep='first', inplace=True)

    df.sort_values(by=['MMSI', 'UnixTime'], ascending=[True, True], inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df


def load_and_merge_ais(base_file, exp_file, target_ship_type):

    is_all_mode = (target_ship_type == 'All')

    try:

        use_cols = ['MMSI', 'IMO', 'UnixTime', 'Lon_d', 'Lat_d', 'Speed', 'Course', 'ShipTypeEN', 'Length', 'Width']
        df_base = pd.read_csv(base_file, encoding="gbk", usecols=use_cols)


        df_base = clean_ais_data(df_base)


        if not is_all_mode:
            df_base = df_base[df_base['ShipTypeEN'] == target_ship_type].copy()


        if not df_base.empty:
            in_zone = (df_base['Lon_d'].between(EXP_ZONE['lon_min'], EXP_ZONE['lon_max'])) & \
                      (df_base['Lat_d'].between(EXP_ZONE['lat_min'], EXP_ZONE['lat_max']))
            df_base = df_base[~in_zone].copy()
            df_base['is_exp'] = 0


        df_exp = pd.DataFrame()
        if os.path.exists(exp_file):

            df_exp = pd.read_csv(exp_file, encoding="gbk")


            df_exp = clean_ais_data(df_exp)


            if not is_all_mode:
                if 'ShipTypeEN' in df_exp.columns:
                    df_exp = df_exp[df_exp['ShipTypeEN'] == target_ship_type].copy()
                else:

                    df_exp['ShipTypeEN'] = target_ship_type


            if not df_exp.empty:
                in_zone_exp = (df_exp['Lon_d'].between(EXP_ZONE['lon_min'], EXP_ZONE['lon_max'])) & \
                              (df_exp['Lat_d'].between(EXP_ZONE['lat_min'], EXP_ZONE['lat_max']))
                df_exp = df_exp[in_zone_exp].copy()
                df_exp['is_exp'] = 1


        df_merged = pd.concat([df_base, df_exp], ignore_index=True)
        if df_merged.empty:
            return df_merged

        df_merged = clean_ais_data(df_merged)

        return df_merged

    except Exception as e:

        return pd.DataFrame()


def interpolate_gaps(df):

    if df.empty: return df

    # 预先计算差值列，方便判断
    df['deltaT'] = df['UnixTime'] - df['UnixTime'].shift(1)
    df['changeMMSI'] = df['MMSI'] != df['MMSI'].shift(1)

    new_rows = []

    prev_row = None

    for row in df.itertuples(index=False):

        current_dict = row._asdict()
        new_rows.append(current_dict)


        if prev_row is None or row.changeMMSI:
            prev_row = row
            continue

        delta_t = row.deltaT


        is_candidate = (row.is_exp == 0) and (600 < delta_t <= 3600)
        speed_ok = (row.Speed > 0.5) and (prev_row.Speed > 0.5)

        if is_candidate and speed_ok:

            t0, t1 = prev_row.UnixTime, row.UnixTime
            lon0, lat0, spd0, cog0 = prev_row.Lon_d, prev_row.Lat_d, prev_row.Speed, prev_row.Course
            lon1, lat1, spd1, cog1 = row.Lon_d, row.Lat_d, row.Speed, row.Course


            R, K = 0.1, 0.5
            if spd0 < spd1 and spd0 / spd1 < R:
                spd0 = spd1 * K
            elif spd0 > spd1 and spd1 / spd0 < R:
                spd1 = spd0 * K


            nums = int(np.floor(delta_t / 600))
            dt_step = delta_t / (nums + 1)

            dist_lat = 111000
            dist_lon = 111000 * np.cos(np.radians(lat1))

            for k in range(1, nums + 1):
                ratio = k / (nums + 1)


                t_new = t0 + k * dt_step

                spd_new = spd0 + ratio * (spd1 - spd0)

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

                new_point = current_dict.copy()
                new_point['UnixTime'] = t_new
                new_point['Lon_d'] = lon_new
                new_point['Lat_d'] = lat_new
                new_point['Speed'] = spd_new
                new_point['Course'] = cog0
                new_point['is_exp'] = 2

                new_rows.append(new_point)


        prev_row = row


    df_dense = pd.DataFrame(new_rows)

    df_dense = clean_ais_data(df_dense)

    return df_dense


def calculate_emission_grid(df, grid_size=0.01):



    min_lat, max_lat = -90, 90
    min_lon, max_lon = -180, 180
    rows = int(np.ceil((max_lat - min_lat) / grid_size)) + 1
    cols = int(np.ceil((max_lon - min_lon) / grid_size)) + 1
    emission_grid = np.zeros((rows, cols))


    df['deltaT'] = df['UnixTime'] - df['UnixTime'].shift(1)
    df['changeMMSI'] = df['MMSI'] != df['MMSI'].shift(1)

    df['ShipTypeID'] = df['ShipTypeEN'].map(SHIP_TYPES_TRANS).fillna(6).astype(int)

    ship_params = {}
    current_mmsi = None

    for row in df.itertuples(index=False):

        if row.changeMMSI or pd.isna(row.deltaT):
            current_mmsi = row.MMSI
            imo = row.IMO

            stype = 3 if (current_mmsi in STSD_DATA['CONT_MMSI']) else row.ShipTypeID
            ship_params = get_ship_params(imo, current_mmsi, stype, row.Length, row.Width)
            continue

        delta_t = row.deltaT


        if delta_t > 3600 or delta_t <= 0:
            continue

        row_i = int(rows - np.ceil((row.Lat_d - min_lat) / grid_size) - 1)
        col_j = int(np.ceil((row.Lon_d - min_lon) / grid_size))

        if 0 <= row_i < rows and 0 <= col_j < cols:
            mds = ship_params['MDS']
            speed = row.Speed


            lf = pow(speed / mds, 3) if mds > 0 else 0.8
            act_mode = get_activity_mode(speed, lf)
            factor = delta_t * 1e-6 / 3600  # 转换为吨

            e_me = ship_params['MCR'] * ship_params['MEEF'] * lf * ship_params['LLAF'] * factor
            e_bo = float(ship_params['BoilerPower'][act_mode]) * ship_params['BoilerEF'] * factor
            e_ae = float(ship_params['AEPower'][act_mode]) * ship_params['AEEF'] * factor

            emission_grid[row_i][col_j] += (e_me + e_bo + e_ae)

    return emission_grid




if __name__ == "__main__":

    input_ais_file = f"G:\Globalemission\AAA经验插值碳排放计算\经验插值_武汉.csv"
    exp_file = f"G:\Globalemission\AAA经验插值碳排放计算\经验插值_武汉.csv"


    EXP_ZONE = {
        'lon_min': 113.7, 'lon_max': 114.2,
        'lat_min': 29.9, 'lat_max': 30.4
    }


    RUN_MODE = 'All'


    load_resources()


    if RUN_MODE == 'All':
        process_list = ['All']
    elif RUN_MODE:
        process_list = [RUN_MODE]
    else:
        process_list = DEFAULT_TYPE_LIST




    for s_type in process_list:

        t0 = time.time()


        df = load_and_merge_ais(input_ais_file, exp_file, s_type)

        if not df.empty:

            df = interpolate_gaps(df)


            grid = calculate_emission_grid(df)


            folder_name = "All" if s_type == 'All' else s_type.replace('/', '_')
            out_file = f"Results/{folder_name}/try.csv"

            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            pd.DataFrame(grid).to_csv(out_file, sep=',', index=False, header=False)

            print(f"[{s_type}] 成功保存至: {out_file}")
        else:
            print(f"[{s_type}] 无有效数据，跳过。")

        print(f"[{s_type}] 耗时: {time.time() - t0:.2f}s")

    print("\n所有任务完成。")