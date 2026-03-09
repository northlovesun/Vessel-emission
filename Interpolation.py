# -*- coding: utf-8 -*-
import pandas as pd
import json
import numpy as np
import math
import warnings
from geopy.distance import geodesic
from multiprocessing import Pool, cpu_count, Lock
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from fastdtw import fastdtw
from numba import njit
import os
import csv
from tqdm import tqdm

# ===================== 全局配置 =====================
TARGET_EXTENT = [113.7, 114.2, 29.9, 30.4]
# 聚类参数
CLUSTER_EPS = 1.2
CLUSTER_MIN_SAMPLES = 5
CLUSTER_TOP_PERCENTAGE = 0.3
# 插值参数
MAX_DIST_KM = 1
MAX_FAR_POINTS = 20
POST_INTERP_MAX_DIST_KM = 1
# 输出路径
FINAL_OUTPUT_JSON = r"G:\Globalemission\AAA经验插值碳排放计算\经验插值_武汉.json"
FINAL_OUTPUT_CSV = r"G:\Globalemission\AAA经验插值碳排放计算\经验插值_武汉.csv"
# 原始数据路径
RAW_CSV_PATH = r"G:\HQ\NCC\2022-08_武汉.csv"

# 全局锁（解决多进程写入文件冲突）
file_lock = Lock()
warnings.filterwarnings('ignore')


# ===================== 阶段1：数据清洗函数 =====================
def geo_dis(p1, p2):
    """计算两点间地理距离（海里）"""
    return geodesic((p1[1], p1[0]), (p2[1], p2[0])).nautical


def find_large_gaps_indices(sequence):
    """查找时间序列中的大间隔"""
    result = []
    indexlist = [0]
    for i in range(len(sequence) - 1):
        diff = sequence[i + 1] - sequence[i]
        if diff > 3600:
            indexlist.append(i + 1)
    indexlist.append(len(sequence))
    for j in range(len(indexlist) - 1):
        result.append([indexlist[j], indexlist[j + 1]])
    return result


def find_segments(lst):
    """查找连续段"""
    if not lst:
        return []
    segments = []
    start = lst[0]
    prev = lst[0]
    for num in lst[1:]:
        if num - prev < 10:
            prev = num
        else:
            segments.append((start, prev + 1))
            start = num
            prev = num
    segments.append((start, prev + 1))
    newsegments = [seg for seg in segments if seg[1] - seg[0] > 9]
    return newsegments


def is_point_in_extent(lon, lat, extent):
    """判断单个点是否在目标区域内"""
    min_lon, max_lon, min_lat, max_lat = extent
    return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)


def calculate_course_diff(course1, course2):
    """计算航向差（处理0-360度环形，返回最小差值）"""
    diff = abs(course1 - course2)
    return diff if diff <= 180 else 360 - diff


def split_trajectory_by_conditions(traj):
    """仅保留航向变化分段逻辑"""
    if len(traj) < 3:
        return [traj]

    courses = [p[3] for p in traj]
    split_indices = {0, len(traj) - 1}

    for i in range(len(traj) - 1):
        diff = calculate_course_diff(courses[i], courses[i + 1])
        if diff > 90:
            split_indices.add(i + 1)

    split_indices = sorted(list(split_indices))
    sub_trajs = []
    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1] + 1
        sub_traj = traj[start_idx:end_idx]
        if len(sub_traj) >= 10:
            sub_trajs.append(sub_traj)

    return sub_trajs


def process_single_sub_trajectory(sub_traj, mmsi, st):
    """处理单个子轨迹（校验+简化）"""
    try:
        # 时间戳去重
        sub_traj_dedup = []
        prev_unix_time = None
        for point in sub_traj:
            unix_time = point[4]
            if unix_time != prev_unix_time:
                sub_traj_dedup.append(point)
            prev_unix_time = unix_time
        if len(sub_traj_dedup) < 5:
            return None

        sub_traj_downsampled = [sub_traj_dedup[0]]
        for p in sub_traj_dedup[1:]:
            if p[4] - sub_traj_downsampled[-1][4] > 15:
                sub_traj_downsampled.append(p)
        if len(sub_traj_downsampled) < 5:
            return None

        timelist = [p[4] for p in sub_traj_downsampled]
        time_diffs = np.diff(timelist)
        if np.any(time_diffs > 3600):
            return None

        try:
            from DPsuanfa import simplify
            sub_traj_simplified = simplify(sub_traj_downsampled, 20)
        except ImportError:
            sub_traj_simplified = sub_traj_downsampled

        if len(sub_traj_simplified) >= 5:
            sub_traj_simplified.insert(0, [mmsi, st])
            return sub_traj_simplified
    except Exception:
        pass
    return None


def process_single_mmsi(args):
    """处理单个MMSI的数据"""
    mmsi, g, shiptype, extent = args

    try:
        if len(str(mmsi)) != 9:
            return None

        st = g['ShipTypeEN'].iloc[0] if len(g) > 0 else None
        if st not in shiptype:
            return None

        # 区域过滤
        g_filtered = g[
            (g['Lon_d'] >= extent[0]) & (g['Lon_d'] <= extent[1]) &
            (g['Lat_d'] >= extent[2]) & (g['Lat_d'] <= extent[3])
            ]
        if len(g_filtered) < 5:
            return None

        # 提取轨迹数据
        time = g_filtered['UnixTime'].tolist()
        lon = g_filtered['Lon_d'].tolist()
        lat = g_filtered['Lat_d'].tolist()
        newlon = [round(num, 6) for num in lon]
        newlat = [round(num, 6) for num in lat]
        speed = g_filtered['Speed'].tolist()
        course = g_filtered['Course'].tolist()

        tralist = []
        for i in range(len(newlon)):
            if 0.5 <= speed[i] <= 40:
                tralist.append((newlon[i], newlat[i], speed[i], course[i], time[i]))
        if len(tralist) < 10:
            return None

        # 按航向变化拆分轨迹
        sub_trajs = split_trajectory_by_conditions(tralist)
        if not sub_trajs:
            return None

        # 处理每个子轨迹
        valid_sub_trajs = []
        for sub_traj in sub_trajs:
            processed_sub_traj = process_single_sub_trajectory(sub_traj, mmsi, st)
            if processed_sub_traj is not None:
                valid_sub_trajs.append(processed_sub_traj)

        return valid_sub_trajs

    except Exception:
        return None


def clean_trajectory_data(csv_path, extent, shiptype):

    df = pd.read_csv(csv_path,
                     usecols=['MMSI', 'ShipTypeEN', 'UnixTime', 'Lon_d', 'Lat_d', 'Speed', 'Course'],
                     encoding='ISO-8859-1')

    # 分组处理
    group = df.groupby('MMSI')
    total_groups = len(group)
    print(f"总MMSI分组数量：{total_groups}")

    # 准备多进程参数
    args_list = [(mmsi, g, shiptype, extent) for mmsi, g in group]

    # 多进程处理
    num_processes = min(cpu_count(), 30)
    tras_list = []

    with Pool(processes=num_processes) as pool:
        with tqdm(total=total_groups, desc="清洗轨迹数据", unit="MMSI") as pbar:
            for result in pool.imap_unordered(process_single_mmsi, args_list):
                pbar.update(1)
                if result is not None and len(result) > 0:
                    tras_list.extend(result)

    return tras_list


# ===================== 阶段2：轨迹聚类融合函数=====================
@njit
def euclid_dis(x, y):
    """计算两点间欧氏距离（聚类用）"""
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def fastdtw_estimate(A, B):
    """使用FastDTW计算轨迹间距离"""
    distance, _ = fastdtw(A, B, dist=euclid_dis)
    return distance


def compute_distance_matrix(data):
    """并行计算距离矩阵"""
    num = len(data)
    distance_matrix = np.zeros((num, num))
    tasks = [(i, j) for i in range(num) for j in range(i, num)]
    results = []

    with tqdm(total=len(tasks), desc="计算距离矩阵", unit="对") as pbar:
        for i, j in tasks:
            results.append(fastdtw_estimate(data[i], data[j]))
            pbar.update(1)

    idx = 0
    for i in range(num):
        for j in range(i, num):
            distance_matrix[i, j] = results[idx]
            distance_matrix[j, i] = results[idx]
            idx += 1

    return distance_matrix


def ddbscan_optimized(data, eps, min_samples, distance_matrix):
    """优化的DDbscan聚类算法"""
    num = len(data)
    unvisited = set(range(num))
    C = np.full(num, -1, dtype=int)
    k = -1

    with tqdm(total=len(unvisited), desc="DDBSCAN聚类", unit="轨迹") as pbar:
        while unvisited:
            p = unvisited.pop()
            neighbors = np.where(distance_matrix[p] <= eps)[0].tolist()

            if len(neighbors) >= min_samples:
                k += 1
                C[p] = k
                for pi in neighbors:
                    if pi in unvisited:
                        unvisited.remove(pi)
                        pi_neighbors = np.where(distance_matrix[pi] <= eps)[0].tolist()
                        if len(pi_neighbors) >= min_samples:
                            neighbors.extend(pi_neighbors)
                        if C[pi] == -1:
                            C[pi] = k
            else:
                C[p] = -1
            pbar.update(1)

    return C


def normalize_trajectory_to_fixed_length(traj, target_length):
    if len(traj) < 2:
        return []

    lon = [p[0] for p in traj]
    lat = [p[1] for p in traj]

    # 累积距离
    distances = np.zeros(len(lon))
    for i in range(1, len(lon)):
        dx = lon[i] - lon[i - 1]
        dy = lat[i] - lat[i - 1]
        distances[i] = distances[i - 1] + math.sqrt(dx * dx + dy * dy)

    if distances[-1] == 0:
        return [traj[0][:2]] * target_length

    normalized_distances = distances / distances[-1]

    f_lon = interp1d(normalized_distances, lon, kind='linear', bounds_error=False, fill_value="extrapolate")
    f_lat = interp1d(normalized_distances, lat, kind='linear', bounds_error=False, fill_value="extrapolate")
    sample_points = np.linspace(0, 1, target_length)

    interp_lon = f_lon(sample_points)
    interp_lat = f_lat(sample_points)
    normalized_traj = [[x, y] for x, y in zip(interp_lon, interp_lat)]

    return normalized_traj


def normalize_trajectories_in_cluster(traj_list):
    if not traj_list:
        return []

    max_length = max(len(traj) for traj in traj_list)
    normalized_trajs = []

    with tqdm(total=len(traj_list), desc="轨迹归一化", unit="条") as pbar:
        for traj in traj_list:
            norm_traj = normalize_trajectory_to_fixed_length(traj, max_length)
            if norm_traj:
                normalized_trajs.append(norm_traj)
            pbar.update(1)

    return normalized_trajs


def merge_trajectories_by_average(traj_list):
    if not traj_list:
        return [], []

    point_count = len(traj_list[0])
    merged_lon = []
    merged_lat = []

    with tqdm(total=point_count, desc="轨迹融合", unit="点") as pbar:
        for i in range(point_count):
            lons = []
            lats = []
            for traj in traj_list:
                point = traj[i]
                lons.append(point[0])
                lats.append(point[1])
            if lons:
                merged_lon.append(np.mean(lons))
                merged_lat.append(np.mean(lats))
            pbar.update(1)

    return merged_lon, merged_lat


def filter_trajectories_by_length(trajectories, top_percentage=0.3):
    if not trajectories:
        return []

    trajectory_lengths = [len(traj) for traj in trajectories]
    threshold_index = max(1, int(len(trajectories) * top_percentage))
    sorted_indices = np.argsort(trajectory_lengths)[::-1]
    selected_indices = sorted_indices[:threshold_index]
    filtered_trajectories = [trajectories[i] for i in selected_indices]

    return filtered_trajectories


def cluster_and_merge_trajectories(cleaned_trajs, eps, min_samples, top_percentage):
    newtras = []
    with tqdm(total=len(cleaned_trajs), desc="提取轨迹数据", unit="条") as pbar:
        for tra in cleaned_trajs:
            tra_data = tra[1:]
            if len(tra_data) > 1:
                newtras.append(tra_data)
            pbar.update(1)

    filtered_trajectories = filter_trajectories_by_length(newtras, top_percentage)
    if not filtered_trajectories:
        raise ValueError("筛选后无有效轨迹")

    distance_matrix = compute_distance_matrix(filtered_trajectories)

    cluster_labels = ddbscan_optimized(filtered_trajectories, eps, min_samples, distance_matrix)

    # 统计聚类结果
    unique_labels = [label for label in np.unique(cluster_labels) if label != -1]

    # 融合每个簇的轨迹
    merged_routes = {}
    with tqdm(total=len(unique_labels), desc="融合轨迹簇", unit="簇") as pbar:
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_trajs = [filtered_trajectories[i] for i in cluster_indices]

            normalized_trajs = normalize_trajectories_in_cluster(cluster_trajs)
            if not normalized_trajs:
                pbar.update(1)
                continue

            merged_lon, merged_lat = merge_trajectories_by_average(normalized_trajs)
            if merged_lon:
                merged_routes[label] = [[float(lon), float(lat)] for lon, lat in zip(merged_lon, merged_lat)]
            pbar.update(1)

    return merged_routes


# ===================== 阶段3：轨迹插值处理函数=====================
def calculate_bearing(lat1, lon1, lat2, lon2):
    """手动计算方位角"""
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlon = lon2_rad - lon1_rad
    y = math.sin(dlon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    return (bearing_deg + 360) % 360


def interpolate_angles(angle_start, angle_end, num_steps):
    """环形角度的线性插值"""
    diff = angle_end - angle_start
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    angles = angle_start + np.linspace(0, diff, num_steps)
    return angles % 360


def get_nearest_point_on_route(original_point, route_points, prev_route_idx=-1, search_range=5):
    """找到原始点在习惯航路上的最近点"""
    min_dist = float('inf')
    nearest_point = None
    nearest_idx = -1

    if 0 <= prev_route_idx < len(route_points):
        start_idx = max(0, prev_route_idx - search_range)
        end_idx = min(len(route_points), prev_route_idx + search_range + 1)
        search_points = enumerate(route_points[start_idx:end_idx], start=start_idx)
    else:
        search_points = enumerate(route_points)

    for idx, rp in search_points:
        dist = geo_dis(original_point, rp)
        if dist < min_dist:
            min_dist = dist
            nearest_point = rp
            nearest_idx = idx

    if nearest_idx == -1:
        for idx, rp in enumerate(route_points):
            dist = geo_dis(original_point, rp)
            if dist < min_dist:
                min_dist = dist
                nearest_point = rp
                nearest_idx = idx

    if nearest_point is None:
        nearest_point = original_point
        nearest_idx = 0

    return nearest_point, nearest_idx, min_dist


def get_route_segment_between_points(route_points, p1, p2):
    """找到习惯航路上两个点之间的有序段"""
    idx1 = -1
    idx2 = -1
    for i, rp in enumerate(route_points):
        if abs(rp[0] - p1[0]) < 1e-6 and abs(rp[1] - p1[1]) < 1e-6:
            idx1 = i
        if abs(rp[0] - p2[0]) < 1e-6 and abs(rp[1] - p2[1]) < 1e-6:
            idx2 = i

    if idx1 == -1 or idx2 == -1:
        return [p1, p2]
    if idx1 <= idx2:
        return route_points[idx1:idx2 + 1]
    else:
        return route_points[idx2:idx1 + 1][::-1]


def get_segment_bearing(segment):
    """计算航路段的方位角"""
    if len(segment) < 2:
        return 0.0
    start_lon, start_lat = segment[0]
    end_lon, end_lat = segment[-1]
    return calculate_bearing(start_lat, start_lon, end_lat, end_lon)


def interpolate_route_segment(route_segment, num_points):
    if len(route_segment) < 2 or num_points <= 0:
        return route_segment * num_points if num_points > 0 else route_segment

    total_dist = 0
    segment_dists = []
    for i in range(len(route_segment) - 1):
        d = geo_dis(route_segment[i], route_segment[i + 1])
        segment_dists.append(d)
        total_dist += d

    if total_dist < 1e-6:
        return [route_segment[0]] * num_points

    target_dists = np.linspace(0, total_dist, num_points)
    interp_points = []
    current_dist = 0
    seg_idx = 0

    for target_dist in target_dists:
        while seg_idx < len(segment_dists) and current_dist + segment_dists[seg_idx] < target_dist:
            current_dist += segment_dists[seg_idx]
            seg_idx += 1

        if seg_idx >= len(segment_dists):
            interp_points.append(route_segment[-1])
            continue

        seg_remaining = target_dist - current_dist
        seg_total = segment_dists[seg_idx]
        ratio = seg_remaining / seg_total

        lon = route_segment[seg_idx][0] + ratio * (route_segment[seg_idx + 1][0] - route_segment[seg_idx][0])
        lat = route_segment[seg_idx][1] + ratio * (route_segment[seg_idx + 1][1] - route_segment[seg_idx][1])
        interp_points.append([lon, lat])

    return interp_points


def linear_interpolate_trajectory_points(curr_point, next_point, num_points):
    if num_points <= 0:
        return []

    curr_lon, curr_lat, curr_speed, curr_course, curr_time = curr_point
    next_lon, next_lat, next_speed, next_course, next_time = next_point

    lons = np.linspace(curr_lon, next_lon, num_points)
    lats = np.linspace(curr_lat, next_lat, num_points)
    speeds = np.linspace(curr_speed if curr_speed is not None else 0.0,
                         next_speed if next_speed is not None else 0.0,
                         num_points)
    courses = interpolate_angles(curr_course if curr_course is not None else 0.0,
                                 next_course if next_course is not None else 0.0,
                                 num_points)
    times = np.linspace(curr_time, next_time, num_points)

    interp_points = []
    for i in range(num_points):
        interp_points.append([
            round(lons[i], 6),
            round(lats[i], 6),
            round(speeds[i], 1),
            round(courses[i], 1),
            int(times[i])
        ])

    return interp_points


def calculate_offset_angle(original_point, nearest_route_point):
    orig_lon, orig_lat = original_point
    route_lon, route_lat = nearest_route_point
    return calculate_bearing(route_lat, route_lon, orig_lat, orig_lon)


def offset_point_by_distance(lon, lat, angle, distance):
    """从指定经纬度点沿指定角度偏移指定距离"""
    distance_m = distance * 1852
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    angle_rad = math.radians(angle)
    R = 6378137.0

    new_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(distance_m / R) +
        math.cos(lat_rad) * math.sin(distance_m / R) * math.cos(angle_rad)
    )
    new_lon_rad = lon_rad + math.atan2(
        math.sin(angle_rad) * math.sin(distance_m / R) * math.cos(lat_rad),
        math.cos(distance_m / R) - math.sin(lat_rad) * math.sin(new_lat_rad)
    )

    new_lat = math.degrees(new_lat_rad)
    new_lon = math.degrees(new_lon_rad)
    new_lon = (new_lon + 180) % 360 - 180

    return [new_lon, new_lat]


def is_trajectory_in_extent(trajectory, extent):
    """判断轨迹是否在目标区域内"""
    min_lon, max_lon, min_lat, max_lat = extent
    for point in trajectory:
        lon = point[0]
        lat = point[1]
        if (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat):
            return True
    return False


def count_far_points(raw_trajectory, route_points, max_dist_km=1.0):
    """统计轨迹中到习惯航路距离超过指定公里数的点数"""
    max_dist_nm = max_dist_km / 1.852
    far_point_count = 0

    for point in raw_trajectory:
        lon, lat = point[0], point[1]
        min_dist = min([geo_dis([lon, lat], rp) for rp in route_points])
        if min_dist > max_dist_nm:
            far_point_count += 1

    need_filter = (far_point_count > 0) and (far_point_count <= MAX_FAR_POINTS)
    use_linear_interp = far_point_count > MAX_FAR_POINTS
    return need_filter, use_linear_interp, far_point_count


def check_interpolated_trajectory(new_trajectory, route_points, max_dist_km=1.0):
    """检查插值后的轨迹是否有任何点离习惯航路超过指定公里数"""
    max_dist_nm = max_dist_km / 1.852
    for point in new_trajectory[1:]:
        lon, lat = point[0], point[1]
        min_dist = min([geo_dis([lon, lat], rp) for rp in route_points])
        if min_dist > max_dist_nm:
            return True
    return False


def match_best_cluster(raw_trajectory, merged_routes):
    if not merged_routes or len(raw_trajectory) < 2:
        return None

    raw_points = [[p[0], p[1]] for p in raw_trajectory]
    key_points = [
        raw_points[0],
        raw_points[-1],
        raw_points[len(raw_points) // 2] if len(raw_points) > 2 else raw_points[0]
    ]

    min_avg_dist = float('inf')
    best_route_points = None

    for cluster_id, cluster_points in merged_routes.items():
        total_dist = 0.0
        valid_key_points = 0

        for key_p in key_points:
            dists = [geo_dis(key_p, cp) for cp in cluster_points]
            if dists:
                total_dist += min(dists)
                valid_key_points += 1

        if valid_key_points > 0:
            avg_dist = total_dist / valid_key_points
            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                best_route_points = cluster_points

    if best_route_points is None and len(merged_routes) > 0:
        best_route_points = list(merged_routes.values())[0]

    return best_route_points


def load_mmsi_static_attr(csv_path):
    df = pd.read_csv(
        csv_path,
        usecols=['MMSI', 'IMO', 'Length', 'Width'],
        encoding='ISO-8859-1'
    )
    df = df.fillna(np.nan)

    mmsi_attr_map = {}
    for mmsi, group in df.groupby('MMSI'):
        mmsi_int = int(mmsi) if pd.notna(mmsi) else mmsi
        imo = group['IMO'].dropna().iloc[0] if not group['IMO'].dropna().empty else np.nan
        length = group['Length'].dropna().iloc[0] if not group['Length'].dropna().empty else np.nan
        width = group['Width'].dropna().iloc[0] if not group['Width'].dropna().empty else np.nan

        mmsi_attr_map[mmsi_int] = {
            'IMO': imo,
            'Length': length,
            'Width': width
        }

    return mmsi_attr_map


def init_output_files(output_json_path, output_csv_path, csv_headers):
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(csv_headers)

    # 初始化JSON
    if not os.path.exists(output_json_path):
        with open(output_json_path, 'w', encoding='utf-8') as f:
            f.write('[')
    else:
        if os.path.getsize(output_json_path) == 0:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                f.write('[')


def append_trajectory_to_files(trajectory, mmsi_static_attr, output_json_path, output_csv_path, is_first=False):
    """将单条轨迹追加保存到JSON和CSV文件"""
    with file_lock:
        # 追加到JSON
        with open(output_json_path, 'a', encoding='utf-8') as f:
            if not is_first:
                f.write(',')
            json_str = json.dumps(trajectory, ensure_ascii=False)
            f.write(json_str)

        # 追加到CSV
        mmsi = trajectory[0][0]
        ship_type_en = trajectory[0][1]
        mmsi_key = int(mmsi) if isinstance(mmsi, (str, float)) else mmsi
        static_attr = mmsi_static_attr.get(mmsi_key, {'IMO': np.nan, 'Length': np.nan, 'Width': np.nan})

        csv_rows = []
        for point in trajectory[1:]:
            csv_rows.append([
                mmsi, ship_type_en, static_attr['IMO'], static_attr['Length'], static_attr['Width'],
                point[0], point[1], point[2], point[3], point[4]
            ])

        with open(output_csv_path, 'a', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(csv_rows)


def complete_json_file(output_json_path):
    """补全JSON文件"""
    with file_lock:
        with open(output_json_path, 'a', encoding='utf-8') as f:
            f.write(']')


def process_single_interp_trajectory(args):
    tra, merged_routes, extent, max_dist_km, post_interp_max_dist_km, max_far_points = args

    try:
        current_mmsi = tra[0][0]
        if not is_trajectory_in_extent(tra[1:], extent):
            return None

        best_route_points = match_best_cluster(tra[1:], merged_routes)
        if not best_route_points:
            return None

        need_filter, use_linear_interp, far_point_count = count_far_points(tra[1:], best_route_points, max_dist_km)
        if need_filter:
            return None

        new_trajectory = []
        new_trajectory.append(tra[0])
        raw_points = tra[1:]
        prev_route_idx = -1

        for idx in range(len(raw_points) - 1):
            curr_raw_point = raw_points[idx]
            next_raw_point = raw_points[idx + 1]
            curr_lon, curr_lat, curr_speed, curr_course, curr_time = curr_raw_point
            next_lon, next_lat, next_speed, next_course, next_time = next_raw_point
            time_diff = next_time - curr_time

            if time_diff <= 600:
                new_trajectory.append(curr_raw_point)
                if not use_linear_interp:
                    _, prev_route_idx, _ = get_nearest_point_on_route(
                        [curr_lon, curr_lat], best_route_points, prev_route_idx
                    )
                continue

            if use_linear_interp:
                num_interp_points = max(1, int(time_diff / 600))
                linear_interp_points = linear_interpolate_trajectory_points(
                    curr_raw_point, next_raw_point, num_interp_points
                )
                new_trajectory.extend(linear_interp_points)
            else:
                curr_route_point, curr_route_idx, curr_dist = get_nearest_point_on_route(
                    [curr_lon, curr_lat], best_route_points, prev_route_idx
                )
                next_route_point, next_route_idx, next_dist = get_nearest_point_on_route(
                    [next_lon, next_lat], best_route_points, curr_route_idx
                )
                prev_route_idx = next_route_idx

                route_segment = get_route_segment_between_points(
                    best_route_points, curr_route_point, next_route_point
                )
                orig_bearing = calculate_bearing(curr_lat, curr_lon, next_lat, next_lon)
                route_bearing = get_segment_bearing(route_segment)
                bearing_diff = abs((orig_bearing - route_bearing + 180) % 360 - 180)

                if bearing_diff > 90:
                    route_segment = route_segment[::-1]

                num_interp_points = max(1, int(time_diff / 600))
                route_interp_points = interpolate_route_segment(route_segment, num_interp_points)

                curr_offset_angle = calculate_offset_angle([curr_lon, curr_lat], curr_route_point)
                next_offset_angle = calculate_offset_angle([next_lon, next_lat], next_route_point)

                dists = np.linspace(curr_dist, next_dist, num_interp_points)
                angles = interpolate_angles(curr_offset_angle, next_offset_angle, num_interp_points)
                times = np.linspace(curr_time, next_time, num_interp_points)

                for j in range(num_interp_points):
                    base_lon, base_lat = route_interp_points[j] if j < len(route_interp_points) else curr_route_point
                    offset_dist = dists[j] if j < len(dists) else curr_dist
                    offset_angle = angles[j] if j < len(angles) else curr_offset_angle

                    interp_lon, interp_lat = offset_point_by_distance(
                        base_lon, base_lat, offset_angle, offset_dist
                    )

                    new_trajectory.append([
                        round(interp_lon, 6),
                        round(interp_lat, 6),
                        0.0 if curr_speed is None else curr_speed,
                        0.0 if curr_course is None else curr_course,
                        int(times[j] if j < len(times) else curr_time)
                    ])

        last_point = tra[-1]
        new_trajectory.append([
            last_point[0], last_point[1],
            last_point[2], last_point[3],
            last_point[4]
        ])

        if len(new_trajectory) < 2:
            return None

        if not use_linear_interp:
            has_far_point = check_interpolated_trajectory(new_trajectory, best_route_points, post_interp_max_dist_km)
            if has_far_point:
                return None

        return new_trajectory

    except Exception:
        return None


def interpolate_trajectories(cleaned_trajs, merged_routes, csv_path, output_json, output_csv, extent):
    mmsi_static_attr = load_mmsi_static_attr(csv_path)

    # 定义CSV表头
    csv_headers = ['MMSI', 'ShipTypeEN', 'IMO', 'Length', 'Width', 'Lon_d', 'Lat_d', 'Speed', 'Course', 'UnixTime']

    # 初始化输出文件
    init_output_files(output_json, output_csv, csv_headers)

    # 准备参数
    total_trajectories = len(cleaned_trajs)
    args_list = [(tra, merged_routes, extent, MAX_DIST_KM, POST_INTERP_MAX_DIST_KM, MAX_FAR_POINTS)
                 for tra in cleaned_trajs]

    # 统计变量
    successful_count = 0
    first_successful = True

    # 并行处理
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        with tqdm(total=total_trajectories, desc="插值处理轨迹", unit="条") as pbar:
            for result in pool.imap_unordered(process_single_interp_trajectory, args_list):
                pbar.update(1)

                if result is not None:
                    append_trajectory_to_files(
                        result, mmsi_static_attr,
                        output_json, output_csv,
                        is_first=first_successful
                    )
                    if first_successful:
                        first_successful = False
                    successful_count += 1

    # 补全JSON文件
    complete_json_file(output_json)
    print(f"最终结果文件：")
    print(f"JSON: {output_json}")
    print(f"CSV: {output_csv}")


# ===================== 主函数 =====================
def main():
    """主流程：清洗→聚类→插值"""
    print("=" * 80)
    print("开始轨迹处理流程（清洗→聚类→插值）")
    print("=" * 80)

    # 配置参数
    shiptype = ['Cargo ship', 'Tanker', 'Passenger ship', 'Tug', 'Fishing']

    # 阶段1：清洗轨迹数据
    print("\n【阶段1：清洗轨迹数据】")
    cleaned_trajs = clean_trajectory_data(RAW_CSV_PATH, TARGET_EXTENT, shiptype)

    # 阶段2：聚类融合轨迹
    print("\n【阶段2：聚类融合轨迹】")
    merged_routes = cluster_and_merge_trajectories(
        cleaned_trajs,
        eps=CLUSTER_EPS,
        min_samples=CLUSTER_MIN_SAMPLES,
        top_percentage=CLUSTER_TOP_PERCENTAGE
    )

    # 阶段3：插值处理轨迹
    print("\n【阶段3：插值处理轨迹】")
    interpolate_trajectories(
        cleaned_trajs,
        merged_routes,
        RAW_CSV_PATH,
        FINAL_OUTPUT_JSON,
        FINAL_OUTPUT_CSV,
        TARGET_EXTENT
    )

    print("\n" + "=" * 80)
    print("所有处理完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()