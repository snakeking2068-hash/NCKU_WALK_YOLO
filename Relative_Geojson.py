# -*- coding: utf-8 -*-
"""
YOLO CSV -> risk routes GeoJSON (L2 / L1)  [Relative Path Version]

需求：
- 讀取含有 lng/lat/risk_level 的 CSV（YOLO-only 產生的 routes_large_points_with_yolo.csv）
- 產生兩個線段 GeoJSON：
  1) risk_route_L2.geojson
     - 對每個 risk_level == 2 的點：
       在「半徑 RADIUS_M 公尺內」找最近的「其他 2 個點」連線（不足 2 個就連能找到的）
  2) risk_route_L1.geojson
     - 對每個 risk_level == 1 的點：
       在「半徑 RADIUS_M 公尺內」找最近的「其他 1 個點」連線（不限對方等級）

【相對路徑核心】
- REPORT_ROOT = 本 .py 所在資料夾（Report）
- 預設自動讀取：Report/goal_film/<最新 run_...>/tables_out/routes_large_points_with_yolo.csv
- 輸出到同一個 tables_out（跟 CSV 同資料夾）
"""

from pathlib import Path
import json
import math
import pandas as pd
import argparse
from datetime import datetime


RADIUS_M_DEFAULT = 5.0
DEFAULT_CSV_NAME = "routes_large_points_with_yolo.csv"


# ================== 工具：座標距離（m） ==================
def haversine_m(lon1, lat1, lon2, lat2) -> float:
    """WGS84 經緯度 haversine 距離（公尺）"""
    R = 6371000.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlmb = math.radians(float(lon2) - float(lon1))

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


# ================== 工具：把 pandas/numpy 型別轉 JSON 可序列化 ==================
def to_jsonable(x):
    """把 pandas/numpy 型別轉成 Python 原生型別，避免 json.dump() 爆 int64/float64"""
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
    except Exception:
        pass

    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass

    if isinstance(x, (str, int, float, bool, list, dict)) or x is None:
        return x

    return str(x)


# ================== 自動找欄位（lon/lat/risk） ==================
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {str(col).lower(): col for col in df.columns}
    for c in candidates:
        cc = str(c).lower()
        if cc in lower_map:
            return lower_map[cc]
    return None


def detect_columns(df):
    lon_col = pick_col(df, ["lng", "lon", "longitude", "x"])
    lat_col = pick_col(df, ["lat", "latitude", "y"])
    risk_col = pick_col(df, ["risk_level", "risk", "emergency_level", "level"])

    if lon_col is None or lat_col is None:
        raise ValueError(
            f"找不到經緯度欄位。請確認 CSV 有 lng/lat（或 lon/lat）。\n"
            f"目前欄位：{list(df.columns)}"
        )
    if risk_col is None:
        raise ValueError(
            f"找不到 risk 欄位。請確認 CSV 有 risk_level（或 emergency_level/level）。\n"
            f"目前欄位：{list(df.columns)}"
        )
    return lon_col, lat_col, risk_col


# ================== 建立 GeoJSON Feature ==================
def make_line_feature(df, i, j, dist_m, tag, lon_col, lat_col, risk_col):
    lon1 = float(df.at[i, lon_col])
    lat1 = float(df.at[i, lat_col])
    lon2 = float(df.at[j, lon_col])
    lat2 = float(df.at[j, lat_col])

    props = {
        "type": to_jsonable(tag),
        "from_index": to_jsonable(i),
        "to_index": to_jsonable(j),
        "from_risk": to_jsonable(df.at[i, risk_col]),
        "to_risk": to_jsonable(df.at[j, risk_col]),
        "dist_m": to_jsonable(dist_m),
    }

    for maybe in ["seq", "dept", "origin", "origin_typ", "level", "dist_m", "det_summary"]:
        if maybe in df.columns:
            props["from_" + maybe] = to_jsonable(df.at[i, maybe])
            props["to_" + maybe] = to_jsonable(df.at[j, maybe])

    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]]},
    }


# ================== 核心：找半徑內最近鄰 ==================
def nearest_within_radius(df, idx, lon_col, lat_col, radius_m, k):
    lon0 = float(df.at[idx, lon_col])
    lat0 = float(df.at[idx, lat_col])

    cand = []
    for j in df.index:
        if j == idx:
            continue
        lon1 = df.at[j, lon_col]
        lat1 = df.at[j, lat_col]
        if pd.isna(lon1) or pd.isna(lat1):
            continue
        d = haversine_m(lon0, lat0, float(lon1), float(lat1))
        if d <= radius_m:
            cand.append((j, d))

    cand.sort(key=lambda x: x[1])
    return cand[:k]


# ================== run 目錄：自動抓最新 run_YYYYMMDD_HHMMSS ==================
def find_latest_run_tables_csv(report_root: Path, csv_name: str) -> Path:
    goal_root = report_root / "goal_film"
    if not goal_root.exists():
        raise FileNotFoundError(f"找不到 goal_film：{goal_root}")

    run_dirs = [p for p in goal_root.glob("run_*") if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"goal_film 底下找不到 run_* 資料夾：{goal_root}")

    # 以資料夾名稱時間排序（run_YYYYMMDD_HHMMSS）
    def parse_run_time(p: Path):
        m = p.name.replace("run_", "")
        try:
            return datetime.strptime(m, "%Y%m%d_%H%M%S")
        except Exception:
            return datetime.min

    run_dirs.sort(key=parse_run_time, reverse=True)

    for rd in run_dirs:
        candidate = rd / "tables_out" / csv_name
        if candidate.exists():
            return candidate

    # 如果最新的 run 裡沒有，就把第一個 run 印出來提示（但仍 raise）
    raise FileNotFoundError(
        f"找不到 {csv_name}。\n已掃描：{len(run_dirs)} 個 run_* 資料夾。\n"
        f"例如第一個掃描目標：{run_dirs[0] / 'tables_out' / csv_name}"
    )


def safe_int(x):
    try:
        return int(float(x))
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser(description="Generate risk route GeoJSON from YOLO CSV (relative paths)")
    ap.add_argument("--radius", type=float, default=RADIUS_M_DEFAULT, help="Search radius in meters (default: 5.0)")
    ap.add_argument("--csv", default="", help="(Optional) CSV path. If empty, auto-pick latest run tables_out CSV.")
    ap.add_argument("--csv-name", default=DEFAULT_CSV_NAME, help="CSV filename under each run/tables_out/")
    args = ap.parse_args()

    report_root = Path(__file__).resolve().parent

    # 1) 決定 CSV 來源
    if args.csv.strip():
        csv_path = (report_root / args.csv).resolve() if not Path(args.csv).is_absolute() else Path(args.csv).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"找不到 CSV：{csv_path}")
    else:
        csv_path = find_latest_run_tables_csv(report_root, args.csv_name)

    # 2) 輸出資料夾：跟 CSV 同層（tables_out）
    out_dir = csv_path.parent
    out_l2 = out_dir / "risk_route_L2.geojson"
    out_l1 = out_dir / "risk_route_L1.geojson"

    # 3) 讀 CSV
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    lon_col, lat_col, risk_col = detect_columns(df)

    df = df.dropna(subset=[lon_col, lat_col]).copy()
    df.reset_index(drop=True, inplace=True)
    df[risk_col] = df[risk_col].apply(safe_int)

    features_l2 = []
    features_l1 = []
    radius_m = float(args.radius)

    # L2：risk==2 -> 半徑內最近 2 個
    idx_l2 = df.index[df[risk_col] == 2].tolist()
    for i in idx_l2:
        neigh = nearest_within_radius(df, i, lon_col, lat_col, radius_m, k=2)
        for j, d in neigh:
            features_l2.append(make_line_feature(df, i, j, d, "L2", lon_col, lat_col, risk_col))

    # L1：risk==1 -> 半徑內最近 1 個
    idx_l1 = df.index[df[risk_col] == 1].tolist()
    for i in idx_l1:
        neigh = nearest_within_radius(df, i, lon_col, lat_col, radius_m, k=1)
        for j, d in neigh:
            features_l1.append(make_line_feature(df, i, j, d, "L1", lon_col, lat_col, risk_col))

    geojson_l2 = {"type": "FeatureCollection", "features": features_l2}
    geojson_l1 = {"type": "FeatureCollection", "features": features_l1}

    with open(out_l2, "w", encoding="utf-8") as f:
        json.dump(geojson_l2, f, ensure_ascii=False, indent=2)

    with open(out_l1, "w", encoding="utf-8") as f:
        json.dump(geojson_l1, f, ensure_ascii=False, indent=2)

    print("✅ 完成輸出 GeoJSON（相對路徑版）")
    print("CSV =", csv_path)
    print("OUT =", out_dir)
    print("L2 lines:", len(features_l2), "->", out_l2)
    print("L1 lines:", len(features_l1), "->", out_l1)
    print("使用欄位：", {"lon": lon_col, "lat": lat_col, "risk": risk_col})
    print("半徑閾值：", radius_m, "m")


if __name__ == "__main__":
    main()
