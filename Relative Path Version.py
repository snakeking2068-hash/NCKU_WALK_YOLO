# -*- coding: utf-8 -*-
"""
YOLO Only (No LLM) - Relative Path Version
- 畫框（可切換）
- 危險分級：用 YOLO 偵測物件數量做規則分級（可重現）
- 只輸出 YOLO 物件統計文字（det_summary）
- 不覆蓋任何既有成果：每次輸出都會建立一個新的 run_YYYYMMDD_HHMMSS 資料夾
- 生成新的 CSV（基於原 CSV 疊加新欄位，不覆蓋原檔）
- 追加輸出照片「相對路徑」欄位（方便 QGIS portable）

【相對路徑核心】
以「此 .py 所在資料夾」作為 REPORT_ROOT（也就是 Report 資料夾）。
同學只要整包 Report 資料夾複製到自己的電腦，路徑不用改。
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import pandas as pd
import os
from datetime import datetime


# ================== 0) 你可以調的開關 ==================
DRAW_BOXES = True  # True: YOLO 畫框；False: 不畫框


# ================== 1) 路徑設定（全部相對於 Report） ==================
# Report 根目錄：這支 .py 所在資料夾
REPORT_ROOT = Path(__file__).resolve().parent

# 你資料夾內的相對位置（同學電腦也會一樣）
IMAGES_DIR = REPORT_ROOT / "01_original_information"/"images_large"
SRC_CSV = REPORT_ROOT / "01_original_information" / "routes_large_points.csv"
GOAL_ROOT = REPORT_ROOT / "goal_film"

MODEL_CANDIDATES = [
    REPORT_ROOT / "yolo11m.pt",
    REPORT_ROOT / "models_medium" / "yolo11m.pt",
    REPORT_ROOT / "models" / "yolo11m.pt",
]

if not IMAGES_DIR.exists():
    raise FileNotFoundError(f"找不到影像資料夾：{IMAGES_DIR}")

if not SRC_CSV.exists():
    raise FileNotFoundError(f"找不到原始 CSV：{SRC_CSV}")

GOAL_ROOT.mkdir(parents=True, exist_ok=True)

# ✅ 每次跑都建立新資料夾，避免覆蓋
run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
RUN_DIR = GOAL_ROOT / run_tag
OUT_IMG_DIR = RUN_DIR / "images_out"
OUT_TAB_DIR = RUN_DIR / "tables_out"

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_TAB_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_TAB_DIR / "routes_large_points_with_yolo.csv"

# 找到模型權重
MODEL_PATH = None
for p in MODEL_CANDIDATES:
    if p.exists():
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    raise FileNotFoundError(
        "找不到 YOLO 權重檔 yolo11m.pt。\n"
        "請確認以下任一相對位置存在：\n- " + "\n- ".join(str(x) for x in MODEL_CANDIDATES)
    )


# ================== 2) YOLO ==================
model = YOLO(str(MODEL_PATH))


# ================== 3) 規則分級（0/1/2） ==================
def rule_risk_level(counts: dict) -> int:
    car = counts.get("car", 0)
    bike = counts.get("bicycle", 0)
    moto = counts.get("motorcycle", 0)
    bm = bike + moto

    if car >= 5:
        return 2
    if 2 <= car <= 4 and bm > 5:
        return 2
    if car <= 1 and bm <= 5:
        return 0
    return 1


# ================== 4) 文字顏色：Level 0 綠色；Level 1/2 紅色 ==================
def _level_color(level: int):
    # OpenCV color is BGR
    if level == 0:
        return (0, 255, 0)   # green
    return (0, 0, 255)       # red


def _put_wrapped_text(img, text, x, y, max_chars=52, font_scale=0.8, thickness=2, line_gap=32, color=(0, 0, 255)):
    lines = []
    t = (text or "").strip()
    while len(t) > max_chars:
        lines.append(t[:max_chars])
        t = t[max_chars:]
    if t:
        lines.append(t)

    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x, y + i * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )


# ================== 5) 主流程：跑 YOLO + 存圖 + 組 det_rows ==================
results = model(source=str(IMAGES_DIR), stream=True)

det_rows = []

for r in results:
    r_path = Path(getattr(r, "path", ""))
    img_filename = r_path.name if r_path.name else "unknown.jpg"

    # 要不要畫框
    if DRAW_BOXES:
        img = r.plot()
    else:
        img = r.orig_img.copy()

    # 統計 YOLO 物件
    counts = {}
    if r.boxes is not None and r.boxes.cls is not None and len(r.boxes.cls) > 0:
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        for cid in cls_ids:
            label = model.names[int(cid)]
            counts[label] = counts.get(label, 0) + 1

    det_summary = ", ".join([f"{v} {k}" for k, v in counts.items()]) if counts else ""

    # 用規則算 level
    level = rule_risk_level(counts)

    # 疊加文字（只放 YOLO summary，沒有 warning_text）
    h, w, _ = img.shape
    y0 = max(40, h - 130)
    color = _level_color(level)

    cv2.putText(
        img,
        f"Emergency level: {level}",
        (30, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA
    )

    if det_summary:
        _put_wrapped_text(img, det_summary, 30, y0 + 40, color=color)

    # 存圖（檔名含 level）
    out_img_name = f"{Path(img_filename).stem}_lvl{level}.jpg"
    out_path = OUT_IMG_DIR / out_img_name
    cv2.imwrite(str(out_path), img)

    # ✅ 相對路徑（相對於 tables_out，方便 QGIS portable）
    det_img_relpath = os.path.relpath(out_path, OUT_TAB_DIR).replace("\\", "/")

    det_rows.append({
        "img_file": img_filename,
        "det_img_file": str(out_path),          # 絕對路徑（方便 debug）
        "det_img_relpath": det_img_relpath,     # 相對路徑（QGIS portable）
        "det_summary": det_summary,
        "risk_level": level
    })


# ================== 6) 合併 CSV（不覆蓋原檔） ==================
df_src = pd.read_csv(SRC_CSV)
df_det = pd.DataFrame(det_rows)

# 自動找影像欄位 merge
candidate_cols = []
for c in df_src.columns:
    cl = c.lower()
    if any(k in cl for k in ["img", "image", "file", "filename", "path"]):
        candidate_cols.append(c)

merge_col = None
for c in candidate_cols:
    if c.lower() == "img_file":
        merge_col = c
        break
if merge_col is None and candidate_cols:
    merge_col = candidate_cols[0]

if merge_col is not None:
    df_src["_img_key_"] = df_src[merge_col].astype(str).apply(lambda x: Path(x).name)
    df_det["_img_key_"] = df_det["img_file"].astype(str)

    df_out = df_src.merge(
        df_det.drop(columns=["img_file"]),
        left_on="_img_key_",
        right_on="_img_key_",
        how="left"
    ).drop(columns=["_img_key_"])
else:
    # 找不到可對齊欄位，退而求其次用順序塞
    df_out = df_src.copy()
    n = min(len(df_out), len(df_det))
    df_out.loc[:n-1, "det_img_file"] = df_det.loc[:n-1, "det_img_file"].values
    df_out.loc[:n-1, "det_img_relpath"] = df_det.loc[:n-1, "det_img_relpath"].values
    df_out.loc[:n-1, "det_summary"] = df_det.loc[:n-1, "det_summary"].values
    df_out.loc[:n-1, "risk_level"] = df_det.loc[:n-1, "risk_level"].values

df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("完成：YOLO Only（無 LLM）+ 規則分級 + 相對路徑 + 不覆蓋輸出")
print("Report 根目錄：", REPORT_ROOT)
print("本次輸出資料夾：", RUN_DIR)
print("🖼輸出影像資料夾：", OUT_IMG_DIR)
print("輸出 CSV：", OUT_CSV)
print("YOLO model：", MODEL_PATH)
print("來源影像資料夾：", IMAGES_DIR)
print("DRAW_BOXES =", DRAW_BOXES)
