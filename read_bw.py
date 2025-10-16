"""
clean_segments/ 폴더의 clean 세그먼트에 BW 노이즈를 SNR 타깃으로 섞어 저장하고 메타 파일 생성.

- input : clean_segments/segments.csv
- output : clean_segemnts_bw/segments_bw.csv
- SNR level : +24, +18, +12, +6, 0, -6 (dB)
- 샘플링률 : 360 Hz
"""

import argparse
import csv
import hashlib
import math
from pathlib import Path

import numpy as np 
import wfdb
from scipy.signal import resample_poly
from tqdm import tqdm

SNR_LEVELS = [24, 18, 12, 6, 0, -6]
TARGET_FS = 360
WIN_SECONDS = 10
WIN_SAMPLES = TARGET_FS * WIN_SECONDS
TOL_DB = 0.2 # 허용 오차

def read_bw_siganl(bw_dir: Path):
    rec = wfdb.rdrecord(str(bw_dir / "bw"))
    sig = rec.p_signal.squeeze().astype(np.float64)
    fs = int(rec.fs)
    return sig, fs

def to_target_fs(x: np.ndarray, fs_src: int, fs_tgt: int) -> np.ndarray:
    if fs_src == fs_tgt:
        return x
    up, down = fs_tgt, fs_src
    g = math.gcd(up, down)
    up //= g; down //= g
    return resample_poly(x, up, down)

def get_bw_slice(bw_long: np.ndarray, L: int, seed: int) -> np.ndarray:
    """ 
    길이 L만큼 bw를 슬라이싱해서 반환, 부족하면 tiling 후 랜덤 오프셋
    """
    rng = np.random.default_rng(seed)
    if len(bw_long) < L:
        reps = int(np.ceil(L / len(bw_long)))
        bw_long = np.tile(bw_long, reps)
    start = rng.integers(0, len(bw_long) - L)
    return bw_long[start:start+L]

def mix_with_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float, tol_db=TOL_DB):
    if clean.ndim == 1:
        clean = clean[None, :]
    C, L = clean.shape
    assert noise.shape[0] == L

    p_sig = np.mean(clean**2)
    p_noise_target = p_sig / (10**(snr_db / 10))
    p_noise_src = np.mean(noise**2)

    alpha = np.sqrt(p_noise_target / (p_noise_src + 1e-12))
    if noise.ndim == 2 and noise.shape[0] == clean.shape[-1]:
        # (3600, 2) → (2, 3600)로 전치
        noise = noise.T
    elif noise.ndim == 1:
        noise = noise[None, :]

    mixed = clean + alpha * noise

    resid = mixed - clean
    p_resid = np.mean(resid**2)
    snr_meas = 10 * np.log10(p_sig / (p_resid + 1e-12))

    if abs(snr_meas - snr_db) > tol_db:
        alpha *= 10 ** ((snr_meas - snr_db) / 20)
        mixed = clean + alpha * noise[None, :]
        resid = mixed - clean
        p_resid = np.mean(resid**2)
        snr_meas = 10 * np.log10(p_sig / (p_resid + 1e-12))

    return mixed.squeeze(), float(alpha), float(snr_meas)

def deterministic_seed(*tokens) -> int:
    s = "_".join(map(str, tokens))
    h = hashlib.md5(s.encode()).hexdigest()[:8]
    return int(h, 16) % (2**31 - 1)

def main(root: Path, bw_dir: Path, out_root: Path):
    bw_raw, fs_bw = read_bw_siganl(bw_dir)
    bw_rs = to_target_fs(bw_raw, fs_bw, TARGET_FS)

    meta_in = root / "segments.csv"
    assert meta_in.exists(), f"Not found: {meta_in}"
    rows = []
    with open(meta_in, newline="", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    
    out_root.mkdir(parents=True, exist_ok=True)
    for sp in ["train", "val", "test"]:
        (out_root / sp).mkdir(parents=True, exist_ok=True)
    
    meta_out_path = out_root / "segments_bw_csv"
    fieldnames = ["id","record","start_s","duration_s","sampling_rate",
                  "noise","snr_db","scale_alpha","seed","split","path"]
    fout = open(meta_out_path, "w", newline="", encoding='utf-8')
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    print("Mixing!")

    for r in tqdm(rows, total=len(rows)):
        split = r["split"]
        in_path = (root.parent / r["path"]).resolve()
        clean = np.load(in_path)  # (L,) 또는 (C, L)
        # 채널/길이 확인 및 3600 고정
        if clean.ndim == 1:
            clean = clean[None, :]
        if clean.shape[-1] != WIN_SAMPLES:
            raise ValueError(f"Unexpected length {clean.shape[-1]} for {in_path.name}")

        # (옵션) 채널이 2라면 그대로 둠 (Lead I, II)
        C = clean.shape[0]

        for snr in SNR_LEVELS:
            # 재현 가능한 seed (파일명+snr 기반)
            seed = deterministic_seed(r["id"], snr)
            bw_slice = get_bw_slice(bw_rs, WIN_SAMPLES, seed)

            mixed, alpha, snr_meas = mix_with_snr(clean, bw_slice, snr_db=float(snr), tol_db=TOL_DB)

            # 저장 파일명: <id>_bw_<SNR>dB.npy
            stem = Path(r["id"]).stem if r["id"].endswith(".npy") else r["id"]
            out_name = f"{stem}_bw_{snr}dB.npy"
            out_path = out_root / split / out_name
            np.save(out_path, mixed.astype(np.float64))

            # 메타 기록
            writer.writerow({
                "id": out_name.replace(".npy",""),
                "record": r["record"],
                "start_s": r["start_s"],
                "duration_s": r["duration_s"],
                "sampling_rate": TARGET_FS,
                "noise": "bw",
                "snr_db": snr,
                "scale_alpha": alpha,
                "seed": seed,
                "split": split,
                "path": str(out_path).replace("\\","/"),
            })

    fout.close()
    print(f"[DONE] Saved mixed segments to: {out_root}")
    print(f"[DONE] Metadata: {meta_out_path}")
    print("[TIP] 샘플 몇 개를 열어 실제 SNR을 재검증해보세요.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="clean_segments 폴더 경로 (segments.csv 포함)")
    ap.add_argument("--bw_dir", required=True, help="NSTDB BW 파일이 있는 경로 (bw.dat/bw.hea)")
    ap.add_argument("--out", default=None, help="출력 폴더 (기본: clean_segments_bw)")
    args = ap.parse_args()

    root = Path(args.root)
    bw_dir = Path(args.bw_dir)
    out_root = Path(args.out) if args.out else (root.parent / "clean_segments_bw")

    main(root, bw_dir, out_root)