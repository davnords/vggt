from pathlib import Path
import json
import gzip
import numpy as np
import torch
from tqdm import tqdm

# Root folder where everything starts
root = Path("/mimer/NOBACKUP/groups/3d-dl/MVS-Synth/GTAV_540")

out = {}

for scene_dir in tqdm(root.iterdir()):

    if scene_dir.name.startswith('num_images'):
        continue
    
    frames = sorted([p.name for p in (scene_dir / "images").iterdir() if p.suffix == ".png"])
    sequence_data = []
    for frame in frames:
        pose_path = scene_dir / "poses" / (frame.replace(".png", ".json"))
        depth_path = scene_dir / "depths" / (frame.replace(".png", ".exr"))

        with open(pose_path) as f:
            cam = json.load(f)

        extrinsic_4x4 = np.array(cam["extrinsic"], dtype=np.float32)
        R = extrinsic_4x4[:3, :3]
        t = extrinsic_4x4[:3, 3]
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
            t[2] *= -1
        pose_w2c = np.hstack([R, t.reshape(3, 1)])

        K = np.array([
            [cam["f_x"], 0, cam["c_x"]],
            [0, cam["f_y"], cam["c_y"]],
            [0, 0, 1]
        ], dtype=np.float32)

        # pose_w2c = read_scannet_pose(pose_path)
        frame_data = {
            "filepath": f"{scene_dir.name}/images/{frame}",
            "extri": pose_w2c.tolist(),
            "intri": K.tolist(),
            "depthpath": f"{scene_dir.name}/depths/{frame.replace('.png', '.exr')}",
        }
        sequence_data.append(frame_data)

    out[scene_dir.name] = sequence_data

root = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt"

with gzip.open(root+"/annotations/mvssynth.jgz", "wt", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=4)

print(f"Processed {len(out)} scenes with a total of {sum(len(v) for v in out.values())} images.")

