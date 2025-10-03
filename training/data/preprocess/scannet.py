from pathlib import Path
import json
import gzip
import numpy as np
import torch
from tqdm import tqdm

def read_scannet_pose(path):
    """ Read ScanNet's Camera2World pose and transform it to World2Camera.

    Returns:
        pose_w2c (np.ndarray): (4, 4)
    """
    cam2world = np.loadtxt(path, delimiter=' ')
    world2cam = np.linalg.inv(cam2world)
    return world2cam


def read_scannet_intrinsic(path):
    """ Read ScanNet's intrinsic matrix and return the 3x3 matrix.
    """
    intrinsic = np.loadtxt(path, delimiter=' ')
    return torch.tensor(intrinsic[:-1, :-1], dtype = torch.float)

# Root folder where everything starts
root = Path("/mimer/NOBACKUP/groups/3d-dl/scannet/scans/scans_train")

out = {}

chunk_size = 24


for scene_dir in tqdm(root.iterdir()):

    intrinsics = read_scannet_intrinsic(scene_dir / "intrinsic/intrinsic_color.txt")
    
    frames = sorted([p.name for p in (scene_dir / "color").iterdir() if p.suffix == ".jpg"])

    # Maybe resized undistorted images are too high resolution?
    num_frames = len(frames)

    # Since the images are taken in a sequence we will just chunk up the sequences

    sequences = []
    # Calculate how many full chunks we can take, stopping before the last chunk
    num_full_chunks = (num_frames - 1) // chunk_size  # leave room for overflow in last chunk

    for i in range(num_full_chunks - 1):
        sequences.append(frames[i * chunk_size: (i + 1) * chunk_size])

    # Last chunk gets the rest of the frames
    sequences.append(frames[(num_full_chunks - 1) * chunk_size:])

    for i, seq in enumerate(sequences):
        sequence_data = []
        for frame in seq:
            pose_path = scene_dir / "pose" / (frame.replace(".jpg", ".txt"))
            pose_w2c = read_scannet_pose(pose_path)
            frame_data = {
                "filepath": f"{scene_dir.name}/color/{frame}",
                "extri": pose_w2c[:3].tolist(),
                "intri": intrinsics.tolist(),
                "depthpath": f"{scene_dir.name}/depth/{frame.replace('.jpg', '.png')}",
            }
            # Sanity check
            assert len(pose_w2c) == 4 and len(pose_w2c[0]) == 4
            assert len(intrinsics) == 3 and len(intrinsics[0]) == 3

            sequence_data.append(frame_data)

        out[scene_dir.name+"_"+str(i)] = sequence_data

    print(f"  Created {len(sequences)} sequences for {scene_dir.name}")

root = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt"

with gzip.open(root+"/annotations/scannet.jgz", "wt", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=4)

print(f"Processed {len(out)} scenes with a total of {sum(len(v) for v in out.values())} images.")

