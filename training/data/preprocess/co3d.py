from pathlib import Path
import json
import gzip
import numpy as np
import torch
from tqdm import tqdm
import os.path as osp

root = Path("/mimer/NOBACKUP/groups/3d-dl/co3d_full")


for category_dir in tqdm(root.iterdir()):
    frame_file = osp.join(category_dir, "frame_annotations.jgz")
    sequence_file = osp.join(category_dir, "sequence_annotations.jgz")

    set_list = json.load(open(osp.join(category_dir, "set_lists.json"), "r"))
    
    train_sequences = set()
    for split in ["train_known", "train_unseen"]:
        if split in set_list:
            for entry in set_list[split]:
                sequence_id = entry[0]  # first element is the sequence ID
                train_sequences.add(sequence_id)

    # Convert to a sorted list if you want
    train_sequences = sorted(train_sequences)

    print('Set list: ', train_sequences)
    break

    with gzip.open(frame_file, "r") as fin:
        frame_data = json.loads(fin.read())
    with gzip.open(sequence_file, "r") as fin:
        sequence_data = json.loads(fin.read())

    frame_data_processed = {}
    for f_data in frame_data:
        sequence_name = f_data["sequence_name"]
        frame_data_processed.setdefault(sequence_name, {})[f_data["frame_number"]] = f_data

    print('frame data: ', frame_data_processed['20_716_1426'])

    # intrinsics = read_scannet_intrinsic(scene_dir / "intrinsic/intrinsic_color.txt")
    
    break



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

# root = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt"

# with gzip.open(root+"/annotations/scannet.jgz", "wt", encoding="utf-8") as f:
#     json.dump(out, f, ensure_ascii=False, indent=4)

# print(f"Processed {len(out)} scenes with a total of {sum(len(v) for v in out.values())} images.")

