import torch
import clip
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# def clip_encode_video(path, chunks:int = 16, last_chunk:bool = False):
#     # For each 16 frame chunk in the video, extract CLIP features
#     # Each clip feature per frame is of shape (1, 512)
#     import cv2
#     vidcap = cv2.VideoCapture(path)
#     success, image = vidcap.read()
#     frames = []
#     frame_features = []
#     chunk_features = []
#     last_frames = []
#     count = 0
#     num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f'Total number of frames in the video: {num_frames}')
#     remaining_frames = num_frames % chunks
#     last_frames_start = num_frames - chunks
#     while success:
#         frames.append(image)
#         if remaining_frames > 0 and count >= last_frames_start:
#             last_frames.append(image)
#         if len(frames) == chunks:
#             pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
#             inputs = torch.stack([preprocess(f) for f in pil_frames]).to(device)
#             with torch.no_grad():
#                 chunk_features = model.encode_image(inputs).cpu()  # (chunks, 512)
#                 frame_features.append(chunk_features.flatten().unsqueeze(0))  # (1, chunks*512)
#             frames = []
#         success, image = vidcap.read()
#         count += 1
#     if len(last_frames) > 0 and last_chunk:
#         pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in last_frames]
#         inputs = torch.stack([preprocess(f) for f in pil_frames]).to(device)
#         with torch.no_grad():
#             chunk_features = model.encode_image(inputs).cpu()  # (remaining_frames, 512)
#             print(f'last chunk features shape: {chunk_features.shape}')
#             frame_features.append(chunk_features.flatten().unsqueeze(0))  # (1, remaining_frames*512)
#     if len(frame_features) > 0:
#         print(f'Frame features {len(frame_features)}')
#         frame_features = torch.cat(frame_features, dim=0)  # (num_chunks, chunk_size*512)
#     return frame_features

def clip_encode_video(path, batch_size: int = 16):
    """
    Extract per-frame CLIP features from a video.
    Returns a tensor of shape (num_frames, 512) on CPU.
    No padding, no dropped frames — every frame is encoded.
    batch_size only controls GPU batching, not feature shape.
    """
    import cv2

    vidcap = cv2.VideoCapture(path)
    if not vidcap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames in the video: {num_frames}")

    all_features = []
    batch = []

    def flush(batch):
        """Encode a batch of PIL frames -> (len(batch), 512) on CPU."""
        if not batch:
            return None
        inputs = torch.stack([preprocess(f) for f in batch]).to(device)
        with torch.no_grad():
            feats = model.encode_image(inputs).cpu()  # (len(batch), 512)
        return feats

    while True:
        success, image = vidcap.read()
        if not success:
            break
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        batch.append(pil)
        if len(batch) == batch_size:
            all_features.append(flush(batch))
            batch = []

    # leftover frames (the partial chunk) — encoded normally, no padding needed
    if batch:
        all_features.append(flush(batch))

    vidcap.release()

    if not all_features:
        print("Warning: no frames decoded.")
        return torch.empty(0, 512)

    frame_features = torch.cat(all_features, dim=0)  # (num_frames, 512)
    print(f"Encoded {frame_features.shape[0]} frames -> {tuple(frame_features.shape)}")
    return frame_features

def process_folder(folder_path, feature_save_path, split_file=None, chunks:int =16, last_chunk:bool = False):
    import os
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
    # Create target directory if it doesn't exist
    os.makedirs(feature_save_path, exist_ok=True)
    # Read split file to get list of videos to process
    if split_file is not None:
        with open(split_file, 'r') as f:
            split_videos = set(line.strip() for line in f.readlines())
        # Filter video files based on split
        video_files = [vf for vf in video_files if vf.replace('.mp4', '') in split_videos]
    for vf in video_files:
        path = os.path.join(folder_path, vf)
        # Skip if features already exist
        save_path = os.path.join(feature_save_path, vf.replace('.mp4', '.npy'))
        if os.path.exists(save_path):
            print(f'Features for {vf} already exist at {save_path}, skipping.')
            continue
        frame_features = clip_encode_video(path, batch_size=chunks)
        np.save(save_path, frame_features.numpy())
        print(f'Saved features for {vf} at {save_path}, shape: {frame_features.shape}')

path = "/home/ulas/codebase/fauna/data/34993-405620722.mp4"
frame_features = clip_encode_video(path, batch_size=16)
np.save("/home/ulas/codebase/fauna/data/clip/video_clip.npy", frame_features.numpy())
print(f'Extracted frame features shape: {frame_features.shape}')


