import numpy as np
from ultralytics import YOLO
import os

# 1. Setup - Point to your AP-10K or AnimalPose .pt file
# Example: "ap10k-pose.pt" or "yolo11n-pose-dog.pt"
model_path = "ap10k-pose.pt" 
model = YOLO(model_path)

video_folder = "/home/ulas/datasets/animal_reconst"
output_root = "/home/ulas/datasets/animal_reconst/kps"

if not os.path.exists(output_root):
    os.makedirs(output_root)

# 2. Dynamic Feature Discovery
# We check the model's internal keypoint shape (usually [17, 3] or [24, 3])
sample_results = model("https://ultralytics.com/images/bus.jpg") # Dummy check
num_kpts = model.model.yaml['kpt_shape'][0]
kpt_features = num_kpts * 3
# Total features: 1 (Track ID) + 4 (BBox) + 1 (Class) + kpt_features
total_features = 6 + kpt_features 

print(f"Model detected with {num_kpts} keypoints. Feature vector length: {total_features}")

# 3. Processing Loop
video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

for v_file in video_files:
    video_path = os.path.join(video_folder, v_file)
    video_name = os.path.splitext(v_file)[0]
    
    # We use persist=True to track individual animals (e.g., Lion #1 vs Lion #2)
    results = model.track(video_path, conf=0.3, stream=True, imgsz=1280, persist=True)
    
    all_video_data = []

    for result in results:
        frame_container = np.zeros((10, total_features))
        frame_container[:, 0] = -1 # Default ID to -1
        
        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            bboxes = result.boxes.xyxy.cpu().numpy()
            kpts = result.keypoints.data.cpu().numpy() # [N, num_kpts, 3]
            
            num_found = min(len(track_ids), 10)
            
            for i in range(num_found):
                frame_container[i, 0] = track_ids[i]    # Persistent Track ID
                frame_container[i, 1] = classes[i]      # Animal Species ID
                frame_container[i, 2:6] = bboxes[i]     # BBox [x1, y1, x2, y2]
                frame_container[i, 6:] = kpts[i].flatten() # All Kpts [x, y, conf, ...]

        all_video_data.append(frame_container)

    # Save logic
    final_array = np.array(all_video_data)
    save_path = os.path.join(output_root, f"{video_name}_animal_features.npy")
    np.save(save_path, final_array)
    print(f"Video {v_file} saved: {final_array.shape}")

print("\nAll animals tracked and saved.")
