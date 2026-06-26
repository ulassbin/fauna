import numpy as np
from ultralytics import YOLO
import cv2
import os

# 1. Setup
video_folder = "/home/ulas/codebase/fauna/data"
output_root = "/home/ulas/codebase/fauna/data/out"
model_path = "/home/ulas/codebase/fauna/yolo/weights/animalkingdom/best.pt"
model = YOLO(model_path)

if not os.path.exists(output_root):
    os.makedirs(output_root)

video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

for v_file in video_files:
    video_path = os.path.join(video_folder, v_file)
    video_name = os.path.splitext(v_file)[0]
    
    print(f"Processing with Tracking: {v_file}")
    
    # persist=True keeps the tracker alive across frames
    results = model.track(video_path, conf=0.1, stream=True, imgsz=640, persist=True, device=0) # Lets go for 0.2 specific to animal kingdom. # can also change to 640?
    
    frames_dir = os.path.join(output_root, video_name)
    os.makedirs(frames_dir, exist_ok=True)

    all_video_data = []
    frame_idx = 0

    for result in results:
        # Fixed container for 10 people, each with 56 features
        # (1 ID + 4 BBox + 51 Kpts)
        frame_container = np.zeros((10, 74)) # 56 human, 1+4+69=75 Animals
        
        if result.boxes is not None and len(result.boxes) > 0:
            # Extract data
            classes = result.boxes.cls.cpu().numpy()     # [N]
            bboxes = result.boxes.xyxy.cpu().numpy()     # [N, 4]
            # YOLO keypoints: [N, 17, 3]
            kpts = result.keypoints.data.cpu().numpy()   
            
            num_found = min(len(classes), 10)
            
            for i in range(num_found):
                # Column 0: Class ID
                frame_container[i, 0] = classes[i]
                
                # Columns 1-4: BBox
                frame_container[i, 1:5] = bboxes[i]
                
                # Columns 5-55: Flattened Keypoints (17*3 = 51) # Animal kingdom is 23
                frame_container[i, 5:] = kpts[i].flatten()

        all_video_data.append(frame_container)

        annotated = result.plot()
        cv2.imwrite(os.path.join(frames_dir, f"{frame_idx:05d}.png"), annotated)
        frame_idx += 1

    # Save as [Frames, 10, 56] # 56 human ,69 animal
    final_array = np.array(all_video_data)
    save_path = os.path.join(output_root, f"{video_name}.npy")
    np.save(save_path, final_array)
    print(f"Saved features shape: {final_array.shape}")

print("\nDone! All video features extracted.")
