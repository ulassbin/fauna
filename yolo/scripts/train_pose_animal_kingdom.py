from ultralytics import YOLO
from datetime import datetime
# Old approach:
# # 1. Load the architecture (YAML) and pretrained weights (PT)
# # We use the 'n' (nano) version for speed, or 'm' (medium) for better accuracy.
# model = YOLO("yolov8n-pose.yaml")  # This defines the network structure
# model.load("yolov8n-pose.pt")     # This loads the 'smart' weights to start from

# # 2. Train the model
# # IMPORTANT: Point 'data' to YOUR custom yaml, not coco8-pose.yaml
# results = model.train(
#     data="/home/ulas/datasets/animal_kingdom_yolop1/data.yaml", 
#     epochs=100, 
#     imgsz=640, 
#     batch=16,       # Adjust based on your GPU memory (8, 16, 32)
#     device=0,        # Use 0 for your first GPU, or 'cpu' if no GPU
#     project="animal_kingdom_pose", 
#     name="v1_ak_p1"
# )

# Load the Medium model
#model = YOLO("yolo11m-pose.yaml").load("yolo11m-pose.pt")

# Train with high-compute settings
# results = model.train(
#     data="/home/ulas/datasets/animal_kingdom_yolop1/data.yaml", 
#     epochs=1000,      
#     imgsz=640,       # Keep at 640 since source is 640x360
#     batch=32,       # Cranked up to fill your 16GB VRAM
#     workers=12,
#     device=0,
#     optimizer='SGD', 
#     cos_lr=True,     
#     # Better augmentation for diverse animals
#     augment=True,
#     overlap_mask=True, # Helps with crowded animal scenes
#     project=f"animal_kingdom_{datetime.now().strftime('%Y%m%d_%H%M')}", 
#     name="yolo11m_ak_p1_v1"
# )

# AI Assisted new paramaters:

# 1. Load the Large pose model
# 'yolo11l-pose.pt' is the Large version (better accuracy, higher compute)
#model = YOLO("yolo11l-pose.pt")

# 2. OPTIONAL: Callback for Validation Interval
# Since 'val_period' doesn't exist, we use this to validate every 5 epochs
def on_train_epoch_end(trainer):
    # Change '5' to your desired validation frequency
    validate_every_n = 5
    if (trainer.epoch + 1) % validate_every_n == 0:
        trainer.args.val = True
    else:
        trainer.args.val = False

#model.add_callback("on_train_epoch_end", on_train_epoch_end)

# 3. Start Training
#results = model.train(
#    data="/home/ulas/datasets/animal_kingdom_yolop1/data.yaml",
#    epochs=1000,
#    imgsz=640,
#    batch=16,            # Reduced to 16 for Large model on 16GB VRAM
#    device=0,
#    name="yolo11l_animal_pose",
#    save_period=50,      # Correct argument for saving checkpoints
#    exist_ok=True,
#    # label_smoothing=0.1, # Remove or comment out to clear the warning
#    # val_period=5,       # REMOVED: This was causing your error
#)



from ultralytics import YOLO

# Load your best checkpoint from the previous run
#model = YOLO("runs/pose/yolo11l_bird_pose/weights/best.pt") # Load checkpoint
#model.add_callback("on_train_epoch_end", on_train_epoch_end)
model = YOLO('yolo11l-pose.pt')
# Basic resume setting
results = model.train(
    data="/home/ulas/datasets/animal_kingdom_yolo_p3_bird/data.yaml",
    device=0,
    resume=False, # WITHOUT RESUME IT RESETS!
    epochs=500,        # Run for another 500 epochs
    imgsz=640,        # Keep high res for those 23 points
    batch=16,
    lr0=0.04,          # A slight bump (Original was 0.03, decayed to 0.017)
    optimizer='AdamW', # AdamW is better at handling "stuck" models than SGD
    #overlap_mask=True,
    warmup_epochs=5,
    name="yolo11l_bird_pose"
)

# Kinda restary resume:

#model.train(
#    data="/home/ulas/datasets/animal_kingdom_yolop1/data.yaml",
#    epochs=500,
#    imgsz=1280,
#    optimizer='AdamW',  # Smarter than SGD
#    lr0=0.002,          # Standard starting point for AdamW refinement
#    augment=True,       # Turn on extra flippin/scaling
#    overlap_mask=True,  # Good for overlapping animals
#    name="ak_refinement_v3"
#)
