from ultralytics import YOLO
import cv2
import os
from collections import defaultdict

# === CONFIGURATION ===
MODEL_PATH = "model/best.pt"               # path to your trained YOLOv8 model
IMG_PATH = "test_images/sample1.jpg"       # path to your test image
CONFIDENCE = 0.3                           # minimum confidence threshold for predictions

# === LOAD YOLOv8 MODEL ===
model = YOLO(MODEL_PATH)

# === RUN INFERENCE ===
results = model(IMG_PATH, save=True, conf=CONFIDENCE)
predict_dir = results[0].save_dir
filename = os.path.basename(IMG_PATH)

# === RESULT IMAGE PATH (Fix filenames with spaces/parentheses) ===
clean_filename = filename.replace(" ", "_").replace("(", "").replace(")", "")
result_img_path = os.path.join(predict_dir, clean_filename)

# === PRINT DETECTION COUNTS ===
boxes = results[0].boxes
names = model.names

detected_counts = defaultdict(int)

for box in boxes:
    cls_id = int(box.cls[0])
    class_name = names[cls_id]
    detected_counts[class_name] += 1

print("\n✅ Detected Organisms:")
for name, count in detected_counts.items():
    print(f" - {name}: {count}")

# === SHOW RESULT IMAGE ===
if os.path.exists(result_img_path):
    img = cv2.imread(result_img_path)
    cv2.imshow("YOLOv8 Detection Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"\n❌ Processed image not found at {result_img_path}")
