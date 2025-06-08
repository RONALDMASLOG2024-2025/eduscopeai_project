from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid
import shutil
from collections import defaultdict
from pathlib import Path
import csv
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

model = YOLO('model/best.pt')

# List of organisms your model can detect
ORGANISM_LIST = [
    'Amoeba', 'Euglena', 'Hydra', 'Paramecium',
    'Rod_bacteria', 'Spherical_bacteria', 'Spiral_bacteria', 'Yeast'
]

# CSV logger
def log_detection_csv(uid, counts, output_path='detections.csv'):
    file_exists = os.path.isfile(output_path)
    total = sum(counts.values())
    row = {
        'timestamp': datetime.now().isoformat(),
        'filename': f"result_{uid}.png",
        'total_detections': total,
        'detected_anything': total > 0
    }
    for name in ORGANISM_LIST:
        row[name] = counts.get(name, 0)
    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

@app.route('/', methods=['GET', 'POST'])
def index():
    result_message = None
    upload_preview = None
    result_image = None
    organism_counts = {}

    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            ext = os.path.splitext(file.filename)[1]
            uid = uuid.uuid4().hex
            safe_filename = f"{uid}{ext}"

            # Save uploaded file
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            file.save(upload_path)
            upload_preview = '/' + upload_path.replace("\\", "/")

            # Run YOLO detection
            results = model(upload_path, save=True, conf=0.3)
            yolo_output_dir = Path(results[0].save_dir)
            detected_img_path = next(yolo_output_dir.glob("*.*"), None)

            # Process detection results
            names = model.names
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]
                organism_counts[label] = organism_counts.get(label, 0) + 1

            total = sum(organism_counts.values())

            if total > 0:
                # Detections found – use YOLO result image
                if detected_img_path and detected_img_path.exists():
                    final_name = f"result_{uid}{detected_img_path.suffix}"
                    final_path = os.path.join(app.config['RESULT_FOLDER'], final_name)
                    shutil.copy(str(detected_img_path), final_path)
                    result_image = '/' + final_path.replace("\\", "/") + f"?v={uuid.uuid4().hex}"
                result_message = f"✅ {total} microorganisms detected."
            else:
                # No detections – show original image
                result_image = upload_preview
                result_message = "🧪 No microorganisms detected in this image."

            # Log detection regardless
            log_detection_csv(uid, organism_counts)

    return render_template(
        'index.html',
        result=result_message,
        upload_image=upload_preview,
        result_image=result_image,
        counts=organism_counts
    )

if __name__ == '__main__':
    app.run(debug=True)
