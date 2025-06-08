from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid
import shutil
from collections import defaultdict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load YOLOv8 model
model = YOLO('model/best.pt')

@app.route('/', methods=['GET', 'POST'])
def index():
    result_message = None
    upload_preview = None
    result_image = None
    organism_counts = {}

    if request.method == 'POST':
        file = request.files['image']
        if file:
            ext = os.path.splitext(file.filename)[1]
            uid = uuid.uuid4().hex
            safe_filename = f"{uid}{ext}"

            # Save upload
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            file.save(upload_path)
            upload_preview = '/' + upload_path.replace("\\", "/")

            # Run YOLOv8
            results = model(upload_path, save=True, conf=0.3)

            # Actual output file saved by YOLO
            yolo_output_path = results[0].save_dir + '/' + os.path.basename(results[0].path)

            # Copy to results for web display
            final_result_name = f"result_{safe_filename}"
            final_result_path = os.path.join(app.config['RESULT_FOLDER'], final_result_name)
            if os.path.exists(yolo_output_path):
                shutil.copy(yolo_output_path, final_result_path)
                result_image = '/' + final_result_path.replace("\\", "/")

                # Count detected organisms
                names = model.names
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    label = names[cls_id]
                    organism_counts[label] = organism_counts.get(label, 0) + 1

                total = sum(organism_counts.values())
                result_message = f"✅ {total} microorganisms detected in this image."
            else:
                result_message = "❌ YOLOv8 result image not found."

    return render_template(
        'index.html',
        result=result_message,
        upload_image=upload_preview,
        result_image=result_image,
        counts=organism_counts
    )

if __name__ == '__main__':
    app.run(debug=True)
