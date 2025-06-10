from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
import uuid
import shutil
from pathlib import Path
import csv
from datetime import datetime
import glob
import json
from openai import OpenAI  # Correct import for modern SDK
from dotenv import load_dotenv
from PIL import Image
load_dotenv()

# Initialize Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

STATIC_PATHS = [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER']]
for path in STATIC_PATHS:
    os.makedirs(path, exist_ok=True)


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


# Resize uploaded image to 640x640 (YOLO default)
img = Image.open(upload_path)
img = img.resize((640, 640))
img.save(upload_path)


# Load YOLO model
try:
    model = YOLO('model/best.pt')
except Exception as e:
    print("Model loading failed:", e)


# Organism classes (same as your YOLO training labels)
ORGANISM_LIST = [
    'Amoeba', 'Euglena', 'Hydra', 'Paramecium',
    'Rod_bacteria', 'Spherical_bacteria', 'Spiral_bacteria', 'Yeast'
]

# CSV logging
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
    organism_info = {}

    # Load extended explanations
    explanations = {}
    try:
        with open("organism_explanations_extended.json", "r") as f:
            explanations = json.load(f)
    except Exception as e:
        print("❌ Error loading explanation file:", e)

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

            # Clear YOLO output folders
            for folder in glob.glob('runs/detect/predict*'):
                shutil.rmtree(folder)

            # Run YOLO
            results = model(upload_path, save=True, conf=0.3)
            yolo_output_dir = Path(results[0].save_dir)
            detected_img_path = next(yolo_output_dir.glob("*.*"), None)

            # Count detections
            names = model.names
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]
                organism_counts[label] = organism_counts.get(label, 0) + 1

            total = sum(organism_counts.values())

            if detected_img_path and detected_img_path.exists():
                final_ext = detected_img_path.suffix
                final_name = f"result_{uid}{final_ext}"
                final_path = os.path.join(app.config['RESULT_FOLDER'], final_name)
                shutil.copyfile(detected_img_path, final_path)
                result_image = '/' + final_path.replace("\\", "/") + f"?v={uuid.uuid4().hex}"

            if total > 0:
                result_message = f"✅ {total} microorganisms detected."
            else:
                result_message = "🧪 No microorganisms detected in this image."
                result_image = upload_preview + f"?v={uuid.uuid4().hex}"

            # Build organism_info dict
            for label in organism_counts.keys():
                if label in explanations:
                    organism_info[label] = {
                        "count": organism_counts[label],
                        "common_name": explanations[label].get("common_name", label),
                        "scientific_name": explanations[label].get("scientific_name", "Unknown"),
                        "category": explanations[label].get("category", "Unknown"),
                        "habitat": explanations[label].get("habitat", "Unknown"),
                        "description": explanations[label].get("description", ""),
                        "interactions": explanations[label].get("interactions", [])
                    }

            log_detection_csv(uid, organism_counts)

    return render_template(
        'index.html',
        result=result_message,
        upload_image=upload_preview,
        result_image=result_image,
        counts=organism_counts,
        explanations=organism_info
    )

# ✅ OpenAI SDK client setup (using GPT-4o-mini)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    data = request.json
    question = data.get('question', '')
    context = data.get('context', '')
    history = data.get('history', [])

    system_message = {
        "role": "system",
        "content": (
            "You are a helpful, scientific AI assistant focused on microbiology. "
            "Only show calculation steps if the user explicitly asks for them. "
            "Keep answers concise and professional, but friendly."
        )
    }

    # Start with system message, context as a message, and then full history
    messages = [system_message]

    # Optional: inject context as a fixed message
    messages.append({
        "role": "user",
        "content": f"Context: These microorganisms were detected: {context}"
    })

    # Add chat history
    messages.extend(history)

    # Add the current question
    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )
        answer = response.choices[0].message.content
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'answer': f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
