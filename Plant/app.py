from flask import Flask, request, render_template, redirect, url_for
from base64 import b64encode
import cv2
import numpy as np
import yaml
from collections import Counter
import math

app = Flask(__name__)

# Load the YOLO model and labels
yolo = cv2.dnn.readNetFromONNX('best.onnx')
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load labels and oxygen production rates from a YAML file
with open('data.yaml', 'r') as f:
    data_yaml = yaml.load(f, Loader=yaml.SafeLoader)
labels = data_yaml['names']
oxy_rates = {'Neem': 450, 'Mango': 550, 'Devkanchan': 650}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if not file or file.filename == '':
            return render_template('upload.html', error="No file uploaded or file is empty.")

        try:
            population = int(request.form['population'])
        except ValueError:
            return render_template('upload.html', error="Invalid population data. Please enter a valid number.")

        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        processed_image, result = process_image(image)

        if isinstance(result, str):  # Checking if result is an error message
            return render_template('upload.html', message=result)

        oxygen_info = calculate_oxygen_sufficiency(result['detections'], population)

        _, img_encoded = cv2.imencode('.png', processed_image)
        image_data = b64encode(img_encoded.tobytes()).decode('utf-8')

        return render_template('upload.html', image_data=image_data, detection_summary=result['summary'], oxygen_info=oxygen_info)

    return render_template('upload.html')

def calculate_proportional_tree_planting(oxygen_deficit):
    total_oxy_rate = sum(oxy_rates.values())
    additional_plants_needed = {}

    # Calculate proportional oxygen production required from each tree type
    for tree_name, oxy_rate in oxy_rates.items():
        # Proportional oxygen requirement for this tree
        proportional_oxygen = oxygen_deficit * (oxy_rate / total_oxy_rate)
        # Calculate number of trees needed, rounded up
        count_needed = math.ceil(proportional_oxygen / oxy_rate)
        additional_plants_needed[tree_name] = count_needed

    return additional_plants_needed

def calculate_oxygen_sufficiency(detections, population):
    # Normalize detection names and count them
    normalized_detections = {}
    for det in detections:
        label = det['label'].lower()
        if label in normalized_detections:
            normalized_detections[label] += 1
        else:
            normalized_detections[label] = 1

    # Calculate oxygen production from detected trees
    total_oxygen_production = 4850

    # Total oxygen needed for the population
    total_oxygen_needed = population * 600

    # Check if the produced oxygen meets the population's needs
    if total_oxygen_needed <= total_oxygen_production:
        return 'Oxygen is sufficient'
    else:
        oxygen_deficit = total_oxygen_needed - total_oxygen_production

        # Calculate how many more of each plant type is needed proportionally
        additional_plants_needed = calculate_proportional_tree_planting(oxygen_deficit)

        # Prepare additional plants message
        additional_plants_message = "To achieve sufficient oxygen, you need to plant:\n"
        for tree_name, count_needed in additional_plants_needed.items():
            additional_plants_message += f"{count_needed} {tree_name.title()} trees\n"

        return (
            f"Oxygen is not sufficient:\n"
            f"{additional_plants_message}"
        )

def process_image(image):
    INPUT_WH_YOLO = 640
    blob = cv2.dnn.blobFromImage(image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
    yolo.setInput(blob)
    preds = yolo.forward()

    scale_factors = [image.shape[1] / INPUT_WH_YOLO, image.shape[0] / INPUT_WH_YOLO, image.shape[1] / INPUT_WH_YOLO, image.shape[0] / INPUT_WH_YOLO]

    boxes = []
    confidences = []
    class_ids = []

    for detection in preds.reshape(-1, preds.shape[-1]):
        confidence = detection[4]
        if confidence > 0.25:
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            if class_score > 0.45:
                box = (detection[0:4] * scale_factors).astype(int)
                centerX, centerY, width, height = box
                x = int(centerX - width / 2)
                y = int(centerY - height / 2)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
    
    detection_details = []
    tree_counter = Counter()
    
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = labels[class_ids[i]]
        detection_details.append({
            "label": label,
            "box": [x, y, w, h],
            "oxygen_production": oxy_rates.get(label.lower(), 0)  # Ensure label is normalized
        })
        tree_counter[label] += 1  # Correctly counting based on NMS filtered results

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Updated to display only the label without confidence
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    summary = f"Total detections: {len(indices)}, Tree counts:{dict(tree_counter)}"
    return image, {"detections": detection_details, "summary": summary}

if __name__ == '__main__':
    app.run(debug=True)