from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the YOLO model
model = YOLO("best (2).pt")  # Load your trained model for solar panel detection

# Default detection settings
detection_settings = {
    "confidence": 0.5  # default confidence level
}

# Function to process image or video
def process_file(file_path):
    if file_path.endswith(('.jpg', '.jpeg', '.png')):
        image = cv2.imread(file_path)
        results = model(image)
        return annotate_image(image, results)
    elif file_path.endswith(('.mp4', '.avi', '.mov')):
        return process_video(file_path)
    return None

# Function to annotate image
def annotate_image(image, results):
    total_boxes = 0

    for result in results:
        labels = result.names
        boxes = result.boxes.data

        for box in boxes:
            cls = int(box[5].item())
            label = labels[cls]
            confidence = box[4].item()

            if confidence >= detection_settings["confidence"]:
                # Count all bounding boxes
                total_boxes += 1
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box[:4].tolist())
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, f'{label} {confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Add total bounding box count in a separate annotation
    cv2.putText(image, f'Solar Panel Count: {total_boxes}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
    cv2.imwrite(output_path, image)
    return output_path

# Function to process video
def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.avi')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    total_boxes = 0  # Counter for total bounding boxes

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame_boxes = count_boxes(results)
        total_boxes += frame_boxes
        annotated_frame = annotate_video_frame(frame, results, frame_boxes, total_boxes)
        out.write(annotated_frame)

    cap.release()
    out.release()
    return output_path

# Function to annotate video frames
def annotate_video_frame(frame, results, frame_boxes, total_boxes):
    for result in results:
        labels = result.names
        boxes = result.boxes.data

        for box in boxes:
            cls = int(box[5].item())
            label = labels[cls]
            confidence = box[4].item()

            if confidence >= detection_settings["confidence"]:
                x1, y1, x2, y2 = map(int, box[:4].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Add total bounding box count in a separate annotation
    cv2.putText(frame, f'Frame Solar Panel Count: {frame_boxes}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Function to count boxes from results
def count_boxes(results):
    box_count = 0
    for result in results:
        boxes = result.boxes.data
        for box in boxes:
            confidence = box[4].item()
            if confidence >= detection_settings["confidence"]:
                box_count += 1
    return box_count

# Route to the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        output_path = process_file(file_path)
        return redirect(url_for('uploaded_file', filename=os.path.basename(output_path)))

# Route to display the processed file
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route to update settings
@app.route('/update_settings', methods=['POST'])
def update_settings():
    detection_settings["confidence"] = float(request.form.get("confidence"))
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)
