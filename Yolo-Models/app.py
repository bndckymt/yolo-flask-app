from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os

app = Flask(__name__)

# Path to models directory
MODEL_DIR = "models"
models = {model_name: os.path.join(MODEL_DIR, model_name) for model_name in os.listdir(MODEL_DIR)}
selected_model_path = None
current_model = None

# Video capture
camera = cv2.VideoCapture(0)

def load_model(model_path):
    """Load YOLO model dynamically based on the model file."""
    print(f"Loading model: {model_path}")
    if model_path.endswith(".pt"):  # For PyTorch YOLO models (e.g., YOLOv5, YOLOv8)
        from ultralytics import YOLO  # Ensure YOLO library is installed
        return YOLO(model_path)
    elif model_path.endswith(".onnx"):  # For ONNX YOLO models
        import onnxruntime as ort
        return ort.InferenceSession(model_path)
    else:
        raise ValueError(f"Unsupported model format: {model_path}")

@app.route('/')
def home():
    """Render home page."""
    return render_template('home.html', models=list(models.keys()), selected_model=selected_model_path)

@app.route('/select_model', methods=['POST'])
def select_model():
    """Handle model selection."""
    global selected_model_path, current_model
    selected_model_name = request.form['model']
    if selected_model_name not in models:
        return "Invalid model selected", 400
    selected_model_path = models[selected_model_name]
    try:
        current_model = load_model(selected_model_path)  # Load the selected model
    except Exception as e:
        return f"Error loading model: {e}", 500
    return redirect(url_for('home'))

def generate_frames():
    """Generate video frames with YOLO detections."""
    global current_model
    while True:
        success, frame = camera.read()
        if not success:
            break
        if current_model is None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
            continue

        if hasattr(current_model, "predict"):  # For YOLOv8 and similar
            results = current_model.predict(source=frame, conf=0.25, show=False)
            annotated_frame = results[0].plot()
        elif hasattr(current_model, "run"):  # For ONNX models
            import numpy as np
            input_name = current_model.get_inputs()[0].name
            input_shape = current_model.get_inputs()[0].shape
            frame_resized = cv2.resize(frame, (input_shape[3], input_shape[2]))
            input_data = np.expand_dims(frame_resized, axis=0).astype(np.float32)
            outputs = current_model.run(None, {input_name: input_data})
            annotated_frame = frame  # Add your own logic for annotations
        else:
            annotated_frame = frame

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video feed route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
