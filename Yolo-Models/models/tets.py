# import cv2
# from ultralytics import YOLO  # Replace with YOLOv11 API if applicable

# # Function to load YOLOv11 model
# def load_model(model_path=None):
#     """
#     Load YOLOv11 model dynamically. Requires the `ultralytics` library or YOLOv11-specific code.
#     """
#     if model_path:
#         print(f"Loading custom model from: {model_path}")
#         model = YOLO(model_path)
#     else:
#         print("Loading YOLOv11 default model")
#         model = YOLO('yolov11n.pt')  # Replace with the YOLOv11 default model path or config
#     return model

# # Function to perform real-time detection
# def run_detection(model):
#     """
#     Perform real-time detection using the given YOLOv11 model.
#     """
#     cap = cv2.VideoCapture(0)  # Open the default camera
#     if not cap.isOpened():
#         print("Error: Unable to access the camera")
#         return

#     print("Press 'q' to exit the detection")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Unable to read frame from camera")
#             break

#         # Perform inference
#         results = model.predict(source=frame, save=False, conf=0.25, show=False)

#         # Annotate the frame with detections
#         annotated_frame = results[0].plot()  # YOLOv8/YOLOv11 uses `plot()` to draw boxes

#         # Display the frame
#         cv2.imshow("YOLO Real-Time Detection", annotated_frame)

#         # Break the loop on 'q' key press
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the camera and close windows
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # Prompt the user to choose a model
#     print("Enter the path to your YOLO model (.pt file), or press Enter to use the default YOLOv11 model:")
#     model_path = input("Model path: ").strip()

#     # Load the YOLO model
#     model = load_model(model_path if model_path else None)

#     # Run real-time detection
#     run_detection(model)

