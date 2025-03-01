
import os
import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
from moviepy.editor import ImageSequenceClip
from PIL import Image

# Load the trained model
model = YOLO(r'C:\Users\swath\OneDrive\Desktop\u\best (2).pt')

# Define the class labels
class_labels = {
    0: "crab",
    1: "fish",
    2: "jellyfish",
    3: "shrimp",
    4: "small fish",
    5: "starfish"
}

def image_detection_pipeline(image):
    # Convert image to OpenCV format
    img = np.array(image)

    # If the image has 4 channels (RGBA), convert it to 3 channels (RGB)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Run YOLO model on the image
    results = model(img)

    # Draw bounding boxes and labels on the image
    for result in results:
        for box in result.boxes:
            # Get the predicted class ID
            class_id = int(box.cls)

            # Get the class label from the dictionary
            class_label = class_labels.get(class_id, 'Unknown')

            # Get the bounding box coordinates
            bbox = box.xyxy[0].cpu().numpy()

            # Draw bounding box and label on the image
            img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(img, class_label, (int(bbox[0]), int(bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return img


# Define the detection pipeline function for videos
def video_detection_pipeline(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return []
    
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break  # Break when the video ends

        # Run YOLO model on each frame
        results = model(frame)

        # Draw bounding boxes and labels on the frame
        for result in results:
            for box in result.boxes:
                # Get the predicted class ID
                class_id = int(box.cls)

                # Get the class label from the dictionary
                class_label = class_labels.get(class_id, 'Unknown')

                # Get the bounding box coordinates
                bbox = box.xyxy[0].cpu().numpy()

                # Draw bounding box and label on the frame
                frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                cv2.putText(frame, class_label, (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        frames.append(frame)

    cap.release()
    
    return frames

# Streamlit app layout
st.title("Under Water Object Detection")

st.sidebar.title("Upload an Image or Video")
media_type = st.sidebar.selectbox("Choose the media type", ['Image', 'Video'])

if media_type == 'Image':
    uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Load the image using PIL
        image = Image.open(uploaded_image)

        # Perform image detection
        processed_image = image_detection_pipeline(image)

        # Convert the processed image back to RGB for Streamlit display
        st.image(processed_image, caption="Processed Image", use_column_width=True)

elif media_type == 'Video':
    uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        # Create the directory 'temp_video' if it doesn't exist
        if not os.path.exists("temp_video"):
            os.makedirs("temp_video")
        
        # Save the video file temporarily
        video_temp_path = os.path.join("temp_video", uploaded_video.name)
        with open(video_temp_path, "wb") as f:
            f.write(uploaded_video.read())
        
        # Perform video detection
        processed_frames = video_detection_pipeline(video_temp_path)

        # Convert frames to a clip and save as video
        if processed_frames:
            clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in processed_frames], fps=24)
            clip.write_videofile("output_video.mp4", codec="libx264")

            # Display the processed video in Streamlit
            st.video("output_video.mp4")
