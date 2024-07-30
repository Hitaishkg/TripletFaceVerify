import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Hide GPU devices

import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Define the Triplet Loss Function
@tf.function
def _triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:, 0, :], y_pred[:, 1, :], y_pred[:, 2, :]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + 0.5  # Adjust margin if needed
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss

# Function to load anchor images from a folder
def load_anchor_images(anchor_folder):
    anchor_images = []
    for filename in os.listdir(anchor_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(anchor_folder, filename)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, (100, 100))  # Resize as needed
                image = image / 255.0  # Normalize to [0, 1]
                anchor_images.append(image)
    return np.array(anchor_images)

# Load your trained model with custom object scope
custom_objects = {'_triplet_loss': _triplet_loss}
triplet_model = load_model('/home/hitaish/Documents/facerecognition/triplet_model.h5', custom_objects=custom_objects)

# Function to preprocess a single image for prediction
def preprocess_image(image):
    resized = cv2.resize(image, (100, 100))
    normalized = resized / 255.0
    processed = np.expand_dims(normalized, axis=0)
    return processed

# Function to predict using the loaded triplet model
def predict_from_camera(anchor_images=None):
    cap = cv2.VideoCapture(0)
    
    # Load anchor images if not provided
    if anchor_images is None:
        print("Calculating embeddings for anchor images...")
        anchor_images = load_anchor_images('/home/hitaish/Documents/facerecognition/build_siamies_network/data/anchor')
        anchor_embeddings = triplet_model.predict([anchor_images, anchor_images, anchor_images])
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to capture image")
            break
        
        processed_frame = preprocess_image(frame)
        
        # Predict using the triplet model
        embeddings = triplet_model.predict([processed_frame, processed_frame, processed_frame])
        
        # Perform face verification (example logic, replace with your own)
        similarity_scores = np.sum((embeddings - anchor_embeddings)**2, axis=1)
        threshold = 0.8 # Adjust as needed
        verified = similarity_scores < threshold
        
        # Draw bounding box and display verification status
        if np.all(verified):
            cv2.rectangle(frame, (20, 20), (180, 180), (0, 255, 0), 2)  # Green bounding box for verified
            cv2.putText(frame, "Verified", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (20, 20), (180, 180), (0, 0, 255), 2)  # Red bounding box for not verified
            cv2.putText(frame, "Not Verified", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Camera Feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run the prediction function
predict_from_camera(anchor_images=None)
