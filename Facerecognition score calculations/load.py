import os
import face_recognition
import numpy as np

# Load images from folder
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image = face_recognition.load_image_file(img_path)
        images.append(image)
        labels.append(filename.split('.')[0])
    return images, labels

# Load dataset
X, y = load_images_from_folder(r'C:\Users\PC\Downloads\Blackeyes\lfw_images')

# Encode the images
X_encodings = [face_recognition.face_encodings(image)[0] for image in X if face_recognition.face_encodings(image)]

# Example of recognizing a new face
def recognize_face(unknown_image):
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
    results = face_recognition.compare_faces(X_encodings, unknown_encoding)
    return results

# Load a test image
test_image = face_recognition.load_image_file(r'C:\path\to\test_image.jpg')
matches = recognize_face(test_image)

print(matches)
