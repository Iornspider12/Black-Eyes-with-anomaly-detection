import face_recognition
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import os
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

# Load images and labels from your dataset
def load_dataset(image_dir):
    images = []
    labels = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:  # Only add if encoding is found
                images.append(encoding[0])
                # Assuming the label is part of the filename (e.g., person1.jpg)
                labels.append(filename.split('.')[0])
    return np.array(images), np.array(labels)

# Load your dataset
X, y = load_dataset(r'C:\Users\PC\Downloads\Blackeyes\lfw_images')

# Debugging: Print the number of loaded images and labels
print(f"Total encodings: {len(X)}, Total labels: {len(y)}")

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Debugging: Print unique labels in training and test sets
print("Unique labels in training set:", np.unique(y_train))
print("Unique labels in test set:", np.unique(y_test))

# Train a simple classifier (e.g., KNN) on the embeddings
knn_clf = KNeighborsClassifier(n_neighbors=3)  # You can try adjusting n_neighbors
knn_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_clf.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Print the metrics
print("Performance Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
