import streamlit as st
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

st.title("📊 YOLOv8 Model Performance")

# Load trained model
model_path = "D:/anomaly/runs/train/best.pt"
model = YOLO(model_path)

# Load validation dataset
data_yaml = "D:/anomaly/datasets/data.yaml"

# Run validation
results = model.val(data=data_yaml)

# Extract ground truth & predicted values
y_true, y_pred = [], []
for result in results:
    for pred in result.boxes.cls.tolist():  
        y_pred.append(int(pred))
    for gt in result.boxes.target.tolist():  
        y_true.append(int(gt[0]))

# Compute accuracy metrics
precision = precision_score(y_true, y_pred, average="weighted", zero_division=1)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=1)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=1)

# Display results
st.metric(label="🎯 Precision", value=f"{precision:.4f}")
st.metric(label="🔄 Recall", value=f"{recall:.4f}")
st.metric(label="🔥 F1 Score", value=f"{f1:.4f}")
