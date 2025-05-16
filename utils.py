import face_recognition as frg
import pickle as pkl
import os
import cv2
import numpy as np
import yaml
from collections import defaultdict
from twilio.rest import Client
import time
import requests
import streamlit as st  # Added for UI alerts

# === CONFIGURATION PATHS ===
PKL_PATH = r"D:\Black-Eyes-Intruders-detection-system-main\Blackeyes\dataset\database.pkl"

# Twilio credentials
account_sid = 'x'
auth_token = 'x'
from_number = 'x'
to_number = 'x'

client = Client(account_sid, auth_token)

alert_log = {}
ALERT_INTERVAL = 60  # seconds

def send_alert(message):
    try:
        msg = client.messages.create(
            body=message,
            from_=from_number,
            to=to_number
        )
        print(f"✅ Alert sent: {msg.sid}")
        return True
    except Exception as e:
        print(f"❌ Error sending alert: {e}")
        return False

def get_location():
    try:
        res = requests.get("https://ipinfo.io/json", timeout=5)
        data = res.json()
        city = data.get("city", "")
        region = data.get("region", "")
        country = data.get("country", "")
        loc = data.get("loc", "")  # lat,long
        return f"{city}, {region}, {country} (Coordinates: {loc})"
    except Exception as e:
        print(f"Location fetch failed: {e}")
        return "Location unavailable"

def check_and_alert(name):
    current_time = time.time()
    if name == 'Unknown':
        if current_time - alert_log.get(name, 0) > ALERT_INTERVAL:
            location = get_location()
            message = f"⚠️ Unknown face detected!\n📍 Location: {location}"
            message_sent = send_alert(message)
            if message_sent:
                alert_log[name] = current_time
                return True
            else:
                return False
    return None

def send_anomaly_alert(anomaly_type):
    location = get_location()
    message = f"🚨 Anomaly Detected: {anomaly_type}\n📍 Location: {location}"
    success = send_alert(message)
    if success:
        st.success(f"✅ Anomaly alert sent for {anomaly_type}")
    else:
        st.error(f"❌ Failed to send anomaly alert for {anomaly_type}")

def get_databse():
    with open(PKL_PATH, 'rb') as f:
        database = pkl.load(f)
    return database

def recognize(image, TOLERANCE):
    database = get_databse()
    known_encoding = [database[id]['encoding'] for id in database.keys()]
    name = 'Unknown'
    id = 'Unknown'
    face_locations = frg.face_locations(image)
    face_encodings = frg.face_encodings(image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = frg.compare_faces(known_encoding, face_encoding, tolerance=TOLERANCE)
        distance = frg.face_distance(known_encoding, face_encoding)
        name = 'Unknown'
        id = 'Unknown'

        if True in matches:
            match_index = matches.index(True)
            name = database[match_index]['name']
            id = database[match_index]['id']
            distance = round(distance[match_index], 2)
            cv2.putText(image, str(distance), (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    return image, name, id

def isFaceExists(image):
    face_location = frg.face_locations(image)
    return len(face_location) > 0

def submitNew(name, id, image, old_idx=None):
    database = get_databse()

    if type(image) != np.ndarray:
        image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)

    if not isFaceExists(image):
        return -1

    encoding = frg.face_encodings(image)[0]
    existing_id = [database[i]['id'] for i in database.keys()]

    if old_idx is not None:
        new_idx = old_idx
    else:
        if id in existing_id:
            return 0
        new_idx = len(database)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    database[new_idx] = {'image': image, 'id': id, 'name': name, 'encoding': encoding}

    with open(PKL_PATH, 'wb') as f:
        pkl.dump(database, f)
    return True

def get_info_from_id(id):
    database = get_databse()
    for idx, person in database.items():
        if person['id'] == id:
            name = person['name']
            image = person['image']
            return name, image, idx
    return None, None, None

def deleteOne(id):
    database = get_databse()
    id = str(id)
    for key, person in list(database.items()):
        if person['id'] == id:
            del database[key]
            break
    with open(PKL_PATH, 'wb') as f:
        pkl.dump(database, f)
    return True

# Placeholder for dataset builder (if needed)
def build_dataset():
    DATASET_DIR = r"D:\Black-Eyes-Intruders-detection-system-main\Blackeyes\dataset"
    information = defaultdict(dict)
    counter = 0
    for image in os.listdir(DATASET_DIR):
        image_path = os.path.join(DATASET_DIR, image)
        image_name = image.split('.')[0]
        parsed_name = image_name.split('_')
        person_id = parsed_name[0]
        person_name = ' '.join(parsed_name[1:])
        if not image_path.endswith('.jpg'):
            continue
        image = frg.load_image_file(image_path)
        information[counter]['image'] = image
        information[counter]['id'] = person_id
        information[counter]['name'] = person_name
        information[counter]['encoding'] = frg.face_encodings(image)[0]
        counter += 1

    with open(os.path.join(DATASET_DIR, 'database.pkl'), 'wb') as f:
        pkl.dump(information, f)

if __name__ == "__main__":
    deleteOne(4)
