import streamlit as st
import pickle
import yaml
import numpy as np

# Load configuration and database
try:
    cfg = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    PKL_PATH = cfg['PATH']["PKL_PATH"]
except Exception as e:
    st.error(f"Error loading configuration file: {e}")
    st.stop()

try:
    with open(PKL_PATH, 'rb') as file:
        database = pickle.load(file)
except Exception as e:
    st.error(f"Error loading pickle file: {e}")
    st.stop()

st.set_page_config(layout="wide")
st.title("Database Viewer")
st.write("This page displays the contents of the database.")

# Display data
Index, Id, Name, Image = st.columns([0.5, 0.5, 3, 3])

# Check if database is empty
if not database:
    st.write("No data available.")
else:
    for idx, person in database.items():
        with Index:
            st.write(idx)
        with Id:
            st.write(person.get('id', 'N/A'))
        with Name:
            st.write(person.get('name', 'N/A'))
        with Image:
            image = person.get('image', None)
            if image is not None:
                if isinstance(image, (bytes, bytearray)):  # Check if image is in bytes format
                    st.image(image, width=200)
                elif isinstance(image, np.ndarray):  # Check if image is a NumPy array
                    if image.size > 0:  # Ensure the array is not empty
                        st.image(image, width=200)
                    else:
                        st.write("Empty image array.")
                elif isinstance(image, str) and image.startswith(('http://', 'https://')):  # Check if image is a URL
                    st.image(image, width=200)
                else:
                    st.write("Image format not recognized.")
            else:
                st.write("No image available")
