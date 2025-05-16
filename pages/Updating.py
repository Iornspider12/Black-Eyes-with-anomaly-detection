import streamlit as st 
import cv2
import yaml 
import pickle 
from utils import submitNew, get_info_from_id, deleteOne
import numpy as np

st.set_page_config(layout="wide")
st.title("Face Recognition App")
st.write("This app is used to add new faces to the dataset")

menu = ["Adding","Deleting", "Adjusting"]
choice = st.sidebar.selectbox("Options", menu)

# Helper function to handle submission and errors
def handle_submission(name, id, image, old_idx=None):
    if name == "" or id == "":
        st.error("Please enter both name and ID")
    else:
        ret = submitNew(name, id, image, old_idx=old_idx)
        if ret == 1: 
            st.success("Student Added/Updated Successfully")
        elif ret == 0: 
            st.error("Student ID already exists")
        elif ret == -1: 
            st.error("No face detected in the picture")

if choice == "Adding":
    name = st.text_input("Name", placeholder='Enter name')
    id = st.text_input("ID", placeholder='Enter ID')

    # Create options: Upload image or use webcam
    upload = st.radio("Upload image or use webcam", ("Upload", "Webcam"))

    if upload == "Upload":
        uploaded_image = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'])
        if uploaded_image is not None:
            try:
                st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
                submit_btn = st.button("Submit", key="submit_btn")
                if submit_btn:
                    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
                    handle_submission(name, id, image)
            except Exception as e:
                st.error(f"Error processing image: {e}")
    elif upload == "Webcam":
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            try:
                # To read image file buffer with OpenCV:
                bytes_data = img_file_buffer.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                st.image(cv2_img, caption='Captured Image', use_column_width=True)
                submit_btn = st.button("Submit", key="submit_btn")
                if submit_btn:
                    handle_submission(name, id, cv2_img)
            except Exception as e:
                st.error(f"Error processing webcam image: {e}")

elif choice == "Deleting":
    id = st.text_input("ID", placeholder='Enter ID')
    submit_btn = st.button("Submit", key="submit_btn")
    
    if submit_btn:
        try:
            name, image, _ = get_info_from_id(id)
            if name is None and image is None:
                st.error("Student ID does not exist")
            else:
                st.success(f"Name of student with ID {id} is: {name}")
                st.warning("Please check the image below to ensure you are deleting the correct student")
                st.image(image, caption='Student Image', use_column_width=True)
                del_btn = st.button("Delete", key="del_btn", on_click=deleteOne, args=(id,))
                if del_btn:
                    st.success("Student deleted")
        except Exception as e:
            st.error(f"Error retrieving student data: {e}")

elif choice == "Adjusting":
    id = st.text_input("ID", placeholder='Enter ID')
    submit_btn = st.button("Submit", key="submit_btn")
    
    if submit_btn:
        try:
            old_name, old_image, old_idx = get_info_from_id(id)
            if old_name is None and old_image is None:
                st.error("Student ID does not exist")
            else:
                with st.form(key='my_form'):
                    st.title("Adjusting Student Info")
                    col1, col2 = st.columns(2)
                    
                    new_name = col1.text_input("Name", key='new_name', value=old_name, placeholder='Enter new name')
                    new_id = col1.text_input("ID", key='new_id', value=id, placeholder='Enter new ID')
                    new_image = col1.file_uploader("Upload new image", key='new_image', type=['jpg', 'png', 'jpeg'])
                    
                    col2.image(old_image, caption='Current Image', use_column_width=True)
                    
                    form_submitted = st.form_submit_button(label='Submit')
                    if form_submitted:
                        if new_image is not None:
                            try:
                                new_image = cv2.imdecode(np.frombuffer(new_image.read(), np.uint8), cv2.IMREAD_COLOR)
                            except Exception as e:
                                st.error(f"Error processing new image: {e}")
                                new_image = old_image
                        else:
                            new_image = old_image
                        
                        handle_submission(new_name, new_id, new_image, old_idx=old_idx)
        except Exception as e:
            st.error(f"Error retrieving student data: {e}")
