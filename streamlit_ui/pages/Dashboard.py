import streamlit as st
import PIL
import numpy as np
import pandas as pd
import os

st.title("Dashboard!!")
folder_list = os.listdir()
selected_folder = st.sidebar.selectbox("Select a Folder", folder_list)
if selected_folder:
    folder_path = os.path.join(os.getcwd(), selected_folder)

    if not os.path.isdir(folder_path):
        st.write("Invalid folder path!")
    else:
        file_list = os.listdir(folder_path)
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                st.write(f"File: {file_name}")
                image = PIL.Image.open(file_path)
                numpy_array_label = np.array(image)
                value_counts = pd.DataFrame(numpy_array_label.reshape(-1,1)).value_counts()
                st.write(value_counts)
                st.write("---")
