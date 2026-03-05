import streamlit as st
from models.natural_disaster_dataset import NaturalDisasterDataset
import os
import matplotlib.pyplot as plt
from PIL import Image

st.title("Natural Disaster Image Classification")

st.set_page_config(
    page_title="Natural Disaster Classification",
    page_icon=":rescue_worker_helmet:",
    layout="wide"
)

st.markdown(
    """
    ### Welcome! 
    
    This is an image classifier for natural disasters. 
    This model can classify natural disasters from the following categories:
    """
)

image_path = "./data/raw/Train/"
dataset = NaturalDisasterDataset(image_path)

categories = ["Fire", "Earthquake", "Flood", "Normal"]

col1, col2, col3, col4 = st.columns(4)
cols = [col1, col2, col3, col4]

for i in  range(len(categories)):
    category_path = os.path.join(image_path, categories[i])
    for img_file in os.listdir(category_path):
        full_path = os.path.join(category_path, img_file)
        with cols[i]:
            img = Image.open(full_path)
            st.image(img, caption=f"{categories[i]}")
        break

st.divider()

st.markdown(
    """
    ### Classify Your Own Images!

    Click the link below to upload an image and see how the model classifies it.
    """
)
st.page_link("pages/Model_Demo.py")

st.divider()

st.markdown(
    """
    ### Classifier Analytics

    Click the link below to see the classifier's performance.
    """
)
st.page_link("pages/Model_Performance.py")

st.divider()

st.markdown(
    """
    ### Learn More about this Project

    To learn more about the inspiration for this project, please see below.
    """
)
st.page_link("pages/About.py")
