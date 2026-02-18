import streamlit as st
from PIL import Image

st.title("Model Demo")

st.divider()

st.markdown(
    """
    ### Upload an image and see how the model classifies the image!
    """
)

image_file = st.file_uploader(
    "Upload an image file",
    ["jpg", "jpeg", "png"]
)

if image_file is not None:
    try:
        img = Image.open(image_file)
        st.image(img, caption="Image")
    except Exception as e:
        st.error(f"Error loading image: {e}")

st.divider()
st.subheader("The model predicted:")