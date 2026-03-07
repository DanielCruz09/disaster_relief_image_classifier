import streamlit as st
from PIL import Image
from models.resnet50 import ResNet50
import torch
from torchvision.transforms import Resize, transforms
import torch.nn.functional as F
import os

st.title("Model Demo")

classes = ["Non-Damage", "Earthquake", "Fire", "Flood"]

def get_model(weights_path=None):
    resnet50 = ResNet50(num_classes=4, lr=0.001)
    weights = torch.load(weights_path, map_location="cpu")
    resnet50.model.load_state_dict(weights["model_state_dict"])
    return resnet50

def get_prediction(image, model):
    resize = Resize(size=(224, 224))
    resized_image = resize(image)
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(resized_image).unsqueeze(0)
    logits = resnet50.model(image_tensor)
    probs = F.softmax(logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    return classes[prediction]

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

img = None
if image_file is not None:
    try:
        img = Image.open(image_file)
        st.image(img, caption="Image")
        st.divider()
        resnet50 = get_model("models/model_weights.pth")
        prediction = get_prediction(img, resnet50)
        st.subheader("The model predicted:")
        st.markdown(f"### {prediction}")
    except Exception as e:
        st.error(f"Error loading image: {e}")

