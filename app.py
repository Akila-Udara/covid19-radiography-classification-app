# - Importing the dependencies
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import Net

# - CSS Styling
st.markdown(
    """
    <style>
    .centered-heading {
        text-align: center;
        padding-bottom: 40px;
    }
    .result {
        text-align:center;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Difining transformations
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.markdown("<h1 class='centered-heading'>COVID-19 Radiography Image Classification System</h1>", unsafe_allow_html=True)

# - Uploading multiple images
with st.form("my-form", clear_on_submit=True):
        uploaded_images = st.file_uploader("Upload a single radiography image or multiple radiography images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        submitted = st.form_submit_button("submit")

if uploaded_images:
    image_count = 0
    for uploaded_image in uploaded_images:
        image_count += 1
        image = Image.open(uploaded_image)
        st.image(image, use_column_width=True)

        # - Loading the saved model
        loaded_model = torch.load('covid_classification_model.pt')
        loaded_model.eval()

        # - Performing inference
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            prediction = loaded_model(image)
        
        # - Displaying the prediction
        if prediction.item() >= 0.5:
            st.markdown(
                f"<h5 class='result'>Prediction for the Radiography Image {image_count} - <span style='color:red;'>COVID-19 Positive</span></h1>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h5 class='result'>Prediction for the Radiography Image {image_count} - <span style='color:green;'>COVID-19 Negative</span></h1>",
                unsafe_allow_html=True
            )
