import streamlit as st
import numpy as np
import cv2
import torch
from torchvision import models, transforms
from PIL import Image

# Load a pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# Image preprocessing transformations
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def remove_background(input_image):
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # Move to GPU if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)['out'][0]

    # Get the mask (Class 0 is background, Class 15 is person)
    output_predictions = output.argmax(0).byte().cpu().numpy()

    # Convert the mask into a 3-channel image
    mask = output_predictions == 15  # Assuming class 15 represents the person
    mask = np.stack([mask] * 3, axis=-1)

    # Convert the PIL image to a NumPy array
    image_np = np.array(input_image)

    # Apply the mask to remove the background
    result_image = image_np * mask

    # Convert the result to an image
    result = Image.fromarray(result_image.astype(np.uint8))
    return result

# Streamlit app layout
st.title("Image Background Remover")
st.write("Upload an image, and the app will remove the background.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an image
    input_image = Image.open(uploaded_file).convert("RGB")
    
    # Display the original image
    st.image(input_image, caption='Original Image', use_column_width=True)
    
    # Remove the background
    result = remove_background(input_image)

    # Display the resulting image
    st.image(result, caption='Image with Background Removed', use_column_width=True)

    # Download button for the result image
    result.save("output.png")
    with open("output.png", "rb") as file:
        st.download_button(label="Download Image", data=file, file_name="output.png", mime="image/png")
