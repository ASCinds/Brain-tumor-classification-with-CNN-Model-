import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
from models import CoNvNet
import pandas as pd
import cv2

def predict_image(input_image):
    """
    Predict the class label for an input image.

    Parameters:
    - input_image (numpy.ndarray): Input image as a NumPy array.

    Returns:
    - str: Predicted class label.
    """
    # Load pre-trained model
    model = CoNvNet(num_classes=44)
    model.load_state_dict(torch.load("cancer_classifier_model.pth"))
    model.eval()

    # Load class information
    df = pd.read_csv("dataset.csv")

    # Define the transformation for input images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Preprocess the input image
    img = (input_image * 255).astype('uint8')
    img = 255 - img  # Invert colors

    cv2.imwrite("x.png", img)
    pil_image = Image.fromarray(img)

    # If the image has only one channel (grayscale), convert it to RGB
    if pil_image.mode == 'L':
        pil_image = pil_image.convert('RGB')

    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

    # Move the input tensor to the device used by the model (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_batch = input_batch.to(device)
    model = model.to(device)

    # Make the prediction
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class index
    _, predicted_index = torch.max(output, 1)

    # Get the class name from the DataFrame
    name = df["class_name"][df['class'] == predicted_index.item()].unique()[0]

    return name

# Define the Gradio interface
iface = gr.Interface(fn=predict_image, inputs="image", outputs="text")

# Launch the Gradio interface
iface.launch()
