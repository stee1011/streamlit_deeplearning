import torch
import streamlit as st
from torchvision import models, transforms
from PIL import Image

#  Load CIFAR-10-trained ResNet18
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model

# 🔁 Inference-time transform (match training)
inference_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])

# 🔤 CIFAR-10 label list
cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# 🔍 Predict function
def predict(image: Image.Image, model):
    img_tensor = inference_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = output.argmax(1).item()
        return cifar10_classes[pred_idx]

# 🖼️ Streamlit UI
st.title("CIFAR-10 Image Classifier")
st.write("Upload an image and get the predicted CIFAR-10 class.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "jfif"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model = load_model()
    label = predict(image, model)
    st.success(f"Predicted Label: **{label.upper()}**")
