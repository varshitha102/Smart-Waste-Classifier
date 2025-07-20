import torch
import torch.nn as nn
from torchvision import transforms
import cv2
from PIL import Image
from torchvision import models

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10  # Change this to match your model
weights_path = "efficientnet_garbage.pth"  # Replace with your weight file
class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']  # Adjust as needed

# Load EfficientNet with custom output layer
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

# Transform image for prediction
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Open webcam
cap = cv2.VideoCapture(0)
print("📷 Live EfficientNet camera classification started. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()]

    # Show prediction
    cv2.putText(frame, f"Prediction: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("EfficientNet Garbage Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  