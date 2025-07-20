from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

# App setup
app = Flask(__name__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']
num_classes = len(class_names)
weights_path = "efficientnet_garbage.pth"

# Load model
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/')
def home():
    return render_template('one.html')  # If you're serving the HTML from Flask

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]

    # Define biodegradable classes
    biodegradable_classes = {'biological', 'cardboard', 'clothes', 'paper', 'shoes'}
    waste_type = "Biodegradable" if label in biodegradable_classes else "Non-biodegradable"

    return jsonify({
        'prediction': label,
        'type': waste_type
    })

if __name__ == '__main__':
    app.run(debug=True)
