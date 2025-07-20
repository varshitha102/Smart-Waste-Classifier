♻️ Smart Garbage Classifier
An AI-powered system that not only classifies garbage into categories but also detects it in real-time, maps it to biodegradability, and offers web-based interaction for broader usability.

🚀 Key Features
Multi-Model Classifier
Trained and compared ResNet18, MobileNetV2, and EfficientNetB0 using transfer learning.

Real-Time Detection
Uses OpenCV to predict waste category live from webcam — no uploads required.

Biodegradable Mapping
Automatically maps categories to “Biodegradable” or “Non-Biodegradable”.

Web App Deployment
Built with Flask so users can upload images for prediction via a clean interface.

Model Performance Visualization
Metrics like accuracy, precision, recall, F1-score, and confusion matrix using Matplotlib and Seaborn.

🛠️ Tech Stack
PyTorch – Deep learning & model training

TorchVision & PIL – Image transforms and preprocessing

OpenCV – Webcam integration for live detection

Flask – Web app backend

Matplotlib & Seaborn – Visualizations

📊 Model Accuracy
Model	Accuracy
EfficientNetB0	93.4% ✅
ResNet18	92.1%
MobileNetV2	91.3%

📁 How to Run
Clone the repo


git clone https://github.com/your-username/smart-garbage-classifier.git

cd smart-garbage-classifier

Install dependencies
pip install -r requirements.txt

Run the Web App
python one.py

For Real-time Detection
python real_time_detection.py

🌍 Use Cases
Smart city garbage management

Eco-friendly bin sorting

Awareness and education on waste types

🔗 Live Demo / Code
👉 Link in the comments (if sharing on LinkedIn)
OR
👉 GitHub Repo

🤝 Let's Connect
Have ideas around AI for sustainability? Let’s collaborate!