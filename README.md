
# ♻️ Smart Garbage Classifier

An AI-powered system that classifies garbage into categories, detects them in real-time, maps to biodegradability, and offers a user-friendly web interface.

---

## 🚀 Key Features

- **Multi-Model Classifier**: Trained ResNet18, MobileNetV2, and EfficientNetB0 using transfer learning.
- **Real-Time Detection**: Uses OpenCV to predict waste category live via webcam—no need to upload.
- **Biodegradable Mapping**: Automatically maps categories to *Biodegradable* or *Non-Biodegradable*.
- **Web App Interface**: Built with Flask to allow users to upload and classify images.
- **Model Performance Visualization**: Accuracy, Precision, Recall, F1-score, Confusion Matrix visualized using Matplotlib and Seaborn.

---

## 🛠️ Tech Stack

- **PyTorch** – Deep learning model training  
- **TorchVision + PIL** – Image preprocessing  
- **OpenCV** – Webcam-based real-time detection  
- **Flask** – Web app deployment  
- **Matplotlib + Seaborn** – Model performance plots

---

## 📊 Model Accuracy

| Model           | Accuracy |
|----------------|----------|
| EfficientNetB0 | 93.4% ✅  |
| ResNet18       | 92.1%    |
| MobileNetV2    | 91.3%    |

---

## 📁 How to Run

### Clone the repo:
```bash
git clone https://github.com/varshitha102/Smart-Waste-Classifier.git
cd Smart-Waste-Classifier
````

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the Web App:

```bash
python one.py
```

### For Real-time Detection:

```bash
python real_time_detection.py
```

---

## 🌍 Use Cases

* Smart city garbage management
* Eco-friendly bin sorting
* Awareness & education on waste types

---

## 🔗 Live Demo / Code

👉 Link in the comments (if on LinkedIn)
👉 [GitHub Repo](https://github.com/varshitha102/Smart-Waste-Classifier)

---

## 🖼️ Sample Screens

**Homepage**
`images/home page.png`

**Biodegradable Detection**
`images/bio.png`

**Non-Biodegradable Detection**
`images/de.png`
`images/de1.png`

---

## 🤝 Let’s Connect

Have ideas around AI for sustainability? Feel free to connect and collaborate!


```
