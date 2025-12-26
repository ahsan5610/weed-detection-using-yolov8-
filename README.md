# 🌿 Weed Detection Web Application using YOLOv8 & Flask

An AI-powered **real-time weed detection web application** built using **YOLOv8**, **PyTorch**, and **Flask**.  
The system detects multiple weed species from uploaded images and live webcam streams, displaying **color-coded bounding boxes, confidence scores, coordinates, FPS**, and allows users to **download detection results as CSV**.

---

## 🚀 Features

### 🖼️ Image-Based Detection
- Upload an image for weed detection
- Detects multiple weed species in a single image
- Color-coded bounding boxes for each weed type
- Displays:
  - Weed name
  - Confidence score
  - Bounding box coordinates (x1, y1, x2, y2)
- Adjustable **confidence threshold slider**
- Download detection results as **CSV file**

### 🎥 Real-Time Webcam Detection
- Live weed detection using webcam
- Real-time bounding boxes and labels
- **Live FPS (Frames Per Second)** display
- Confidence filtering applied dynamically

### 🌐 Web Application
- Fully responsive UI (desktop & mobile)
- Modern UI with animations and icons
- Built using Flask with HTML, CSS, and JavaScript

---

## 🧠 Model Details

- **Model**: YOLOv8 (Ultralytics)
- **Training Platform**: Google Colab
- **Dataset**: Custom Cotton Weed Dataset (Roboflow)
- **Classes**: Multiple weed species (custom-trained)
- **Weights File**: `best.pt`

---

## 🛠️ Tech Stack

| Component | Technology |
|--------|-----------|
| Deep Learning | YOLOv8 |
| Framework | PyTorch |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Image Processing | OpenCV |
| Dataset | Roboflow |
| Visualization | Bounding boxes & labels |

---

## 📂 Project Structure

weed-detection-app/
│
├── app.py
├── best.pt
├── requirements.txt
├── README.md
│
├── templates/
│ ├── index.html
│ └── webcam.html
│
├── static/
│ ├── css/
│ ├── js/
│ └── uploads/
│
└── outputs/
├── detections/
└── results.csv


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/weed-detection-app.git
cd weed-detection-app

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
python app.py

5️⃣ Open in Browser
http://127.0.0.1:5000
