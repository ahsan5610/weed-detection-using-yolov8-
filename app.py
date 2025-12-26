from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import torch
import cv2
import pandas as pd
import os
import time

# ================= SAFE GLOBALS FIX (CRITICAL) =================
from torch.nn.modules.container import Sequential
from ultralytics.nn.tasks import DetectionModel

# Allow all required classes safely
with torch.serialization.safe_globals([
    DetectionModel,
    Sequential
]):
    model = YOLO("best.pt")   # ✅ WILL LOAD WITHOUT ERROR
# ===============================================================

# ------------------ Flask App ------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------ Weed Colors ------------------
WEED_COLORS = {
    'Crabgrass': (0,255,0),
    'Eclipta': (255,0,0),
    'Goosegrass': (0,0,255),
    'Morningglory': (255,255,0),
    'Nutsedge': (255,0,255),
    'PalmerAmaranth': (0,255,255),
    'Prickly Sida': (128,0,128),
    'Waterhemp': (128,128,0),
    'purslane': (0,128,128)
}

# ------------------ IMAGE UPLOAD ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result_img_path = None
    detection_csv = None
    confidence = float(request.form.get("confidence", 0.25))

    if request.method == "POST":
        file = request.files["image"]
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        img = cv2.imread(img_path)
        results = model.predict(img, conf=confidence, imgsz=640)

        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = r.names[cls]
                conf = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cx, cy = (x1+x2)//2, (y1+y2)//2

                color = WEED_COLORS.get(name, (0,255,0))
                cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
                cv2.putText(img, f"{name} ({cx},{cy})",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                detections.append([name, conf, x1, y1, x2, y2, cx, cy])

        result_img_path = os.path.join(UPLOAD_FOLDER, "result_" + file.filename)
        cv2.imwrite(result_img_path, img)

        if detections:
            df = pd.DataFrame(detections,
                columns=["Weed","Confidence","x1","y1","x2","y2","cx","cy"])
            detection_csv = os.path.join(UPLOAD_FOLDER, "detections.csv")
            df.to_csv(detection_csv, index=False)

    return render_template("index.html",
        result_img=result_img_path,
        detection_csv=detection_csv
    )

# ------------------ REAL-TIME WEBCAM ------------------
def gen_frames(conf):
    cap = cv2.VideoCapture(0)
    prev = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame, conf=conf, imgsz=640)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = r.names[cls]
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cx, cy = (x1+x2)//2, (y1+y2)//2
                color = WEED_COLORS.get(name, (0,255,0))

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,f"{name} ({cx},{cy})",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        fps = 1 / (time.time() - prev)
        prev = time.time()
        cv2.putText(frame,f"FPS: {fps:.1f}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

@app.route("/video_feed")
def video_feed():
    conf = float(request.args.get("confidence", 0.25))
    return Response(gen_frames(conf),
        mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/webcam")
def webcam():
    return render_template("webcam.html")

# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(debug=True)
