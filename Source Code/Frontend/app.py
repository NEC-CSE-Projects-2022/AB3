from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load trained Logistic Regression model
model = joblib.load("logistic_model.pkl")

# Load Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# ---------------- SCREEN 1: UPLOAD IMAGE (FIXED) ----------------
@app.route("/", methods=["GET", "POST"])
def upload():
    result = None

    if request.method == "POST":
        image = request.files["image"]

        if image:
            image_path = os.path.join("static", "upload.jpg")
            image.save(image_path)

            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                return render_template("upload.html",
                                       result="⚠ Invalid image")

            # 🔑 SAME AS CAMERA: detect eyes first
            eyes = eye_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(eyes) == 0:
                # No eye detected → assume not drowsy for image
                result = "✅ DRIVER NOT DROWSY"
            else:
                (x, y, w, h) = eyes[0]
                eye_img = gray[y:y+h, x:x+w]

                eye_img = cv2.GaussianBlur(eye_img, (5, 5), 0)
                eye_img = cv2.resize(eye_img, (160, 128))
                eye_img = eye_img.astype("float32") / 255.0
                eye_img = eye_img.flatten().reshape(1, -1)

                prob_closed = model.predict_proba(eye_img)[0][1]

                if prob_closed > 0.60:
                    result = "🚨 DRIVER DROWSINESS DETECTED"
                else:
                    result = "✅ DRIVER NOT DROWSY"

    return render_template("upload.html", result=result)

# ---------------- CAMERA STREAM (UNCHANGED – WORKING) ----------------
def generate_frames():
    cap = cv2.VideoCapture(0)
    closed_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(eyes) == 0:
            closed_count += 1
        else:
            (x, y, w, h) = eyes[0]
            eye_img = gray[y:y+h, x:x+w]

            eye_img = cv2.resize(eye_img, (160, 128))
            eye_img = eye_img.astype("float32") / 255.0
            eye_img = eye_img.flatten().reshape(1, -1)

            prob_closed = model.predict_proba(eye_img)[0][1]

            if prob_closed > 0.60:
                closed_count += 1
            else:
                closed_count = 0

            cv2.rectangle(frame, (x, y), (x+w, y+h),
                          (255, 255, 0), 2)

        if closed_count >= 15:
            text = "DROWSINESS DETECTED"
            color = (0, 0, 255)
        else:
            text = "NO DROWSINESS"
            color = (0, 255, 0)

        cv2.putText(frame, text, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()

# ---------------- SCREEN 2: CAMERA PAGE ----------------
@app.route("/camera")
def camera():
    return render_template("camera.html")

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(debug=True)
