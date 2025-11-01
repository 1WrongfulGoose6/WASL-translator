import cv2
import numpy as np
import tensorflow as tf
import joblib
import os


root_dir = r"E:\Ashwin\importantFiles\Programming Projects\WASL-translator\Runs"

preproc_options = {
    "gaussian":  "JustGaussianBlur",
    "otsu":      "JustOtsu",
    "robert":    "JustRobert",
    "grayscale": "JustGrayscale",
    "prewitt":   "JustPrewitt",
    "sobel":     "JustSobel"
}

print("Select preprocessing type:")
for key in preproc_options:
    print(f" - {key}")

choice = input("\nEnter choice: ").lower().strip()

if choice not in preproc_options:
    raise ValueError("Invalid choice. Restart & choose from menu.")

run_folder = preproc_options[choice]
model_path = os.path.join(root_dir, run_folder, "results", f"{run_folder}.keras")
encoder_path = os.path.join(root_dir, run_folder, "label_encoder.pkl")

print(f"\nLoading model: {model_path}")
print(f"Loading encoder: {encoder_path}")

model = tf.keras.models.load_model(model_path)
label_encoder = joblib.load(encoder_path)


# setup preprocessing function based on model
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if choice == "gaussian":
        gray = cv2.GaussianBlur(gray, (5,5), 0)

    elif choice == "otsu":
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif choice == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gray = cv2.convertScaleAbs(cv2.sqrt(gx**2 + gy**2))

    elif choice == "robert":
        kernelx = np.array([[1, 0], [0, -1]])
        kernely = np.array([[0, 1], [-1, 0]])
        gx = cv2.filter2D(gray, -1, kernelx)
        gy = cv2.filter2D(gray, -1, kernely)
        gray = cv2.convertScaleAbs(cv2.sqrt(gx**2 + gy**2))

    elif choice == "prewitt":
        kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
        kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        gx = cv2.filter2D(gray, -1, kernelx)
        gy = cv2.filter2D(gray, -1, kernely)
        gray = cv2.convertScaleAbs(cv2.sqrt(gx**2 + gy**2))

    # grayscale = just grayscale (no change)

    # resize for model
    gray = cv2.resize(gray, (64, 64))
    gray = gray.astype("float32") / 255.0
    gray = np.expand_dims(gray, axis=-1)  # (64,64,1)
    gray = np.expand_dims(gray, axis=0)   # batch dimension

    return gray

# start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam")
    exit()

print("Webcam running. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Preprocess frame for model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))  # adjust to your model input
    input_frame = resized.reshape(1, 64, 64, 1) / 255.0

    # Predict
    preds = model.predict(input_frame)
    pred_label = label_encoder.inverse_transform([np.argmax(preds)])[0]

    # Show prediction on frame
    cv2.putText(frame, f"Prediction: {pred_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Live Translator", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Exited.")
