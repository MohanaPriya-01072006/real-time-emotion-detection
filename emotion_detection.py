import cv2
from fer import FER

# Initialize detector (mtcnn=True gives better accuracy for faces)
detector = FER(mtcnn=True)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Detect emotions
    emotions = detector.detect_emotions(frame)

    # Draw results
    for face in emotions:
        (x, y, w, h) = face["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get top emotion
        emotion, score = max(face["emotions"].items(), key=lambda x: x[1])
        cv2.putText(frame, f"{emotion} ({score:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

    # Show window
    cv2.imshow("Real-Time Emotion Detection", frame)

    # Exit on ESC or q
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
