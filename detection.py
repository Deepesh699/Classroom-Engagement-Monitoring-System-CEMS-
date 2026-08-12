import cv2

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open USB camera.")
    exit()

print("CEMS Face Detection started.")
print("Press Q to exit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a box around every detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            2
        )

    # Show number of detected faces
    cv2.putText(
        frame,
        f"Students Detected: {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("CEMS - Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()