import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

print("CEMS Engagement Detection started.")
print("Press Q to exit.")

while True:
    success, frame = camera.read()

    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    frame_height, frame_width = frame.shape[:2]
    frame_center = frame_width // 2

    engagement_status = "Disengaged"
    engagement_score = 0

    if len(faces) > 0:
        # Use the first detected face for Iteration 1
        x, y, w, h = faces[0]

        face_center = x + w // 2
        distance_from_center = abs(face_center - frame_center)

        # Draw face box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            2
        )

        # Simple prototype engagement logic
        if distance_from_center < frame_width * 0.15:
            engagement_status = "Engaged"
            engagement_score = 80
        else:
            engagement_status = "Neutral"
            engagement_score = 50

    # Display status
    cv2.putText(
        frame,
        f"Engagement: {engagement_status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Score: {engagement_score}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.imshow("CEMS - Engagement Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()