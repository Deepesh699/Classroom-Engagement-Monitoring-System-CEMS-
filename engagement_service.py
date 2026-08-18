def analyse_engagement(frame_width, face):
    x, y, w, h = face

    face_center = x + w // 2
    frame_center = frame_width // 2

    distance_from_center = abs(
        face_center - frame_center
    )

    if distance_from_center < frame_width * 0.15:
        return {
            "status": "Engaged",
            "score": 80
        }

    else:
        return {
            "status": "Neutral",
            "score": 50
        }