def analyse_engagement(detection):
    """
    Estimate engagement from YuNet facial landmarks.

    YuNet detection format:
    0-3   : face bounding box (x, y, width, height)
    4-5   : right eye
    6-7   : left eye
    8-9   : nose
    10-11 : right mouth corner
    12-13 : left mouth corner
    14    : detection confidence
    """

    right_eye = (
        float(detection[4]),
        float(detection[5])
    )

    left_eye = (
        float(detection[6]),
        float(detection[7])
    )

    nose = (
        float(detection[8]),
        float(detection[9])
    )

    right_mouth = (
        float(detection[10]),
        float(detection[11])
    )

    left_mouth = (
        float(detection[12]),
        float(detection[13])
    )

    # Midpoint between both eyes
    eye_mid_x = (
        right_eye[0] + left_eye[0]
    ) / 2

    eye_mid_y = (
        right_eye[1] + left_eye[1]
    ) / 2

    # Midpoint between mouth corners
    mouth_mid_y = (
        right_mouth[1] + left_mouth[1]
    ) / 2

    # Horizontal eye distance
    eye_distance = abs(
        left_eye[0] - right_eye[0]
    )

    # Vertical facial distance
    face_vertical = (
        mouth_mid_y - eye_mid_y
    )

    if eye_distance < 1 or face_vertical < 1:
        return {
            "status": "Unknown",
            "score": 0,
            "orientation": "Unknown",
            "yaw_ratio": 0,
            "pitch_ratio": 0
        }

    # ---------------------------------------
    # LEFT / RIGHT HEAD ORIENTATION
    # ---------------------------------------

    yaw_ratio = (
        nose[0] - eye_mid_x
    ) / eye_distance

    # ---------------------------------------
    # UP / DOWN HEAD ORIENTATION
    # ---------------------------------------

    pitch_ratio = (
        nose[1] - eye_mid_y
    ) / face_vertical

    # ---------------------------------------
    # ORIENTATION LABEL
    # ---------------------------------------

    if yaw_ratio < -0.32:
        orientation = "Head Left"

    elif yaw_ratio > 0.32:
        orientation = "Head Right"

    elif pitch_ratio < 0.32:
        orientation = "Looking Up"

    elif pitch_ratio > 0.72:
        orientation = "Looking Down"

    else:
        orientation = "Forward"

    # ---------------------------------------
    # ENGAGEMENT SCORE
    # ---------------------------------------

    yaw_penalty = min(
        abs(yaw_ratio) / 0.60 * 45,
        45
    )

    normal_pitch = 0.52

    pitch_difference = abs(
        pitch_ratio - normal_pitch
    )

    pitch_penalty = min(
        pitch_difference / 0.35 * 30,
        30
    )

    score = round(
        95
        - yaw_penalty
        - pitch_penalty
    )

    score = max(
        20,
        min(95, score)
    )

    # ---------------------------------------
    # ENGAGEMENT STATUS
    # ---------------------------------------

    if score >= 75:
        status = "Engaged"

    elif score >= 55:
        status = "Neutral"

    else:
        status = "Low Engagement"

    return {
        "status": status,
        "score": score,
        "orientation": orientation,
        "yaw_ratio": round(yaw_ratio, 2),
        "pitch_ratio": round(pitch_ratio, 2)
    }