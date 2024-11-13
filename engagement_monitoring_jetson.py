
import cv2
import dlib

# Load pre-trained facial landmark detector
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_eye_aspect_ratio(eye_points):
    # Calculate Eye Aspect Ratio (EAR) to detect gaze orientation
    # Example calculation for engagement monitoring
    p2_minus_p6 = eye_points[1].y - eye_points[5].y
    p3_minus_p5 = eye_points[2].y - eye_points[4].y
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * (eye_points[0].x - eye_points[3].x))
    return ear

# Start capturing video
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Calculate EAR for each eye
        left_eye_points = [landmarks.part(n) for n in range(36, 42)]
        right_eye_points = [landmarks.part(n) for n in range(42, 48)]
        left_ear = get_eye_aspect_ratio(left_eye_points)
        right_ear = get_eye_aspect_ratio(right_eye_points)

        # Determine if the student is looking at the screen (simple threshold for demo)
        is_engaged = left_ear > 0.2 and right_ear > 0.2
        engagement_status = "Engaged" if is_engaged else "Distracted"

        cv2.putText(frame, f"Engagement: {engagement_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Engagement Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
