import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

tipIds = [4, 8, 12, 16, 20]


def countFingers(image, hand_landmarks, handNo=0):
    if hand_landmarks:
        # Get all Landmarks of the FIRST Hand VISIBLE
        landmarks = hand_landmarks[handNo].landmark
        # print(landmarks)

        # Count Fingers
        fingers = []

        for lm_index in tipIds:
            # Get Finger Tip and Bottom y Position Value
            finger_tip_y = landmarks[lm_index].y
            finger_bottom_y = landmarks[lm_index - 2].y

            # Check if ANY FINGER is OPEN or CLOSED
            if lm_index != 4:
                if finger_tip_y < finger_bottom_y:
                    fingers.append(1)
                    print("FINGER with id ", lm_index, " is Open")

                if finger_tip_y > finger_bottom_y:
                    fingers.append(0)
                    print("FINGER with id ", lm_index, " is Closed")

        # print(fingers)
        totalFingers = fingers.count(1)

        # Display Text
        text = f"Fingers: {totalFingers}"

        cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


def drawHandLandmarks(image, hand_landmarks):
    # Darw connections between landmark points
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)



# use cap.isOpened() insted of while True
while cap.isOpened():
    success, image = cap.read()

    # some cameras take a couple seconds to show show any images. This prevents empty images from causing errors
    if not success or image is None:
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Detect the Hands Landmarks
    results = hands.process(image_rgb)

    # Get landmark position from the processed result
    hand_landmarks = results.multi_hand_landmarks

    # Draw Landmarks
    drawHandLandmarks(image, hand_landmarks)

    # Get Hand Fingers Position
    countFingers(image, hand_landmarks)

    cv2.imshow("Media Controller", image)

    #cleaner way to exit out of the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
