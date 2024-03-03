import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].a * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].b * image_height})'
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].k * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].l * image_height})'
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].m * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].n * image_height})'
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].e * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].f * image_height})'

            )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        cv2.imwrite(
            '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                ##ekrana yazdırma
                x, y = hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y
                x1, y1 = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
                a, b = hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y
                a1, b1 = hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y
                k, l = hand_landmarks.landmark[13].x, hand_landmarks.landmark[13].y
                k1, l1 = hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y
                m, n = hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y
                m1, n1 = hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y
                e, f = hand_landmarks.landmark[1].x, hand_landmarks.landmark[1].y
                e1, f1 = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y

                font = cv2.FONT_HERSHEY_PLAIN

                if (e > e1 and a1 > a) and (k1 > k and m1 > m):
                    cv2.putText(image, "GULE GULE", (10, 50), font, 4, (0, 0, 0), 3)
                if (e1 > e and a1 > a) and (k1 > k and m1 > m):
                    cv2.putText(image, "HAYIR", (10, 50), font, 4, (0, 0, 0), 3)
                if ((e1 < e and a1 < a) and (k1 < k and m1 < m)) and x > x1:
                    cv2.putText(image, "EVET", (10, 50), font, 4, (0, 0, 0), 3)
                if ((e1 > e and a1 < a) and (k1 < k and m1 < m)) and x > x1:
                    cv2.putText(image, "SIKILDIM", (10, 50), font, 4, (0, 0, 0), 3)
                if ((e1 > e and a1 < a) and (k1 < k and m1 < m)) and x1 > x:
                    cv2.putText(image, "1", (10, 50), font, 4, (0, 0, 0), 3)
                if ((e1 > e and a1 > a) and (k1 < k and m1 < m)) and x1 > x:
                    cv2.putText(image, "2", (10, 50), font, 4, (0, 0, 0), 3)
                if ((e1 > e and a1 < a) and (k1 > k and m1 < m)) and x > x1:
                    cv2.putText(image, "3", (10, 50), font, 4, (0, 0, 0), 3)
                if ((e1 > e and a1 < a) and (k1 > k and m1 > m)) and x > x1:
                    cv2.putText(image, "4", (10, 50), font, 4, (0, 0, 0), 3)
                if ((k1 > k and a1 > a) and x1 < x) and (m1 < m and e1 < e):
                    cv2.putText(image, "SENI SEVIYORUM", (10, 50), font, 4, (0, 0, 0), 3)

                ##

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
