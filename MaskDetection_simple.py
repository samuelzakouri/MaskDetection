import cv2
from time import sleep
import os
from datetime import date

# Path to directories
DATA_PATH = "./xml/"
SCREENSHOT_PATH = "./Screenshots_" + date.today().strftime("%b_%d_%Y") + '/'

# OpenCV Haar Cascade Classifiers
casc_face_Path = DATA_PATH + "haarcascade_frontalface_default.xml"  # for face detection
casc_eye_Path = DATA_PATH + "haarcascade_eye.xml"  # for eye detection
casc_mouth_Path = DATA_PATH + "haarcascade_mcs_mouth.xml"  # for mouth detection
casc_upperbody_Path = DATA_PATH + "haarcascade_upperbody.xml"  # for upper body detection

# cv2 Cascade Classifiers
faceCascade = cv2.CascadeClassifier(casc_face_Path)
eyeCascade = cv2.CascadeClassifier(casc_eye_Path)
mouthCascade = cv2.CascadeClassifier(casc_mouth_Path)
upperbodyCascade = cv2.CascadeClassifier(casc_upperbody_Path)

# Size of the screen
screen_width = 1440
screen_height = 990

# Adjust threshold value in range 80 to 105 based on your light.
bw_threshold = 80

# User message
font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30)
weared_mask_font_color = (0, 255, 0)
not_weared_mask_font_color = (0, 0, 255)
thickness = 2
font_scale = 1
weared_mask = "Thank You for wearing MASK"
not_weared_mask = "Please wear MASK to defeat Corona"

count = 1
# Read video
video_capture = cv2.VideoCapture(0)

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if ret:

        # Convert Image into gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert image in black and white
        (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)

        # Detect faces
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(60, 60)
        )

        # Face prediction for black and white
        faces_bw = faceCascade.detectMultiScale(black_and_white, 1.1, 4)

        if len(faces) == 0 and len(faces_bw) == 0:
            cv2.putText(frame, "No face found...", org, font, font_scale, weared_mask_font_color, thickness,
                        cv2.LINE_AA)
        elif len(faces) == 0 and len(faces_bw) == 1:
            # It has been observed that for white mask covering mouth, with gray image face prediction is not happening
            cv2.putText(frame, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
        else:
            # Draw rectangle on face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                # Detect lips counters
                mouth_rects = mouthCascade.detectMultiScale(gray, 1.5, 5)

            # Face detected but Lips not detected which means person is wearing mask
            if len(mouth_rects) == 0:
                cv2.putText(frame, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
            else:
                for (mx, my, mw, mh) in mouth_rects:

                    if (y < my < y + h):
                        # Face and Lips are detected but lips coordinates are within face cordinates which `means lips prediction is true and
                        # person is not waring mask
                        cv2.putText(frame, not_weared_mask, org, font, font_scale, not_weared_mask_font_color,
                                    thickness, cv2.LINE_AA)
                        break

        name = 'Mask Detection'
        # Display the resulting frame
        cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
        cv2.resizeWindow(name, screen_width, screen_height)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        cv2.imshow(name, frame)

    ch = cv2.waitKey(30) & 0xff

    # Screenshot
    if ch == ord('s'):
        # Create a directory screenshot with today's date
        if not os.path.exists(SCREENSHOT_PATH):
            os.makedirs(SCREENSHOT_PATH)
        name = "Screenshot" + str(count) + ".jpg"
        cv2.imwrite(SCREENSHOT_PATH + name, frame)
        count += 1

    # Exit
    if ch == ord("q"):
        break

# Release video
video_capture.release()
cv2.destroyAllWindows()
