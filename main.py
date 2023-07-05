# imports
import math
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
from keras.utils.image_utils import img_to_array
import pyaudio
import audioop
import tkinter as tk

# init the audio
audio = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
THRESHOLD = 1000
HIGH = "high"
LOW = "low"
CURRENT_EMOTION = "neutrual"


# defining functions

# Printing emotion to the screen
def print_emotion_to_console(emotion_label):
    print("Detected emotion:", emotion_label)

# defining angles using math import
def get_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

def detect_head_turn(angle):
    if angle < -5:
        return "Right"
    elif angle > 5:
        return "Left"
    else:
        return "Center"

def print_head_turn_to_console(direction):
    print("Head turned:", direction)

def get_voice_level():
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    data = stream.read(CHUNK)
    rms = audioop.rms(data, 2)  # calculate the RMS amplitude
    return rms

# create blank canvas window
canvas = tk.Canvas(width=1000, height=1000)
canvas.pack()

# dictionary that maps emotion to image path
emotion_images = {
    "Angry": "Character/angry.png",
    "Excited": "Character/excited.png",
    "Upset": "Character/sad.png",
    "Happy": "Character/happy.png",
    "Neutral": "Character/neutral.png",
    "AngryLeft": "Character/angryLeft.png",
    "ExcitedLeft": "Character/excitedLeft.png",
    "NeutralLeft": "Character/neutralLeft.png",
    "NeutralRight": "Character/neutralRight.png",
    "UpsetLeft": "Character/sadLeft.png",
    "HappyLeft": "Character/happyLeft.png",
    "AngryRight": "Character/angryRight.png",
    "ExcitedRight": "Character/excitedRight.png",
    "UpsetRight": "Character/sadRight.png",
    "HappyRight": "Character/happyRight.png",
    "Neutral Loud": "Character/neutralLoud.png",
    "Neutral Loud Left": "Character/neutralLoudLeft.png",
    "Neutral Loud Right": "Character/neutralLoudRight.png",

}

# load images into dictionary
loaded_images = {}
for emotion, image_path in emotion_images.items():
    loaded_images[emotion] = tk.PhotoImage(file=image_path)
    loaded_images[emotion] = loaded_images[emotion].subsample(2)  # resize the image to half its size

mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

emotion_model = load_model('emotion_models.h5')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Could not find camera, please add")
            continue

        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # to retrieve the voice level
        voice_level = get_voice_level()

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face_region = image[y:y+h, x:x+w]

                # Get the eye landmarks
                landmarks = detection.location_data.relative_keypoints
                left_eye = [landmarks[0].x * iw, landmarks[0].y * ih]
                right_eye = [landmarks[1].x * iw, landmarks[1].y * ih]

                # Calculate the angle between the eyes
                angle = get_angle(left_eye[0], left_eye[1], right_eye[0], right_eye[1])

                # Detect head turn direction
                head_turn = detect_head_turn(angle)
                print_head_turn_to_console(head_turn)

                if face_region.shape[0] != 0 and face_region.shape[1] != 0:
                    face_region = cv2.resize(face_region, (48, 48))
                    face_region_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                    face_region_array = img_to_array(face_region_gray)
                    face_region_array = np.expand_dims(face_region_array, axis=0)
                    face_region_array /= 255

                    emotion_prediction = emotion_model.predict(face_region_array, verbose=0)
                    emotion_label_arg = np.argmax(emotion_prediction)
                    emotion_label = emotions[emotion_label_arg]

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (x, y - 10)
                    fontScale = 0.75
                    color = (0, 255, 0)
                    thickness = 2
                    image = cv2.putText(image, emotion_label, org, font, fontScale, color, thickness, cv2.LINE_AA)
                    # print_emotion_to_console(emotion_label)


        # Depending ont he head direction the emotion and position of the avatar will change ---------------------------
        if head_turn == 'Center':
            if emotion_label == 'Angry' and voice_level > THRESHOLD:
                CURRENT_EMOTION = 'Angry'
            elif emotion_label == 'Happy' and voice_level > THRESHOLD:
                CURRENT_EMOTION = 'Excited'
            elif emotion_label == 'Angry' and voice_level < THRESHOLD:
                CURRENT_EMOTION = 'Upset'
            elif emotion_label == 'Happy' and voice_level < THRESHOLD:
                CURRENT_EMOTION = 'Happy'
            elif emotion_label == 'Neutral' and voice_level > THRESHOLD:
                CURRENT_EMOTION = 'Neutral Loud'
            else:
                CURRENT_EMOTION = 'Neutral'

        if head_turn == 'Left':
            if emotion_label == 'Angry' and voice_level > THRESHOLD:
                CURRENT_EMOTION = 'AngryLeft'
            elif emotion_label == 'Happy' and voice_level > THRESHOLD:
                CURRENT_EMOTION = 'ExcitedLeft'
            elif emotion_label == 'Angry' and voice_level < THRESHOLD:
                CURRENT_EMOTION = 'UpsetLeft'
            elif emotion_label == 'Happy' and voice_level < THRESHOLD:
                CURRENT_EMOTION = 'HappyLeft'
            elif emotion_label == 'Neutral' and voice_level > THRESHOLD:
                CURRENT_EMOTION = 'Neutral Loud Left'
            else:
                CURRENT_EMOTION = 'NeutralLeft'

        if head_turn == 'Right':
            if emotion_label == 'Angry' and voice_level > THRESHOLD:
                CURRENT_EMOTION = 'AngryRight'
            elif emotion_label == 'Happy' and voice_level > THRESHOLD:
                CURRENT_EMOTION = 'ExcitedRight'
            elif emotion_label == 'Angry' and voice_level < THRESHOLD:
                CURRENT_EMOTION = 'UpsetRight'
            elif emotion_label == 'Happy' and voice_level < THRESHOLD:
                CURRENT_EMOTION = 'HappyRight'
            elif emotion_label == 'Neutral' and voice_level > THRESHOLD:
                CURRENT_EMOTION = 'Neutral Loud Right'
            else:
                CURRENT_EMOTION = 'NeutralRight'
        # --------------------------------------------------------------------------------------------------------------

        # Display The Emotion to the console ---------------------------------------------------------------------------
        print(CURRENT_EMOTION)
        # --------------------------------------------------------------------------------------------------------------

        # show image corresponding to CURRENT_EMOTION in canvas --------------------------------------------------------
        if CURRENT_EMOTION in loaded_images:
            canvas.delete("all")
            canvas.create_image(250, 250, image=loaded_images[CURRENT_EMOTION])
            canvas.update()
        # --------------------------------------------------------------------------------------------------------------
        cv2.imshow('MediaPipe Face Detection with Emotion Detection', image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        # --------------------------------------------------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()