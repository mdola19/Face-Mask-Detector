import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pygame import mixer
import time

mixer.init()
mixer.music.load('./Resources/noMaskAudio.wav')

webcam_feed_detecting = cv.VideoCapture(0)

haar_face = cv.CascadeClassifier('./Resources/haar_face.xml')

loaded_model = tf.keras.models.load_model('./Machine Learning Model/model.h5')
classes = ["wearing a mask", "wearing no mask"]

def rescaleFrame(frame, scale=1):

    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

menu = cv.imread("./Resources/Menu.jpg")
menu = rescaleFrame(menu, scale = 0.25)
cv.imshow("Menu", menu)

if cv.waitKey(0) & 0xFF==ord('s'):

    cv.destroyAllWindows()

    while True:
        frames_success, frames = webcam_feed_detecting.read()
        frames = cv.rectangle(frames, (0, 450), (640, 480), (255, 0, 0), -1)
        frames = cv.putText(frames, "Click 'D' on the keyboard to exit", (10, frames.shape[0] - 10), cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), thickness=2)
        cv.imshow("Original Webcam Feed", frames)

        frames_resized = rescaleFrame(frames, scale = 0.75) 

        gray = cv.cvtColor(frames_resized, cv.COLOR_BGR2GRAY, dst=None, dstCn=None) 

        face_landmarks = haar_face.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (50,50))
        
        for (x,y,w,h) in face_landmarks:
            try:
                detected = frames_resized[y:y+h, x:x+w]
                rescaleDetected = rescaleFrame(detected, scale = 1.5)
                cv.imshow("Cropped Face", rescaleDetected)

                modelInput = cv.cvtColor(rescaleDetected, cv.COLOR_BGR2RGB)
                modelInput = cv.resize(modelInput, (224, 224), interpolation = cv.INTER_AREA)
                modelInput = modelInput.reshape(1, 224, 224, 3)
                plt.imshow(modelInput.reshape(224, 224, 3))

                predict = loaded_model.predict(modelInput)
                prediction_results = classes[np.argmax(predict[0])]
                confidence_score = int(((predict[0][np.argmax(predict[0])])*100))
                prediction_text = f'I am {confidence_score} percent sure that you are {prediction_results}.'                

                print(prediction_text)
                result = cv.rectangle(frames, (0, 0), (frames.shape[1], 30), (255,0,0), -1)
                result = cv.putText(frames, prediction_text, (20, 20), cv.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), thickness = 2) 
                cv.imshow("Final Result (Shows last frame with face in it)", result) 

                cv.moveWindow("Original Webcam Feed", -10,0)
                cv.moveWindow("Cropped Face", 580, 510)
                cv.moveWindow("Final Result (Shows last frame with face in it)", 638, 0)

            except:
                print("Face was not detected")

            if prediction_results == "wearing no mask":
                mixer.music.play()
                time.sleep(5)

        if cv.waitKey(20) & 0xFF==ord('d'):
            break

    webcam_feed_detecting.release()
    cv.destroyAllWindows