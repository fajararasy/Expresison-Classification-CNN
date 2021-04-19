from tkinter import*
from tkinter import messagebox
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import argparse
from collections import OrderedDict
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imutils import face_utils
import os
import os.path
import seaborn as sns
import csv
import random
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

root=Tk()
root.title("Classification Expression CNN")
header2 = "y_value"

if os.path.isfile('coba.csv'):
    os.remove('coba.csv')
    with open('coba.csv', 'w', newline='') as csvfile:
        fieldnames = ['nilainya']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
else:
    with open('coba.csv', 'w', newline='') as csvfile:
        fieldnames = ['nilainya']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def prediction():
        # Create the model
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))


        model.load_weights('91model.h5')

            # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)

            # dictionary which assigns each label an emotion (alphabetical order)
        # emotion_dict = {0: "Marah", 1: "Jijik", 2: "Takut", 3: "Senang", 4: "Neutral", 5: "Sedih", 6: "Terkerjut"}

            # start the webcam feed
        cap = cv2.VideoCapture(1)
        while True:
                # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            if not ret:
                break
            facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

                

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)   
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)

                emotions = ('Marah', 'Jijik', 'Takut', 'Senang', 'Neutral', 'Sedih', 'Terkerjut')
                maxindex = emotions[int(np.argmax(prediction))]

                
                y_value = maxindex

                with open('coba.csv', 'a', newline='') as csv_file:
                    fieldnames = [header2]
                    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    #csv_writer.writeheader()

                    info = {                        
                        header2: y_value                        
                    }
                    csv_writer.writerow(info)
                    time.sleep(1)    

                cv2.putText(frame, maxindex, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                #cv2.putText(frame,'Number of Faces : ' + str(len(faces)),(40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 1)
                
            cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        cap.release()
        cv2.destroyAllWindows()

b2=Button(root,text="Pengenalan Ekspresi Wajah",font=("Times New Roman", 20), width=20,height=6, bg="blue",fg='white',command=prediction)
b2.place(x=15, y=280)

def piechart():
    data_path = "coba.csv"
    df = pd.read_csv (data_path)
    marks_csv = pd.read_csv(data_path)
    no = marks_csv["nilainya"]
    plt.figure(1).gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(no, marker="o")
    plt.title("Record Hasil Ekspresi")
    plt.xlabel("Waktu(s)")
    plt.ylabel("Nilai Klasifikasi")
    plt.figure(2).gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    persen100 = df['nilainya'].value_counts().rename_axis('nilainya').reset_index(name='jml_label')
    plt.pie(persen100.jml_label, labels=persen100.nilainya, startangle=90, autopct='%.1f%%')
    plt.title("Persentase Ekspresi Yang Diklasifikasi")
    # persen100.apply(pd.value_counts).plot.pie(subplots=True)



    plt.show()

b3=Button(root,text="Hasil Ekspresi",font=("Times New Roman", 20),width=20,height=6, bg="blue",fg='white',command=piechart)
b3.place(x=345, y=280)

def reset():
    os.remove('coba.csv')
    with open('coba.csv', 'w', newline='') as csvfile:
        fieldnames = ['nilainya']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()



b3=Button(root,text="Reset",font=("Times New Roman", 20),width=20,height=6, bg="blue",fg='white',command=reset)
b3.place(x=675, y=280)

root.geometry("1000x720")
root.mainloop()