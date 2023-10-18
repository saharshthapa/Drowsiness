import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import pandas
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
mixer.init()
sound = mixer.Sound('alarm.wav')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

face = cv2.CascadeClassifier('./haar cascade files/frontal_face.xml')
leye = cv2.CascadeClassifier('./haar cascade files/left_eye.xml')
reye = cv2.CascadeClassifier('./haar cascade files/right_eye.xml')

# Specify the folder where the CSV files will be stored
folder_name = "eye_logs"

# Create the folder if it doesn't exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)


# Specify the folder where the CSV files will be stored
folder_name = "eye_logs"

# Get a list of existing CSV files in the folder
existing_files = [file for file in os.listdir(folder_name) if file.endswith(".csv")]

# Generate a new CSV file name based on the number of existing files
new_csv_filename = f"eye_log{len(existing_files) + 1}.csv"

# Specify the full path to the new CSV file
csv_file_path = os.path.join(folder_name, new_csv_filename)

with open(csv_file_path, "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write the header row with column names
    csv_writer.writerow(["Timestamp", "Eye State","Score"])


lbl=['Close','Open']

model = load_model('models/cnnCat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(5, 30)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]
while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = np.argmax(model.predict(r_eye),axis=-1)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = np.argmax(model.predict(l_eye),axis=-1)

        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        eye_state_code=1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        eye_state_code=0
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>15):

        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
        if score >20:
            score=19
        # Get the current timestamp
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    with open(csv_file_path, "a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([current_time, eye_state_code,score])

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

cap.release()
cv2.destroyAllWindows()
# Load the data from the CSV file
data = pd.read_csv(csv_file_path)

data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.Score = pd.to_numeric(data.Score)

data['mov_avg'] = data['Score'].rolling(7).sum()

sns.set(rc={'figure.figsize':(11.7,8.27)})
ax = sns.lineplot(
    x='Timestamp',
    y='Score',
    data=data)

# Highlight the area under the curve where Score > 15
ax.fill_between(data['Timestamp'], 0, data['Score'], where=(data['Score'] > 15), color='red', alpha=0.2)

plt.show()

