import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from tkinter import *  
  
from tkinter import messagebox  
# from PIL import ImageGrab
 
path = 'Images'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr
 
encodeListKnown = findEncodings(images)
cap = cv2.VideoCapture(0)
 
while True:
    success, img = cap.read()
    #img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)
    
    y1,x2,y2,x1 = faceLoc
    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
    top = Tk()
    top.geometry("0x0") 

    def markAttendance(name):
        with open('Attendance.csv','r+') as f:
            myDataList = f.readlines()
            nameList = []
            data = 'Present'
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name in nameList:
                messagebox.showinfo("Message","You are already marked present")  
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString},{data}')
                messagebox.showinfo("Message","Attendance Marked , Thank you")  
    
    
    if faceDis[matchIndex]< 0.50:
        name = classNames[matchIndex].upper()
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
        cv2.imshow('Webcam',img)
        cv2.waitKey(10)
        print("face Detected")
        markAttendance(name)
        break
    else:
        name = 'Unknown'
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        cv2.imshow('Webcam',img)
        key = cv2.waitKey(10)
        if (key == 27):
            break
cap.release()
cv2.destroyAllWindows()

