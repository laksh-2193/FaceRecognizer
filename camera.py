import cv2
import numpy as np
import face_recognition
import os
from flask import Flask, render_template, Response
path='Images'
images=[]
classnames = []
name=""

mylist=os.listdir(path)
print(mylist)

for cl in mylist:
    curImg=cv2.imread(f'{path}/{cl}')
    
    images.append(curImg)
    classnames.append(os.path.splitext(cl)[0])

print(classnames)
print("Images - ",images)
def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodelistknown=findEncodings(images)
print("Encodings Completed !!  - ",len(encodelistknown))
print(encodelistknown)


faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret,frame=self.video.read()
        respose, color_img = self.video.read()
        rgb_frame=frame[:,:,::1]
    
        face_locations=face_recognition.face_locations(rgb_frame)
        face_encodings=face_recognition.face_encodings(rgb_frame,face_locations)
    
    
    
    
    
    
        for(top,right,bottom,left), face_encodings in zip(face_locations,face_encodings):
            x=top
            y=right
            w=bottom
            h=left
            x1,y1=x+w,y+w
            cv2.rectangle(color_img, (x, y), (x + h, y + w), (0, 0, 255), 1)
            cv2.line(color_img, (x, y), (x + 30, y), (255, 0, 255), 12)
            cv2.line(color_img, (x, y), (x, y+30), (255, 0, 255), 12)
            
            cv2.line(color_img, (x1, y), (x1-30, y), (255, 0, 255), 12)
            cv2.line(color_img, (x1, y), (x1, y+30), (255, 0, 255), 12)
            
            cv2.line(color_img, (x, y1), (x + 30, y1), (255, 0, 255), 12)
            cv2.line(color_img, (x, y1), (x, y1-30), (255, 0, 255), 12)
            
            cv2.line(color_img, (x1, y1), (x1 - 30, y1), (255, 0, 255), 12)
            cv2.line(color_img, (x1, y1), (x1, y1-30), (255, 0, 255), 12)
            matches = face_recognition.compare_faces(encodelistknown, face_encodings)
            distan=face_recognition.face_distance(encodelistknown, face_encodings)
            print("Distance = ",distan)
            print("Matching cases = ",matches)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = str(int(round(distan[first_match_index]*100)))+"% "+classnames[first_match_index]
                
            cv2.rectangle(frame,(left,bottom-35),(right,bottom),(0,0,255),cv2.FILLED)
            font=cv2.FONT_HERSHEY_DUPLEX
            print(name)
            
            cv2.putText(frame,name,(left+6,bottom-6),font,1.0,(0,255,255),2)
            
    
        
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()