import face_recognition
import cv2
import numpy as np

#uploade picture and convert BGR to RGB
imgNabil = face_recognition.load_image_file('imagestest/nabil.jpg')
imgNabil =cv2.cvtColor(imgNabil,cv2.COLOR_BGR2RGB)

#uploade picture test and convert BGR to RGB
imgNabtest = face_recognition.load_image_file('imagestest/nabiltest.jpg')
imgNabtest =cv2.cvtColor(imgNabtest,cv2.COLOR_BGR2RGB)

#face location
faceLoc = face_recognition.face_locations(imgNabil)[0]
encodeNabil = face_recognition.face_encodings(imgNabil)[0]
cv2.rectangle(imgNabil, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgNabtest)[0]
encodeNabtest = face_recognition.face_encodings(imgNabtest)[0]
cv2.rectangle(imgNabtest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

#compare two picture
results = face_recognition.compare_faces([encodeNabil],encodeNabtest)
faceDis = face_recognition.face_distance([encodeNabil],encodeNabtest)
print(results,faceDis)
cv2.putText(imgNabtest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('mohamed nabil',imgNabil)
cv2.imshow('mohamed nabil test',imgNabtest)

cv2.waitKey(0)