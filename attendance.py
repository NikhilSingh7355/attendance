import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime


path = r"C:\Users\NIKHIL SINGH\OneDrive\Attachments\Desktop\student_images"
images = []
studentNames = []
myList = os.listdir(path)
print("Found students:", myList)

for file in myList:
    curImg = cv2.imread(f"{path}/{file}")
    if curImg is None:
        print(f"Warning: {file} cannot be read, skipping...")
        continue
    images.append(curImg)
    studentNames.append(os.path.splitext(file)[0])  


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
        else:
            print("Warning: No face found in image, skipping...")
    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding Complete ✅")


def markAttendance(name):
    with open("Attendance.csv", "a+") as f:  
        f.seek(0)
        data = f.readlines()
        nameList = [line.strip().split(",")[0] for line in data]

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f"{name},{dtString}\n")
            print(f"Attendance marked for {name} at {dtString}")


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame from webcam")
        break

    
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        if len(faceDis) > 0:
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = studentNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                markAttendance(name)

    cv2.imshow("Webcam Attendance", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()
