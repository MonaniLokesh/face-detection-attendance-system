from cv2 import VideoCapture
import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
# from soupsieve import match

# from tblib import Frame

video_capture = cv2.VideoCapture(0)

kavish_image = face_recognition.load_image_file("photos/kavish.jpg")
kavish_encoding = face_recognition.face_encodings(kavish_image)[0]

harsh_image = face_recognition.load_image_file("photos/harsh.jpg")
harsh_encoding = face_recognition.face_encodings(harsh_image)[0]

lakshika_image = face_recognition.load_image_file("photos/lakshika.jpg")
lakshika_encoding = face_recognition.face_encodings(lakshika_image)[0]

neeraj_image = face_recognition.load_image_file("photos/neeraj.jpg")
neeraj_encoding = face_recognition.face_encodings(neeraj_image)[0]

lokesh_image = face_recognition.load_image_file("photos/lokesh.jpg")
lokesh_encoding = face_recognition.face_encodings(lokesh_image)[0]

known_face_encoding = [
    kavish_encoding,
    lakshika_encoding,
    harsh_encoding,
    neeraj_encoding,
    lokesh_encoding
]

known_faces_names = [
    "harsh",
    "lakshika",
    "kavish",
    "neeraj",
    "lokesh"
] 

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'csv','w+',newline='')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])

    cv2.imshow("attendace system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()

