import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')



recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
	#Capture frame-by-frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
	for (x,y,w,h) in faces:
		#print(x,y,w,h)
		roi_gray = gray[y:y+h, x:x+w]   #region of interest for the gray frame = gray[Location]
		roi_color = frame[y:y+h, x:x+w] #roi for colored frame 
		

		id_,conf = recognizer.predict(roi_gray)
		if conf>=45 and conf<=85:
			print(id_)
			print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

		#We have the region of interest(rgi). How to recognize them, the frame?
		#Alternative: Deep Learned Model to predict things (Tensorflow, keras, pytorch, scikit-learn): COMPLICATED
		#We'll make our own file : faces-train.py
		img_item = 'my-image.png'    #Save the image
		cv2.imwrite(img_item, roi_gray)

		img_item = '1.png'    #Save the image
		cv2.imwrite(img_item, roi_color)


		#Drawing rectangle around the face
		color = (255, 0, 0)    #Not RGB but it is BGR(0-255)  ##gives blue colored rectangle
		stroke = 2 #Thickness of line
		end_cord_x = x+w
		end_cord_y = y+h
		cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y),color, stroke)  #Grab the frame: Draw on the original colored frame, specify the starting coordinates: x and y, specify the ending coordinates, specify the color and then the stroke 

		#To detect eyes
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

		#To detect smile
		subitems = eye_cascade.detectMultiScale(roi_gray)
		for (sx,sy,sw,sh) in subitems:
			cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)		



	#Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

