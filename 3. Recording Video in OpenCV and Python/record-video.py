import os
import numpy as np
import cv2

filename = 'video.avi'   #.avi  .mp4
frames_per_second = 24.0   #How many frames to record per second
my_res = '720p'  #Resolution of the recorded file #1080p


# Set resolution for the video capture
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

#Dictionary of resolutions
# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height


# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']



cap = cv2.VideoCapture(0)
dims = get_dims(cap, res = my_res)
video_type_cv2 = get_video_type(filename)

out = cv2.VideoWriter(filename, video_type_cv2, frames_per_second, dims)  #dims = width, height

while(True):
	#Capture frame-by-frame
	ret, frame = cap.read()
	#Converting frame into gray frame
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	out.write(frame)
	#Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

#When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()

