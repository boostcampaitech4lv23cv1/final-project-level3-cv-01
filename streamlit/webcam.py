import cv2
# 웹캡 영상 그대로 저장

#Load Web Camera
cap = cv2.VideoCapture(0) #load WebCamera
if not (cap.isOpened()):
	print("File isn't opend!!")

#Set Video File Property
videoFileName = 'webcam.MP4'
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'avc1')
delay = round(1000/fps)

#Save Video
out = cv2.VideoWriter(videoFileName, fourcc, fps, (w,h))
if not (out.isOpened()):
	print("File isn't opend!!")
	cap.release()
	sys.exit()

#Load frame and Save it
while(True): #Check Video is Available
	ret, frame = cap.read() #read by frame (ret=TRUE/FALSE)s

	if ret:
		inversed = cv2.flip(frame, 1) #inversed frame

		out.write(inversed) #save video frame
		
		cv2.imshow('Original VIDEO', frame)
		cv2.imshow('Inversed VIDEO', inversed)

		if cv2.waitKey(delay) == 27: #wait 10ms until user input 'esc'
			break
	else:
		print("ret is false")
		break

cap.release() #release memory
out.release() #release memory
cv2.destroyAllWindows() #destroy All Window