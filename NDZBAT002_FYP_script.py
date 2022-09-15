import numpy as np
import cv2 as cv
print("Start")
image1 = cv.imread("images/video01_pic/2.jpg")              #reads image
cv.imshow("original",image1)                                                               #shows image

def rescaleFrame(frame,scale = 0.5):   
    width = int(frame.shape[1]*scale)    #1 is for width
    height =int(frame.shape[0]*scale)    #0 is for height
    dimensions = (width,height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA) # resizes the frame to the dimensions in all the area
 
resized_image = rescaleFrame(image1)   
cv.imshow("resized_pic",resized_image)

#Cropping images
#cropped = resized_image[300:500 , 100:500]
#cv.imshow("cropped", cropped)


cv.waitKey(0)   #delay

#video breakdown into frames
pick_video_location = "video01"   #04 is the mx
pick_video = pick_video_location + ".mp4"
add1  = "FYP_DATA/"+pick_video

capture = cv.VideoCapture(add1 )
i=0
while True:
    isTrure, frame = capture.read()
    resizedVideo = rescaleFrame(frame)
    #cv.imshow("Video1", frame)                               #original big video
    #cv.imshow("resized_video",resizedVideo)                   #resized video
    ##images\video01_pic
    
    add2 = "images/"+pick_video_location+ "_pic/%d.jpg"
    cv.imwrite(add2%i, frame)
    i =i+1
    if i > 30:                                             # stops after 30 seconds
        break
    if cv.waitKey(20) & 0xFF==ord("d"):
        break
    
capture.release()   
cv.destroyAllWindows()
cv.waitKey(0)   #delay   

#Contour Detection

correct = "images/"+pick_video_location+ "_pic/"+"5.jpg"
#print("correct =",correct)
image2 = cv.imread(correct)              #reads image

gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
#image1 = cv.imread("images/video01_pic/2.jpg")  
print(gray[50][0:4])
cv.imshow("problem",gray)
cv.waitKey(0)   #delay

#Edge cascade
all_edge = cv.Canny(gray, 10,200)
cv.imshow("me",all_edge)
cv.waitKey()

contours, hierarchies = cv.findContours(gray, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(f'{len(contours)} contour(s) found!')    

cv.drawContours(gray, contours, -1, (0,255,0), 1)
cv.imshow('Contours Drawn', gray)  
cv.waitKey(0)
print("out of the loop")    
