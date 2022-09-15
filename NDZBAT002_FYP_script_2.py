import numpy as np
import cv2 as cv
print("Start")

#read video and writes frames to specified folders
#video breakdown into frames
pick_video_location = "video01"   #04 is the mx
pick_video = pick_video_location + ".mp4"
add1  = "FYP_DATA/"+pick_video

capture = cv.VideoCapture(add1 )
i=0
while True:
    isTrure, frame = capture.read()
                                                                                  #resizedVideo = rescaleFrame(frame)
    #cv.imshow("Video1", frame)                               #original big video
                                                            #cv.imshow("resized_video",resizedVideo)                   #resized video
    
    add2 = "images/"+pick_video_location+ "_pic/%d.jpg"   # location where the new images will be saved
    cv.imwrite(add2%i, frame)
    i =i+1
    if i > 30:                                             # stops after i frames or i/30 seconds
        break
    if cv.waitKey(2) & 0xFF==ord("d"):                     
        break
    
capture.release()   
cv.destroyAllWindows()
cv.waitKey(0)   #delay

#read image

image_path= "images/"+pick_video_location+ "_pic/"+"5.jpg"
image1 = cv.imread(image_path)              #reads image
#cv.imshow("original",image1)                                                               #shows image
cv.waitKey(0)   #delay 

def rescaleFrame(frame,scale = 0.5):   
    width = int(frame.shape[1]*scale)    #1 is for width
    height =int(frame.shape[0]*scale)    #0 is for height
    dimensions = (width,height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA) # resizes the frame to the dimensions in all the area

resized_image = rescaleFrame(image1)   #resizing image
#cv.imshow("resized_pic",resized_image)  #showing resized image

# perspective transform

# Pixel values in original image                 
red_point = [350,120]
green_point = [600,120]
black_point = [100,550]
blue_point = [800,550]
 
# Create point matrix
point_matrix = np.float32([red_point,green_point,black_point, blue_point])
 
# Draw circle for each point                     
cv.circle(image1,(red_point[0],red_point[1]),1,(0,0,255),cv.FILLED)
cv.circle(image1,(green_point[0],green_point[1]),1,(0,255,0),cv.FILLED)
cv.circle(image1,(blue_point[0],blue_point[1]),1,(255,0,0),cv.FILLED)
cv.circle(image1,(black_point[0],black_point[1]),1,(0,0,0),cv.FILLED)
 
# Output image size
width, height = 500,600
 
# Desired points value in output images    
converted_red_pixel_value = [0,0]
converted_green_pixel_value = [width,0]
converted_black_pixel_value = [0,height]
converted_blue_pixel_value = [width,height]
 
# Convert points
converted_points = np.float32([converted_red_pixel_value,converted_green_pixel_value,
                               converted_black_pixel_value,converted_blue_pixel_value])
 
# perspective transform
perspective_transform = cv.getPerspectiveTransform(point_matrix,converted_points)
image1_trans = cv.warpPerspective(image1,perspective_transform,(width,height))
 
#cv.imshow("Original Image", image1)
#cv.imshow("trasformed Image", image1_trans)
cv.waitKey(0)

imgray = cv.cvtColor(image1_trans, cv.COLOR_BGR2GRAY)               # changing from colour BGR to Gray
ret, thresh = cv.threshold(imgray, 145, 150, 0)                      #detect only the specified range

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
print("Number of contours = " + str(len(contours)))
#print(contours[0])
cv.imshow('ori_colour with out contours',image1_trans)
cv.drawContours(image1_trans, contours, -1, (0, 255, 0), 1)
cv.drawContours(imgray, contours, -1, (50, 100, 255), 1)
cv.imshow('Image GRAY with contours', imgray)
cv.imshow("original image with contours",image1_trans)


cv.waitKey(0)   #delay 
print("Done")

