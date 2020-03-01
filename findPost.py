import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math

#import video
cap = cv.VideoCapture('samplevid.mp4')

def frameResize(frame,width):
    imgScale = width/frame.shape[1]
    newWidth = frame.shape[1]*imgScale
    newHeight = frame.shape[0]*imgScale   
    frame = cv.resize(frame, (int(newWidth), int(newHeight)))
    return frame

def findEq(line,eq):
    x1,y1,x2,y2 = line
    dist = math.sqrt(pow(x1-x2,2)+pow(y1-y2,2))
    if (x2-x1!=0) and dist>300:
        m = (y2-y1)/(x2-x1) 
        c = y1 - (m * x1)
        eq.append((m,c))

def findIntersect(equations,intersects,frame):
    for equation in equations:
        m1,c1 = equation
        for eq in equations:
            m2,c2 = eq    
            if m1!=m2 :
                x = (c2-c1)/(m1-m2)
                y = (m1*x) + c1
                if x>=0 and x<=frame.shape[1] and y>=0 and y<=frame.shape[0]:
                    intersects.append((x,y))

def process(frame):
    #frame resize
    frame = frameResize(frame,1280)
    
    #apply gaussian blur
    blur = cv.GaussianBlur(frame,(5,5),15)
    #colorspace conversion
    hsv = cv.cvtColor(blur,cv.COLOR_BGR2HSV)
    #HSV threshold
    lowerHSV = np.array([20,0,50])
    upperHSV = np.array([255,80,255])
    filterHSV = cv.inRange(hsv,lowerHSV,upperHSV)
     
    #check for lines
    lines = cv.HoughLinesP(filterHSV,1,np.pi/180,100,100)
    #print(lines.shape)
    eq = []
    #draw lines
    for line in lines:
        x1,y1,x2,y2 = line[0]
        findEq(line[0],eq)
        cv.line(frame,(x1,y1),(x2,y2),(255,0,0),2)

    eq = np.array(eq)
    plt.scatter(eq[:,0],eq[:,1])   
    #find intersections
    intersections = []
    findIntersect(eq,intersections,frame)

    for intersection in intersections:
        x,y = int(intersection[0]),int(intersection[1])
        cv.circle(frame,(x,y),3,(0,0,255))

    
    #print(frame.shape)
    #print(" ")
    #print(lines.shape)
    #print(" ")
    #print(lines)
    #print(" ")
    #print(eq)
    #print(" ")
    #print(eq[0])
    #print(" ")
    #print(eq[0][0])
    #print(" ")
    #print(intersections)

    #display
    #cv.imshow("hsv",hsv)
    cv.imshow("HSV Filtered",filterHSV)
    cv.imshow("frame",frame)    
    plt.show()
                

arrFrame = []
i = 0
#while (cap.isOpened()):
ret = True
while (ret):
    #read frame by frame
    ret,frame = cap.read()

    arrFrame.append(frame)
    print(i)
    i+=1

    #press q to stop
    #if cv.waitKey(100) & 0xFF == ord('q'):
    #        break

#release the capture
cap.release()

process(arrFrame[150])
cv.waitKey(0)
cv.destroyAllWindows()