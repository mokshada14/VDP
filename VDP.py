import numpy as np
import cv2

#blue colour range.
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])
kernel = np.ones((5, 5), np.uint8)
#intialising the points.
x1,y1=0,0

paintWindow=None
camera = cv2.VideoCapture(0)  



while True:
    ret, frame= camera.read()
    frame = cv2.flip( frame, 1 )

    if paintWindow is None:
        #blank window.
        paintWindow = np.ones((512,512,3), np.uint8)
        paintWindow.fill(255)
        cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)
    #create a blue mask.    
    hsv= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)                   
  
    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)
    
    #find contours in the blue region
    cnts,_= cv2.findContours(blueMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:

        cnt =max(cnts, key = cv2.contourArea)
        (x2,y2),radius= cv2.minEnclosingCircle(cnt)
        radius=int(radius)
        M = cv2.moments(cnt)
        center = (int(x2),int(y2))
        if x1 == 0 and y1 == 0:
            #starting point
            x1,y1=x2,y2
    
        else:
            #draw a circle showing the max blue region.
            cv2.circle(frame,center,radius,(0, 255,255),1)
            #to start drawing
            cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)), [0,0,255],5)
            cv2.line(paintWindow,(int(x1),int(y1)),(int(x2),int(y2)), [0,0,0],5)
        x1,y1= x2,y2
    else:
        #to start from initial(0,0) point.
        x1,y1=0,0

    cv2.imshow('Tracking', frame)
    cv2.imshow('Paint', paintWindow)
    k=cv2.waitKey(1)&0xFF
    if k==ord('c'):
        #to clear screen.
        paintWindow=None
        
    
    if k== ord('q'):
        break

camera.release()
cv2.destroyAllWindows()