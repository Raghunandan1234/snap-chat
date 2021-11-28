from cv2 import cv2
import cvzone
c=cv2.VideoCapture(0)
cas=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
o=cv2.imread('sunglass.png',cv2.IMREAD_UNCHANGED)
while True:
    _,frame=c.read()
    g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    f=cas.detectMultiScale(g)
    for(x,y,w,h) in f:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        o_r=cv2.resize(o,(w,h))
        frame=cvzone.overlayPNG(frame,o_r,[x,y])
    cv2.imshow('dfs',frame)
    if cv2.waitKey(10)==ord('q'):
        break
