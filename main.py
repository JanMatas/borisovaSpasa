""" This module loads a picture and identifies tennis court
lines """
import sys
import cv2
import numpy as np
import math


class Point:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def tup(self):
        return (self.x, self.y)
    
    def draw(self, img):
        cv2.circle(img, (self.x, self.y), 3, [255,0,0])

class Line:
    def __init__(self, x1, y1, x2, y2):
        self.A = Point(float(x1), float(y1))
        self.B = Point(float(x2), float(y2))
    
    def equation(self):
        denominator = float(self.B.x - self.A.x)
        slope = (self.B.y - self.A.y) / denominator if denominator else 0.0001
        y_intersect = - int(slope * self.A.x) + self.A.y
        return slope, y_intersect

    def draw_line(self, img):
        slope, y_intersect = self.equation()
        x1 = 0        
        y1 = int(y_intersect)
        x2 = 1000
        y2 = int(slope * x2 + y_intersect)
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)

    def intersection(self, line):
        xdiff = Point(self.A.x - self.B.x, line.A.x - line.B.x)
        ydiff = Point(self.A.y - self.B.y, line.A.y - line.B.y)
        def det(a, b):
            return a.x * b.y - a.y * b.x
        div = det(xdiff, ydiff)
        if div == 0:
            return Point(-1, -1)
        d = Point(det(self.A, self.B), det(line.A, line.B))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return Point(x, y)
        
    def draw_segment(self, img):
        cv2.line(img,self.A.tup(),self.B.tup(),(0,0,255),2)




def find_line_intersections(lines):
    points = []
    for line1 in lines:
        for line2 in lines:
            points.append(line1.intersection(line2))
    return points
            
def process_img(img):
    crop = img[int(len(img)*0.3):len(img),0: len(img[0]) ]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([0,0,180], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(crop,crop, mask= mask)
    
    
    houghLines = cv2.HoughLinesP(mask,rho = 5,theta = 1*np.pi/90,threshold = 100,minLineLength = 100,maxLineGap = 5)
    
    lines = map(lambda l: Line(*l[0]), houghLines if houghLines is not None else [])
    map(lambda l: l.draw_line(crop), lines)
    return crop
    
    points = find_line_intersections(lines)
    print points
    map(lambda p: p.draw(crop), points)
    return crop


def main1():

    fn = sys.argv[1]
    cap = cv2.VideoCapture(fn)
    #fgbg = cv2.createBackgroundSubtractorMOG2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while(cap.isOpened()):
        ret, frame = cap.read()
        #res = process_img(frame)
        fgmask = fgbg.apply(frame)
        opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        cv2.imshow('lines',opening)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main2():
    fn = sys.argv[1]
    img = cv2.imread(fn)
    res = process_img(img)
    cv2.imshow('lines',res)
    #edges = cv2.Canny(smooth,100,150,apertureSize = 3)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main1()
