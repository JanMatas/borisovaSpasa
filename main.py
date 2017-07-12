""" This module loads a picture and identifies tennis court
lines """
import sys
import cv2
import numpy as np
import math
from itertools import chain


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

    def draw_line(self, img, color):
        slope, y_intersect = self.equation()
        x1 = 0        
        y1 = int(y_intersect)
        x2 = 1000
        y2 = int(slope * x2 + y_intersect)
        cv2.line(img,(x1,y1),(x2,y2),color,3)

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





 
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    pts = np.array([[40.0, 108.0], [ 604.0, 108.0], [ 472.0, 39.0],[ 100.0, 39.0]],  dtype = "float32")
    (tl, tr, br, bl) = pts
    rect = pts
 
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
 
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (1000, 1000))
 
    # return the warped image
    return warped


def find_line_intersections(lines):
    points = []
    for line1 in lines:
        for line2 in lines:
            points.append(line1.intersection(line2))
    return points
      
def process_img(img):
    crop = img[int(len(img)*0.4):len(img),0: len(img[0]) ]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([0,0,180], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(crop,crop, mask= mask)
    
    
    houghLines = cv2.HoughLinesP(mask,rho = 5,theta = 1*np.pi/90,threshold = 80,minLineLength = 100,maxLineGap = 8)
    
    lines = map(lambda l: Line(*l[0]), houghLines if houghLines is not None else [])
    baselines = filter(lambda l: abs(l.equation()[0]) < 0.05, lines)
    otherlines = filter(lambda l: abs(l.equation()[0]) >= 0.05, lines)
    if not lines:
        return crop
    upper_line = max(baselines, key=lambda line: line.A.y)
    lower_line = min(baselines, key=lambda line: line.A.y)
    upper_line.draw_line(crop, [255, 0, 0])
    lower_line.draw_line(crop, [0,0,255])

    mid = len(img[0]) / 2
    
    upper_points = map(lambda l: upper_line.intersection(l), otherlines)
    lower_points = map(lambda l: lower_line.intersection(l), otherlines)

    map(lambda l: l.draw_line(crop, [0,255,0]), otherlines)
    for point in upper_points:
        if (point.x > mid):
            print(point.x)
    

    lower_left = max(chain([Point(1,3)], filter(lambda p: p.x < mid, lower_points)) , key=lambda p: p.x)
    lower_right = min(chain([Point(1000,1000)], filter(lambda p: p.x >= mid, lower_points)), key=lambda p: p.x)
    
    upper_left = max(chain([Point(1,3)], filter(lambda p: p.x < mid, upper_points)), key=lambda p: p.x)
    upper_right = min(chain([Point(1000,1000)], filter(lambda p: p.x >= mid, upper_points)), key=lambda p: p.x)

    lower_left.draw(crop)
    lower_right.draw(crop)

    upper_left.draw(crop)
    upper_right.draw(crop)
    
    points = [upper_left, upper_right, lower_right, lower_left]
    print(str(upper_right.x) + " : " + str(upper_right.y))

    warped = four_point_transform(crop, np.array(map(lambda p: p.tup(), points),  dtype = "float32"))
    
    
    #points = find_line_intersections(lines)
    #print points
    #map(lambda p: p.draw(crop), points)
    return warped 


def main1():
    fn = sys.argv[1]
    cap = cv2.VideoCapture(fn)
    #fgbg = cv2.createBackgroundSubtractorMOG2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.createBackgroundSubtractorMOG2()

    count = 0
    while(cap.isOpened()):
        count += 1
        ret, frame = cap.read()        
        if count < 300:
            continue
        res = process_img(frame)
        crop = frame[int(len(frame)*0.4):len(frame),0: len(frame[0]) ]
        fgmask = fgbg.apply(crop)
        opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        cv2.imshow('lines',four_point_transform(opening, None))

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
