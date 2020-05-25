from collections import namedtuple  
import matplotlib.pyplot as plt  
import random
import numpy
import cv2

Point = namedtuple('Point', 'x y')


class ConvexHull(object):  
    _points = []
    _hull_points = []

    def __init__(self, img):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] == 255:
                    self._points.append(Point(j, i))
        #cv2.imshow("Image", img)
        pass

    def add(self, point):
        self._points.append(point)

    def _get_orientation(self, origin, p1, p2):
        '''
        Returns the orientation of the Point p1 with regards to Point p2 using origin.
        Negative if p1 is clockwise of p2.
        :param p1:
        :param p2:
        :return: integer
        '''
        difference = (
            ((p2.x - origin.x) * (p1.y - origin.y))
            - ((p1.x - origin.x) * (p2.y - origin.y))
        )

        return difference

    def compute_hull(self):
        '''
        Computes the points that make up the convex hull.
        :return:
        '''
        points = self._points

        # get leftmost point
        start = points[0]
        min_x = start.x
        for p in points[1:]:
            if p.x < min_x:
                min_x = p.x
                start = p

        point = start
        self._hull_points.append(start)

        far_point = None
        while far_point is not start:
            # get the first point (initial max) to use to compare with others
            p1 = None
            for p in points:
                if p is point:
                    continue
                else:
                    p1 = p
                    break

            far_point = p1

            for p2 in points:
                # ensure we aren't comparing to self or pivot point
                if p2 is point or p2 is p1:
                    continue
                else:
                    direction = self._get_orientation(point, far_point, p2)
                    if direction > 0:
                        far_point = p2

            self._hull_points.append(far_point)
            point = far_point

    def get_hull_points(self):
        if self._points and not self._hull_points:
            self.compute_hull()

        return self._hull_points

    def display(self, shape):
        # all points
        x = [p.x for p in self._points]
        y = [p.y for p in self._points]
        plt.plot(x, y, marker=',', markeredgewidth=1,
                 markerfacecolor='c', linestyle='None')

        # hull points
        hx = [p.x for p in self._hull_points]
        hy = [p.y for p in self._hull_points]
        plt.plot(hx, hy, marker='.', linestyle='--', markeredgewidth=1,
                 markerfacecolor='g', linewidth=2)

        plt.ylim(ymin=0, ymax=shape[0])
        plt.xlim(xmin=0, xmax=shape[1])
        plt.title('Convex Hull')
        plt.show()

def image_acquisition():
    #image = numpy.zeros((400,500,1), dtype = "uint8")
    image = cv2.imread('../1.jpg', 0)
    #cv2.imshow("Input", image)
    
    img = numpy.zeros((image.shape[0],image.shape[1],1), dtype = "uint8")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == 0:
                img[i][j] = 0
            else:
                img[i][j] = 255
    cv2.imshow("Opened Image", img)
    
    #c1 = (250, 200)
    #col1 = (255)
    #new = cv2.circle(image,c1, 125, col1, -1 )
    #cv2.imshow("Circle 1", new)

    #c2 = (250, 245)
    #col2 = (0)
    #new2 = cv2.circle(new,c2, 50, col2, -1 )

    invert = numpy.zeros((image.shape[0],image.shape[1],1), dtype = "uint8")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if img[i][j] == 0:
                invert[i][j] = 255
    #cv2.imshow("Inverted Image", invert)
    
    inner = numpy.zeros((image.shape[0],image.shape[1],1), dtype = "uint8")
    for i in range(1,len(image)-1):
        for j in range(1,len(image[0])-1):
            if invert[i][j] == 255:
                if invert[i][j]!=invert[i-1][j] or invert[i][j]!=invert[i][j-1] or invert[i][j]!=invert[i+1][j] or invert[i][j]!=invert[i][j+1]:
                    inner[i][j] = 255
    cv2.imshow("Converted to edge points", inner)
    return inner


img = image_acquisition()
cv2.imwrite("Img2Points.jpg", img)
ch = ConvexHull(img)

#for _ in range(50):
#    ch.add(Point(random.randint(-100, 100), random.randint(-100, 100)))
#ch.display()

print("Points on hull:", ch.get_hull_points())
ch.display(img.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()
