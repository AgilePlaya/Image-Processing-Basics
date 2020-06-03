import cv2
import numpy

image = cv2.imread('./canny.jpeg', 0)
cv2.imshow("Input",image)

mi = 255
ma = 0

new = numpy.zeros((len(image),len(image[0]),1),dtype="uint8")
#cv2.imshow("Black",new)

for i in range(len(image)):
    for j in range(len(image[0])):
        if image[i][j] > ma:
            ma = image[i][j]
        if image[i][j] < mi:
            mi = image[i][j]
        new[i][j] = image[i][j]
#cv2.imshow("Copy",new)

for i in range(len(image)):
    for j in range(len(image[i])):
        #new[i][j] = int((image[i][j]-mi)/(ma-mi) % 256)
        if image[i][j]<128:
            new[i][j] = 0
        else :
            new[i][j] = 255

cv2.imshow("Normalize 0 to 1",new)

for i in range(len(image)):
    for j in range(len(image[i])):
        new[i][j] = int((image[i][j]-mi)*255/(ma-mi) % 256)

cv2.imshow("Normalize 0 to 255",new)
cv2.imwrite("canny-normalised.jpg", new)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
