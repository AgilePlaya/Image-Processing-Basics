import cv2
import numpy as np

#########################################################
#       FUNCTION TO FIND THE CONNECTED COMPONENTS
#########################################################

def drawComponents(image, adj):
    
    ret, labels = cv2.connectedComponents(image)

    #print(ret)
    #print(labels)
    #cv2.imshow('test1', labels.astype(np.uint8))
    
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, adj)
    sizes = stats[:, -1]
    
    #print(sizes)
    #print(stats)
    #print(output)
    #cv2.imshow('test2', output.astype(np.uint8))
    '''
    for i in range(0, len(output), 50):
        print(output[i])
        cv2.imshow('test'+str(i), output[i].astype(np.uint8))
    '''
    '''
    max_label = 1
    min_label = 0
    min_size = sizes[1]
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
        if sizes[i] < min_size:
            min_label = i
            min_size = sizes[i]
    '''
    
    label_hue = (107*output/np.max(output)).astype(np.uint8)
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2RGB)

    labeled_img[label_hue==0] = 0
    #cv2.imshow('Labeled Image', labeled_img)
    '''
    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    img2 = img2 + output
    '''
    return labeled_img

#########################################################
#                       INPUTS
#########################################################

flag = 15
while flag != 0:
    if flag == 1 or flag == 15:
        adj = int(input("Enter the adjacency for detection (4 or 8): "))
    if flag == 2 or flag == 15:
        thresh = list(map(int, input("Enter the range of threshold separated by space(Example: 150 200): ").split(" ")))
    if adj != 4 and adj != 8:
        flag = 1
        print("Inoperable value for adjacency. Please enter 4 or 8")
        continue
    elif len(thresh) != 2:
        print("Please input exactly 2 numbers in the given format.")
        flag = 2
        continue
    elif thresh[0] > thresh [1]:
        thresh[0], thresh[1] = thresh[1], thresh[0]
    else:
        flag = 0
    if thresh[0] < 0 or thresh[1] > 255:
        print("Values are beyond limits. Please enter values between 0 and 255")
        flag = 2

#########################################################
#                   READING IMAGE
#########################################################

img_orig = cv2.imread('../2.jpg')

cv2.imshow('Original', img_orig)

#im = cv2.UMat(Image.fromarray(img_orig).convert("L"))
#Image.fromarray(img_orig)

bw = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)
cv2.imshow("BW", bw)
#cv2.imwrite("Grayscale.jpg", bw)

x, img = cv2.threshold(bw, thresh[0], thresh[1], cv2.THRESH_BINARY)  #ensuring binary
img[img==x] = 255
cv2.imshow("Binary", img)
cv2.imwrite("Binary Image {V=("+str(thresh[0])+", "+str(thresh[1])+"), adj="+str(adj)+"}.jpg", img)

img2 = drawComponents(img, adj)   # calling implementation function
#print(img2.shape)
cv2.imshow('Connected Components', img2)
cv2.imwrite("Paths{V=("+str(thresh[0])+", "+str(thresh[1])+"), adj="+str(adj)+"}.jpg", img2)

#########################################################
#                   PRINTING OUTPUT
#########################################################

#img3 = bw * (img2.reshape(img2.shape[0],img2.shape[1]))

# Using the hues from img2 and the saturation and luminosity from the
# original image to get proper results.

cvt = cv2.cvtColor(img_orig, cv2.COLOR_RGB2HSV)
img4 = np.zeros(cvt.shape)
img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2HSV)

for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        img4[i][j][0], img4[i][j][1], img4[i][j][2] = img2[i][j][0], (img2[i][j][1]*2 + cvt[i][j][1]*3)//5, cvt[i][j][2]
        if img2[i][j][0] == 0:
            img4[i][j] = 0

img4 = cv2.cvtColor(img4.astype(np.uint8), cv2.COLOR_HSV2RGB)

#img3 = bw + (img2.reshape(img2.shape[0],img2.shape[1]))
#img4 =  [[[i, i, i] for i in j] for j in img2]
#img5 = img_orig * img4
cv2.imshow('Result', img4.astype(np.uint8))
cv2.imwrite("Result{V=("+str(thresh[0])+", "+str(thresh[1])+"), adj="+str(adj)+"}.jpg", img4.astype(np.uint8))
print ("Job done!")

cv2.waitKey(0)
cv2.destroyAllWindows()
