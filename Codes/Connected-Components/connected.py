import cv2
import numpy as np
import random

#########################################################
#       FUNCTION TO FIND THE CONNECTED COMPONENTS
#########################################################

def drawComponents(image, adj, block_size):
    
    #ret, labels = cv2.connectedComponents(image)

    #print(ret)
    #print(labels)
    #cv2.imshow('test1', labels.astype(np.uint8))
    
    image = image.astype('uint8')
    #print (image.shape)

    block_w = block_size
    block_h = block_size
    
    nb = 0
    comp = []
    
    for r in range(0, image.shape[0] - block_w, block_h):
        for c in range(0, image.shape[1] - block_w, block_h):
            window = image[r:r+block_w, c:c+block_h]
            x = list(cv2.connectedComponents(window, adj))
            nb += x[0]
            x[1] = x[1] * random.randint(1, 16) * random.randint(1, 16)
            comp.append(x[1])
    
    bc = image.shape[0]//block_size
    br = image.shape[1]//block_size
    
    img = np.zeros(image.shape)
    #print (img.shape)
    '''
    for r in range(0, img.shape[0] - block_w, block_h):
        for c in range(0, img.shape[1] - block_w, block_h):
            for i in range(len(comp)):
                img[r:r+block_w, c:c+block_h] = comp[i]*255
    
    for k in range(len(comp)):
        for i in range(block_size):
            for j in range(block_size):
                if k%br == 0 and k!=0:
                    c = (((k+1)*block_size)//img.shape[1])*block_size + j
                else:
                    c = ((k*block_size)//img.shape[1])*block_size + j

                r = (k*block_size + i) % (br*block_size)
                
                img[c][r] = comp[k][j][i]
    '''
    #cv2.imshow('Test Image', img)
    
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, adj)
    label_hue = (107*output%np.max(output)).astype(np.uint8)
    
    #label_hue = (107*img%np.max(img)).astype(np.uint8)
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2RGB)

    labeled_img[label_hue==0] = 0
    
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
    block = int(input("Please enter block size (m X m): "))
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

img_orig = cv2.imread('../../Images/2.jpg')

cv2.imshow('Original', img_orig)

#im = cv2.UMat(Image.fromarray(img_orig).convert("L"))
#Image.fromarray(img_orig)

bw = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)
#cv2.imshow("BW", bw)
#cv2.imwrite("./Outputs/Grayscale.jpg", bw)

x, img = cv2.threshold(bw, thresh[0], thresh[1], cv2.THRESH_BINARY)  #ensuring binary
img[img==x] = 255
cv2.imshow("Binary", img)
#cv2.imwrite("./Outputs/Binary Image {V=("+str(thresh[0])+", "+str(thresh[1])+"), adj="+str(adj)+"}.jpg", img)

img2 = drawComponents(img, adj, block)   # calling implementation function
#print(img2.shape)
cv2.imshow('Connected Components', img2)
#cv2.imwrite("./Outputs/Paths{V=("+str(thresh[0])+", "+str(thresh[1])+"), adj="+str(adj)+"}.jpg", img2)

#########################################################
#                   PRINTING OUTPUT
#########################################################

#img3 = bw * (img2.reshape(img2.shape[0],img2.shape[1]))

# Using the hues from img2 and the saturation and luminosity from the original image to get proper results.

cvt = cv2.cvtColor(img_orig, cv2.COLOR_RGB2HSV)
img4 = np.zeros(cvt.shape)
img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2HSV)

for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        img4[i][j][0] = (img2[i][j][0]*4 + cvt[i][j][1]*1)//5       # HUE
        img4[i][j][1] = (img2[i][j][1]*3 + cvt[i][j][1]*7)//10      # SATURATION
        img4[i][j][2] = cvt[i][j][2]                                # LIGHT VALUE
        
        if img2[i][j][0] == 0:
            img4[i][j] = 0

img4 = cv2.cvtColor(img4.astype(np.uint8), cv2.COLOR_HSV2RGB)

#img3 = bw + (img2.reshape(img2.shape[0],img2.shape[1]))
#img4 =  [[[i, i, i] for i in j] for j in img2]
#img5 = img_orig * img4
cv2.imshow('Result', img4.astype(np.uint8))
#cv2.imwrite("./Outputs/Result{V=("+str(thresh[0])+", "+str(thresh[1])+"), adj="+str(adj)+"}.jpg", img4.astype(np.uint8))
print ("Job done!")

cv2.waitKey(0)
cv2.destroyAllWindows()
