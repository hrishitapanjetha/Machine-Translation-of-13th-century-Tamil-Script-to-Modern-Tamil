
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import interpolation as inter
import os

def test():
    print('Hello World')

def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated


def preprocess(input_image):
    #Converting the colored image to greyscale
    angle, rotated = correct_skew(input_image)
    # print(angle)
    # cv2.imwrite('rotated.jpg', rotated)
    gray = cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(rotated, [c], -1, (255,255,255), 5)
        # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(rotated, [c], -1, (255,255,255), 5)
    gray = cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)
    #applying median filter for Salt and pepper/impulse noise
    filter1 = cv2.medianBlur(gray,5)
    #applying gaussian blur to smoothen out the image edges
    filter2 = cv2.GaussianBlur(filter1,(5,5),0)
    #applying non-localized means for final Denoising of the image
    dst = cv2.fastNlMeansDenoising(filter2,None,17,9,17)
    #converting the image to binarized form using adaptive thresholding
    th1 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return th1



# image=preprocess('box.jpg')
# # displaying the preprocessed image
# plt.axis('off')
# plt.imshow(image, cmap='gray')


# image = preprocess('box.jpg')
# def segment(image):
#     # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(image, (7, 7), 0)
# segment(image)


# segement will take the preprocessed image as input and return the segmented image

def segment(image, dirname):
    gray = cv2.GaussianBlur(image, (7, 7), 0)
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
    dilate = cv2.dilate(thresh1, None, iterations=2)
    cnts,_ = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * image.shape[1] )
    orig = cv2.merge([image,image,image])
    i = 0
    # image needs to be saved in ./dataset/{dirname}/segemented/ folder 
    save_path = f"./dataset/{dirname}/segmented/"
    os.makedirs(save_path, exist_ok=True)
    for cnt in sorted_ctrs:
        # Check the area of contour, if it is very small ignore it
        if(cv2.contourArea(cnt) < 200):
            continue

        # Filtered countours are detected
        x,y,w,h = cv2.boundingRect(cnt)
        
        # Taking ROI of the cotour
        roi = image[y:y+h, x:x+w]
        
        # Mark them on the image if you want
        cv2.rectangle(orig,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imwrite(f"{save_path}{i}.png", roi)

        i = i + 1 
    return orig



# segmented_image=segment(preprocess('21.jpg'))
# # save the segmented image
# cv2.imwrite('segmented.jpg', segmented_image)





