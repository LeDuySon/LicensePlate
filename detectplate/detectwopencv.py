import cv2
import numpy as np
import imutils
import os
import pytesseract
image = cv2.imread("car_long/car_long/41224.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
mask = np.zeros((gray.shape),np.uint8)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
# div = np.float32(gray)/(close)
# res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
# res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)


def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	return edged
def get_four_points(image):
    contour_point = []
    cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) > 0:
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
        for contour in cnts:
            length = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, length*0.05, closed = True)
            if len(approx) == 4:
                contour_point.append(approx)
                # return approx
    return contour_point

def reordering(points):

    rects = np.zeros((4, 2), dtype="float32")
    print(np.argmin(np.sum(points, axis = 1)))
    print(points.shape)
    rects[0] = points[np.argmin(np.sum(points, axis = 2))]
    rects[2] = points[np.argmax(np.sum(points, axis = 2))]
    rects[1] = points[np.argmin(np.diff(points, axis = 2))]
    rects[3] = points[np.argmax(np.diff(points, axis = 2))]

    return rects 

def transform(image, pts):
    rect = reordering(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
# edges = auto_canny(gray)
threshold = cv2.threshold(gray, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
# thresh2 = cv2.adaptiveThreshold(res2.copy(),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)
coor = []
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel2)
contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
for c in contours:
    length = cv2.arcLength(c, closed = True)
    approx = cv2.approxPolyDP(c, 0.03*length, closed = True)
    if approx.shape[0] == 4:
        x, y, w, h = cv2.boundingRect(c)
        if(100 > w > 50 and w/h > 3):
            coor.append(approx)
            cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), thickness= 2)
            cv2.imshow("haha", image)
            cv2.waitKey(0)
print(coor)

plate = transform(image, coor[0])
grayp = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
grayp = cv2.GaussianBlur(grayp, (5, 5), 0)
threshp = cv2.threshold(grayp, 0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
# kernelp = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
# open_t = cv2.morphologyEx(threshp, cv2.MORPH_OPEN, kernelp)
n_threshp = cv2.erode(threshp, (3, 3), iterations=2)
n_threshp = cv2.dilate(n_threshp, (3, 3), iterations = 2)
config = ("-l eng --oem 1 --psm 7")
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
text = pytesseract.image_to_string(plate, config=config)
print(text)
cv2.putText(image, text, (50, 100) , cv2.FONT_HERSHEY_SIMPLEX , 1,  
                 (0, 0, 255), 2, cv2.LINE_AA, False)
# c_contours = cv2.findContours(threshp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# c_contours = imutils.grab_contours(c_contours)
# for co in c_contours:
#     x, y, w, h = cv2.boundingRect(co)
#     if(h > plate.shape[0]/2):
        
#         cv2.rectangle(plate, (x, y), (x+w, y+h), (255, 0, 0), thickness= 2)
#         cv2.imshow("character", plate)
#         cv2.waitKey(0)
# cv2.imshow("a", n_threshp)
# cv2.imshow("b", image)

# cv2.waitKey(0)

def convert_name(image):
    print(image)
    image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = cv2.threshold(gray, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    coor = []
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel2)
    contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    for c in contours:
        length = cv2.arcLength(c, closed = True)
        approx = cv2.approxPolyDP(c, 0.03*length, closed = True)
        if approx.shape[0] == 4:
            x, y, w, h = cv2.boundingRect(c)
            if(100 > w > 50 and w/h > 3):
                coor.append(approx)
                cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), thickness= 2)
                cv2.waitKey(0)
    print(coor)
    if(len(coor) > 0):
        plate = transform(image, coor[0])
    else:
        return
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text = pytesseract.image_to_string(plate, config=config)
    cv2.putText(image, text, (50, 100) , cv2.FONT_HERSHEY_SIMPLEX , 1,  
                 (0, 0, 255), 2, cv2.LINE_AA, False)
    cv2.imshow("b", image)
    cv2.waitKey(0)

link_image = os.listdir("car_long\car_long")

for img in link_image[:100]:
    if(img.endswith(".jpg")):
        n_img = os.path.join("car_long\car_long", img)
        print("Process image: " + img)
        convert_name(n_img)


    