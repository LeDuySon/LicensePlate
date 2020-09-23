import cv2
import numpy 
import os

PATH = r"C:\Users\admin\Desktop\licenseplate\data\train_data"

images = os.listdir(PATH)

def convert(size, coor):
    w_img, h_img = size
    x, y, w, h = coor
    
for image in images:
    name = image.split(".")[0]
    coord = []
    path2img = os.path.join(PATH, name + ".txt")
    with open(path2img , "rt") as f:
        text = f.readlines()[0].strip()
        coord = text.split(" ")[1: ]
        coord = list(map(float, coord))
    if "txt" not in image:
        img = cv2.imread(os.path.join(PATH, image))
    try:
        w_img, h_img = img.shape[:2]
    except:
        print("Cannot load:", image) 
    coord[0] = coord[0]-coord[2]//2
    coord[1] -= coord[3]//2
    for i in range(len(coord)):
        if(i % 2 ==0):
            coord[i] = int(w_img*coord[i])
        else:
            coord[i] = int(h_img*coord[i])
        print(coord[i])
    x, y, w, h = coord
    print(x, y, w, h)
    print(w_img, h_img)
    cv2.rectangle(img, (y, x), (y+w, x+h), color = (0, 255, 0), thickness = 2)
    cv2.imshow("crop", img)
    cv2.waitKey(0)