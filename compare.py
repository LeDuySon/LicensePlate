import os
import shutil 

path1 = "data/train_data"
path2 = "LicensePlateDetection-master\LicensePlateDetection-master\dataset"

link1 = os.listdir(path1)
# link2 = os.listdir(path2)

import time
file = "data/train_data/" + link1[0]
print(time.ctime(os.path.getmtime(file)).split(" "))

# for i in link1:
#     f = "data/train_data/" + i
#     m_time = time.ctime(os.path.getmtime(f)).split(" ")
#     if m_time[2] == "28":
#         os.remove(f)

# for i in link1:
#     file = "data/train_data/" + i
#     if i.endswith(".txt"):
#         text = []
#         with open(file, "rt") as f:
#             text = f.readline().split(" ")
#             text[0] = "1"
#         with open(file, "wt") as f:
#             f.write(" ".join(text))

# for i in link2:
#     file = path2 + "/" + i
#     name, tag = i.split(".")
#     os.rename(file, path2+"/"+name+"_motorbike"+"."+tag)


for i in link1:
    name = i.split(".")[0]
    count = 0
    for j in link1:
        if(name==j.split(".")[0]):
            count += 1
    if(count != 2):
        print(i)