import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import cluster
import pandas as pd

csv_path = 'colors.csv'
index = ['color', 'color_name', 'hex', 'R', 'G', 'B']
df = pd.read_csv(csv_path, names=index, header=None)



# Color Name return function

def get_color_name(rgb):
    minimum = 1000
    for i in range(len(df)):
        d = abs(rgb[2] - int(df.loc[i,'R'])) + abs(rgb[1] - int(df.loc[i,'G'])) + abs(rgb[0] - int(df.loc[i,'B']))
        if d <= minimum:
            minimum = d
            cname = df.loc[i, 'color_name']
    return cname

# Using as mask
def color_change(img, maskrgb):
	img[np.all(img == maskrgb, axis=-1)] = (0,0,0)
	return img



img = cv2.imread('car12.jpg')
img = cv2.resize(img, (450, 210))
max_color = []
maxthree = []
color_name  = ""
number = 16




h, w, c = img.shape
img2 = img.reshape(h*w, c)

kmeans_cluster = cluster.KMeans(n_clusters=number)         #cluster form for same color rgb value
kmeans_cluster.fit(img2)
cluster_centers = kmeans_cluster.cluster_centers_
cluster_labels = kmeans_cluster.labels_

img3 = cluster_centers[cluster_labels].reshape(h, w, c)
img3 = img3.astype('uint8')
org_img = img3.copy()
img4 = img3.reshape(-1,3)




# get the unique colors
colors, counts = np.unique(img4, return_counts=True, axis=0)

max_color = colors[list(counts).index(max(counts))]
color_name = get_color_name(max_color)
print("Max :", color_name )
print("Max :", max_color )
maxthree_color = sorted(counts, reverse=True)[:3]
print(maxthree_color)




'''print(lower)
upper = colors[3]

mask = cv2.inRange(img, lower, upper) 
res = cv2.bitwise_or(img, img, mask=mask) 


#unique = zip(colors,counts)'''

# function to convert from r,g,b to hex 
'''
def encode_hex(color):
    b=color[0]
    g=color[1]
    r=color[2]
    hex = '#'+str(bytearray([r,g,b]).hex())
    print(hex)
    return hex

fig = plt.figure()
for i, uni in enumerate(unique):
    color = uni[0]
    count = uni[1]
    plt.bar(i, count, color=encode_hex(color))
'''


for maxcolor in maxthree_color:
    img3 = color_change(img3, list(colors[list(counts).index(maxcolor)]))



#text = color_name + "  B, G, R :" + str(max_color)
#cv2.putText(img3, text, (50,50), 2,0.8, (int(max_color[0]),int(max_color[1]), int(max_color[2])) ,2,cv2.LINE_AA)
difference = cv2.subtract(org_img, img3)
#fig.savefig('barn_color_historgram.png')



cv2.imshow('reduced colors',difference)
cv2.imshow('Normal Image',img)
#plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()