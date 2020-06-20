# import the necessary packages
from colordescriptor import ColorDescriptor
from searcher import Searcher
import cv2, numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as d

# initialize the image descriptor
cd = ColorDescriptor()

# load the query image
query = cv2.imread('queries/3_301.jpg')

# Set the timer
now = d.now()
current = now.strftime("%H:%M:%S")
print(current)

# describe LBP features and perform the search
lbp_features = cd.calculate_lbp(query)
lbp_searcher = Searcher("lbp_value.csv")
lbp_results = lbp_searcher.search(lbp_features)
print(lbp_results)

# Set the timer
now = d.now()
current = now.strftime("%H:%M:%S")
print(current)

# Plot result images from LBP

filenames = []
for (score, resultID) in lbp_results:
    filenames.append("Corel/" + resultID)
print(filenames)

fig, ax = plt.subplots(2, 5, dpi = 100)
plt.subplots_adjust(wspace = 0.9)
for i in range(10):
    with open(filenames[i],'rb') as f:
        image=plt.imread(f)
        ax[i%2][i//2].imshow(image)
        
plt.show()

# Set the timer
now = d.now()
current = now.strftime("%H:%M:%S")
print(current)

# describe BEM features and perform the search
bem_features = cd.calculate_bem(query)
bem_searcher = Searcher("bem_value.csv")
bem_results = bem_searcher.search(bem_features)
print(bem_results)

# Plot result images from LBP

filenames = []
for (score, resultID) in bem_results:
    filenames.append("Corel/" + resultID)
print(filenames)

fig, ax = plt.subplots(2, 5, dpi = 100)
plt.subplots_adjust(wspace = 0.7)
for i in range(10):
    with open(filenames[i],'rb') as f:
        image=plt.imread(f)
        ax[i%2][i//2].imshow(image)
plt.show()

bem_precision_results = lbp_precision_results = []
bem_precision_results = bem_searcher.precision_recall(bem_features)
lbp_precision_results = lbp_searcher.precision_recall(lbp_features)
print(np.multiply(lbp_precision_results, 100))
fig = plt.figure(figsize=(10,8))
ax  = fig.add_subplot(1,2,1)
ax.plot(lbp_precision_results)
ax.set_title("LBP image")
ax.set_xlabel("Number of retrieved images")
ax.set_ylabel("Precision")

ax  = fig.add_subplot(1,2,2)
ax.plot(bem_precision_results)
ax.set_title("LBP with refinement")
ax.set_xlabel("Number of retrieved images")
ax.set_ylabel("Precision")

plt.show()

print("Search operation completed")