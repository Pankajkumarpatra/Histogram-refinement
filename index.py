# import the necessary packages
from colordescriptor import ColorDescriptor
import os
import matplotlib.pyplot as plt
from datetime import datetime as d

# initialize the color descriptor
cd = ColorDescriptor()

dir_images = 'Corel'

imgs = os.listdir(dir_images)

# Set the timer
now = d.now()
current = now.strftime("%H:%M:%S")
print(current)

# Store imageID and LBP features

output = open("lbp_value.csv", "w")
for imgnm in imgs:
	img_rgb = plt.imread(os.path.join(dir_images,imgnm))
	imageID = imgnm
	features = cd.calculate_lbp(img_rgb)
    
	# write the features to file
	features = [str(f) for f in features]
	output.write("%s,%s\n" % (imageID, ",".join(features)))

output.close()

# Set the timer
now = d.now()
current = now.strftime("%H:%M:%S")
print(current)

# Store imageID and BEM features
i = 0
output = open("bem_value.csv", "w")
for imgnm in imgs:
	img_rgb = plt.imread(os.path.join(dir_images,imgnm))
	imageID = imgnm
	features = cd.calculate_lbp(img_rgb)
    
	# write the features to file
	features = [str(f) for f in features]
	output.write("%s,%s\n" % (imageID, ",".join(features)))
	i= i+1
	print(i)
output.close()

# Set the timer
now = d.now()
current = now.strftime("%H:%M:%S")
print(current)