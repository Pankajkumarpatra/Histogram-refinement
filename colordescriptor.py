# import the necessary packages
import numpy as np
import cv2
from numpy import linalg as LA
#from scipy.stats import skew

class ColorDescriptor:
    
	def get_pixel1(self, img, center, x, y):
	    new_value = 0
	    try:
	        if img[x][y] >= center:
	            new_value = 1
	    except:
	        pass
	    return new_value

	def lbp_calculated_pixel(self, img, x, y):

	    center = img[x][y]
	    val_ar = []
	    val_ar.append(self.get_pixel1(img, center, x-1, y+1))     # top_right
	    val_ar.append(self.get_pixel1(img, center, x, y+1))       # right
	    val_ar.append(self.get_pixel1(img, center, x+1, y+1))     # bottom_right
	    val_ar.append(self.get_pixel1(img, center, x+1, y))       # bottom
	    val_ar.append(self.get_pixel1(img, center, x+1, y-1))     # bottom_left
	    val_ar.append(self.get_pixel1(img, center, x, y-1))       # left
	    val_ar.append(self.get_pixel1(img, center, x-1, y-1))     # top_left
	    val_ar.append(self.get_pixel1(img, center, x-1, y))       # top

	    power_val = [1, 2, 4, 8, 16, 32, 64, 128]

	    val = 0
	    for i in range(len(val_ar)):
	        val += val_ar[i] * power_val[i]
	    return val
    
	def get_pixel(self, img, x, y):
		new_value = 0
		try:
			if img[x][y] >= 0:
				new_value = img[x][y]
		except:
			pass
		return new_value

	def lsp_calculated_pixel(self, img, x, y):

		center = img[x][y]
		val_ar = []
		val = 0
		val_ar.append(self.get(img, center, x-1, y+1))     # top_right
		val_ar.append(self.get(img, center, x, y+1))       # right
		val_ar.append(self.get(img, center, x+1, y+1))     # bottom_right
		val_ar.append(self.get(img, center, x+1, y))       # bottom
		val_ar.append(self.get(img, center, x+1, y-1))     # bottom_left
		val_ar.append(self.get(img, center, x, y-1))       # left
		val_ar.append(self.get(img, center, x-1, y-1))     # top_left
		val_ar.append(self.get(img, center, x-1, y))       # top
		val_ar.append(center)
		mean = sum(val_ar)/len(val_ar)
		median = sorted(val_ar) [len(val_ar) // 2]
		if(median < mean):
			val = 1
		return val

	def calculate_lbp(self, img_bgr):
		height, width, _ = img_bgr.shape
		img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

		img_lbp = np.zeros((height, width), np.uint8)

		for i in range(0, height):
		    for j in range(0, width):
		        img_lbp[i, j] = self.lbp_calculated_pixel(img_gray, i, j)
        
		features = []
		hist = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
		hist = cv2.normalize(hist, hist).flatten()
		features.extend(hist)
		return features

	def calculate_lsp(self, img_bgr):
		height, width, _ = img_bgr.shape
		img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

		img_lsp = np.zeros((height, width), np.uint8)
		img_skew = np.zeros((height, width), np.uint8)
		img_lbp = np.zeros((height, width), np.uint8)
		for i in range(0, height):
			for j in range(0, width):
				img_lbp[i, j] = self.lbp_calculated_pixel(img_gray, i, j)
        
		for i in range(0, height):
			for j in range(0, width):
				img_skew[i, j] = self.lsp_calculated_pixel(img_gray, i, j)
        
		(values,counts) = np.unique(img_lbp,return_counts=True)
		ind=np.argmax(counts)
		max_value = values[ind]

		for i in range(0, height):
			for j in range(0, width):
				if(img_lbp[i][j]==max_value and img_skew[i][j]==1):
					img_lsp[i][j] = img_lbp[i][j]
				elif(img_lbp[i][j]==max_value and img_skew[i][j]==0):
					img_lsp[i][j] = img_lbp[i][j]

		features = []
		
		hist = cv2.calcHist([img_lsp], [0], None, [256], [0, 256])
		hist = cv2.normalize(hist, hist).flatten()
		features.extend(hist)
		
		return features
    
	def bem_calculated_pixel(self, img, x, y, area):
		valx = []
		valy = []
		if(self.get_pixel(img, x-2, y-2) == 1):    #starting from top_left - 1
			valx.append(x-2);valy.append(y-2)
		if(self.get_pixel(img, x-2, y-1) == 1):    #2
			valx.append(x-2);valy.append(y-1)
		if(self.get_pixel(img, x-2, y) == 1):    #3
			valx.append(x-2);valy.append(y)
		if(self.get_pixel(img, x-2, y+1) == 1):    #4
			valx.append(x-2);valy.append(y+1)
		if(self.get_pixel(img, x-2, y+2) == 1):    #5
			valx.append(x-2);valy.append(y+2)
		if(self.get_pixel(img, x-1, y+2) == 1):    #6
			valx.append(x-1);valy.append(y+2)
		if(self.get_pixel(img, x, y+2) == 1):    #7
			valx.append(x);valy.append(y+2)
		if(self.get_pixel(img, x+1, y+2) == 1):    #8
			valx.append(x+1);valy.append(y+2)
		if(self.get_pixel(img, x+2, y+2) == 1):    #9
			valx.append(x+2);valy.append(y+2)
		if(self.get_pixel(img, x+2, y+1) == 1):    #10
			valx.append(x+2);valy.append(y+1)
		if(self.get_pixel(img, x+2, y) == 1):    #11
			valx.append(x+2);valy.append(y)
		if(self.get_pixel(img, x+2, y-1) == 1):    #12
			valx.append(x+2);valy.append(y-1)
		if(self.get_pixel(img, x+2, y-2) == 1):    #13
			valx.append(x+2);valy.append(y-2)
		if(self.get_pixel(img, x+1, y-2) == 1):    #14
			valx.append(x+1);valy.append(y-2)
		if(self.get_pixel(img, x, y-2) == 1):    #15
			valx.append(x);valy.append(y-2)
		if(self.get_pixel(img, x-1, y-2) == 1):    #16
			valx.append(x-1);valy.append(y-2)
		if(self.get_pixel(img, x-1, y-1) == 1):    #top_left
			valx.append(x-1);valy.append(y-1)
		if(self.get_pixel(img, x-1, y) == 1):    #top
			valx.append(x-1);valy.append(y)
		if(self.get_pixel(img, x-1, y+1) == 1):    #top_right
			valx.append(x-1);valy.append(y+1)
		if(self.get_pixel(img, x, y+1) == 1):    #right
			valx.append(x);valy.append(y+1)
		if(self.get_pixel(img, x+1, y+1) == 1):    #bottom_right
			valx.append(x+1);valy.append(y+1)
		if(self.get_pixel(img, x+1, y) == 1):    #bottom
			valx.append(x+1);valy.append(y)
		if(self.get_pixel(img, x+1, y-1) == 1):    #bottom_left
			valx.append(x+1);valy.append(y-1)
		if(self.get_pixel(img, x, y-1) == 1):    #left
			valx.append(x);valy.append(y-1)
		
		sumx = sum(valx)
		sumy = sum(valy)
		sumxy = 0
		sumx2 = sumy2 = 0
		for i in range(0, len(valx)):
			sumxy += valx[i] * valy[i]
			sumx2 += valx[i] * valx[i]
			sumy2 += valy[i] * valy[i]

		m00 = area
		m10 = sumx/m00
		m01 = sumy/m00
		u11 = (sumxy/m00) - ((m10 * m01)/m00)
		u20 = (sumx2/m00) - ((m10 * m10)/m00)
		u02 = (sumy2/m00) - ((m01 * m01)/m00)
		
		neg_u11 = u11 / m00
		neg_u20 = u02 / m00
		neg_u02 = u20 / m00

		w, v = LA.eig(np.array([[neg_u20, neg_u11], [neg_u11, neg_u02]]))
		#print(w, v)
		evm = img[x][y] - w
		return img[x][y]

	def calculate_bem(self, img_bgr):
		img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
		t = 150
		bina = 1.0 * (img_gray > t)
		height, width= bina.shape
		area = height * width
		img_bem = np.zeros((height, width), np.uint8)
		for i in range(0, height):
			for j in range(0, width):
				img_bem[i, j] = self.bem_calculated_pixel(bina, i, j, area)

		features = []
		hist = cv2.calcHist([img_bem], [0], None, [256], [0, 256])
		hist = cv2.normalize(hist, hist).flatten()
		features.extend(hist)
		return features