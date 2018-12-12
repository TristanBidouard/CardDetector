import cv2
import numpy as np

WEBCAM_HEIGHT = 720
WEBCAM_WIDTH  = 1280 

BKG_THRESH = 60
CARD_MAX_AREA = 120000
CARD_MIN_AREA = 25000

class Card:

	def __init__(self):
		self.contours = []
		self.height, self.width = 0

def preprocessImage(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)
	bkgLevel = gray[int(WEBCAM_HEIGHT/100)][int(WEBCAM_WIDTH/2)]
	threshLevel = bkgLevel + BKG_THRESH
	thresh = cv2.threshold(blur,threshLevel,255,cv2.THRESH_BINARY)[1]
	return thresh

def findCards(img):
	mask = preprocessImage(img)
	contours, hierarchy = findContours(mask)

	indexSort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]),reverse=True)

	if len(contours) == 0:
		return [], []

	contoursSort = []
	hierarchySort = []
	contourIscard = np.zeros(len(contours),dtype=int)

	for i in indexSort:
		contoursSort.append(contours[i])
		hierarchySort.append(hierarchy[0][i])

	for i in range(len(contoursSort)):
		size = cv2.contourArea(contoursSort[i])
		peri = cv2.arcLength(contoursSort[i],True)
		approx = cv2.approxPolyDP(contoursSort[i],0.01*peri,True)
		if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA) and (hierarchySort[i][3] == -1) and (len(approx) == 4)):
			contourIscard[i] = 1


	return contoursSort, contourIscard


def findContours(img):
	_, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	return contours, hierarchy

def draw(img, contours):
	imgWithContours = cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
	return imgWithContours
	