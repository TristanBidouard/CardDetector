import cv2
import numpy as np

WEBCAM_HEIGHT = 720
WEBCAM_WIDTH  = 1280 

BKG_THRESH = 60
CARD_MAX_AREA = 120000
CARD_MIN_AREA = 25000

class CardObject:

	def __init__(self):
		self.contours 		= []
		self.height 		= 0
		self.width 			= 0
		self.cornerPoints 	= []
		self.center 		= []
		self.warp 			= []
		self.rankImg 		= [] 
		self.suitImg 		= []
		self.bestRankMatch 	= "Unknown" 
		self.bestSuitMatch 	= "Unknown"
		self.rankDiff 		= 0 
		self.SuitDiff 		= 0

def preprocessImage(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)
	bkgLevel = gray[int(WEBCAM_HEIGHT/100)][int(WEBCAM_WIDTH/2)]
	threshLevel = bkgLevel + BKG_THRESH
	thresh = cv2.threshold(blur,threshLevel,255,cv2.THRESH_BINARY)[1]
	return thresh

def preprocessCard(contour, img):

	Card = CardObject()

	Card.contour = contour

	perimeter = cv2.arcLength(contour, True)
	approx = cv2.approxPolyDP(contour, 0.01*perimeter, True) 
	points = np.float32(approx)
	Card.cornerPoints = points


	x, y, w, h = cv2.boundingRect(contour)
	Card.width, Card.height = w, h

	average = np.sum(points, axis=0)/len(points)
	cent_x = int(average[0][0])
	cent_y = int(average[0][1])
	Card.center = [cent_x, cent_y]

	Card.warp = flattener(img, points, w, h)

	#cv2.imshow('Card', Card.warp)
	#cv2.waitKey(1)
	
	"""# Grab corner of warped card image and do a 4x zoom
	Qcorner = qCard.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
	Qcorner_zoom = cv2.resize(Qcorner, (0,0), fx=4, fy=4)

	# Sample known white pixel intensity to determine good threshold level
	white_level = Qcorner_zoom[15,int((CORNER_WIDTH*4)/2)]
	thresh_level = white_level - CARD_THRESH
	if (thresh_level <= 0):
		thresh_level = 1
	retval, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2. THRESH_BINARY_INV)
	
	# Split in to top and bottom half (top shows rank, bottom shows suit)
	Qrank = query_thresh[20:185, 0:128]
	Qsuit = query_thresh[186:336, 0:128]

	# Find rank contour and bounding rectangle, isolate and find largest contour
	dummy, Qrank_cnts, hier = cv2.findContours(Qrank, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea,reverse=True)

	# Find bounding rectangle for largest contour, use it to resize query rank
	# image to match dimensions of the train rank image
	if len(Qrank_cnts) != 0:
		x1,y1,w1,h1 = cv2.boundingRect(Qrank_cnts[0])
		Qrank_roi = Qrank[y1:y1+h1, x1:x1+w1]
		Qrank_sized = cv2.resize(Qrank_roi, (RANK_WIDTH,RANK_HEIGHT), 0, 0)
		qCard.rank_img = Qrank_sized

	# Find suit contour and bounding rectangle, isolate and find largest contour
	dummy, Qsuit_cnts, hier = cv2.findContours(Qsuit, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	Qsuit_cnts = sorted(Qsuit_cnts, key=cv2.contourArea,reverse=True)
	
	# Find bounding rectangle for largest contour, use it to resize query suit
	# image to match dimensions of the train suit image
	if len(Qsuit_cnts) != 0:
		x2,y2,w2,h2 = cv2.boundingRect(Qsuit_cnts[0])
		Qsuit_roi = Qsuit[y2:y2+h2, x2:x2+w2]
		Qsuit_sized = cv2.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
		qCard.suit_img = Qsuit_sized"""

	return Card

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
		perimeter = cv2.arcLength(contoursSort[i], True)
		approx = cv2.approxPolyDP(contoursSort[i], 0.01*perimeter, True)
		if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA) and (hierarchySort[i][3] == -1) and (len(approx) == 4)):
			contourIscard[i] = 1


	return contoursSort, contourIscard


def findContours(img):
	_, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	return contours, hierarchy

def draw(img, contours):
	imgWithContours = cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
	return imgWithContours

def flattener(img, points, w, h):
	
	temp_rect = np.zeros((4,2), dtype = "float32")
	
	s = np.sum(points, axis = 2)

	tl = points[np.argmin(s)]
	br = points[np.argmax(s)]

	diff = np.diff(points, axis = -1)
	tr = points[np.argmin(diff)]
	bl = points[np.argmax(diff)]

	if w <= 0.8*h:
		temp_rect[0] = tl
		temp_rect[1] = tr
		temp_rect[2] = br
		temp_rect[3] = bl

	if w >= 1.2*h:
		temp_rect[0] = bl
		temp_rect[1] = tl
		temp_rect[2] = tr
		temp_rect[3] = br
	
	if w > 0.8*h and w < 1.2*h: 
		if points[1][0][1] <= points[3][0][1]:
			temp_rect[0] = points[1][0]
			temp_rect[1] = points[0][0]
			temp_rect[2] = points[3][0]
			temp_rect[3] = points[2][0]

		if points[1][0][1] > points[3][0][1]:
			temp_rect[0] = points[0][0]
			temp_rect[1] = points[3][0]
			temp_rect[2] = points[2][0]
			temp_rect[3] = points[1][0]
			
		
	maxWidth = 200
	maxHeight = 300

	dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
	M = cv2.getPerspectiveTransform(temp_rect,dst)
	warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
	warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

		

	return warp

class TrainRanks:

	def __init__(self):
		self.img = []
		self.name = "Placeholder"

class TrainSuits:

	def __init__(self):
		self.img = [] 
		self.name = "Placeholder"


def loadRanks(filepath):

	trainRanks = []
	i = 0
	
	for Rank in ['Ace','Two','Three','Four','Five','Six','Seven',
				 'Eight','Nine','Ten','Jack','Queen','King']:
		trainRanks.append(TrainRanks())
		trainRanks[i].name = Rank
		filename = Rank + '.jpg'
		trainRanks[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
		i = i + 1

	return trainRanks

def loadSuits(filepath):

	trainSuits = []
	i = 0
	
	for Suit in ['Spades','Diamonds','Clubs','Hearts']:
		trainSuits.append(TrainSuits())
		trainSuits[i].name = Suit
		filename = Suit + '.jpg'
		trainSuits[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
		i = i + 1

	return trainSuits



	