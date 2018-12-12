import cv2
import numpy as np

WEBCAM_HEIGHT = 720
WEBCAM_WIDTH  = 1280 

BKG_THRESH = 60
CARD_THRESH = 30

CARD_MAX_AREA = 120000
CARD_MIN_AREA = 25000

CORNER_WIDTH = 32
CORNER_HEIGHT = 84

RANK_WIDTH = 70
RANK_HEIGHT = 125

SUIT_WIDTH = 70
SUIT_HEIGHT = 100

RANK_DIFF_MAX = 2000
SUIT_DIFF_MAX = 700

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
	
	Qcorner = Card.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
	QcornerZoom = cv2.resize(Qcorner, (0,0), fx=4, fy=4)

	whiteLevel = QcornerZoom[15,int((CORNER_WIDTH*4)/2)]
	threshLevel = whiteLevel - CARD_THRESH
	if (threshLevel <= 0):
		threshLevel = 1
	_, queryThresh = cv2.threshold(QcornerZoom, threshLevel, 255, cv2.THRESH_BINARY_INV)
	
	Qrank = queryThresh[20:185, 0:128]
	Qsuit = queryThresh[186:336, 0:128]

	cv2.imshow("Warp", Card.warp)
	cv2.moveWindow("Warp", 20,20);
	cv2.imshow("Corner", Qcorner)
	cv2.moveWindow("Corner", 20, 350);
	cv2.imshow("Rank", Qrank)
	cv2.moveWindow("Rank", 250,20);
	cv2.imshow("Suit", Qsuit)
	cv2.moveWindow("Suit", 250,250);

	_, QrankCountours, _ = cv2.findContours(Qrank, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	QrankCountours = sorted(QrankCountours, key=cv2.contourArea,reverse=True)

	if len(QrankCountours) != 0:
		x1,y1,w1,h1 = cv2.boundingRect(QrankCountours[0])
		QrankRoi = Qrank[y1:y1+h1, x1:x1+w1]
		QrankSized = cv2.resize(QrankRoi, (RANK_WIDTH,RANK_HEIGHT), 0, 0)
		Card.rankImg = QrankSized
		cv2.imshow("QrankSized", QrankSized)
		cv2.moveWindow("QrankSized", 500,20);

	
	_, QsuitContours, _ = cv2.findContours(Qsuit, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	QsuitContours = sorted(QsuitContours, key=cv2.contourArea,reverse=True)
	
	if len(QsuitContours) != 0:
		x2,y2,w2,h2 = cv2.boundingRect(QsuitContours[0])
		QsuitRoi = Qsuit[y2:y2+h2, x2:x2+w2]
		QsuitSized = cv2.resize(QsuitRoi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
		Card.suitImg = QsuitSized
		cv2.imshow("QsuiteSized", QsuitSized)
		cv2.moveWindow("QsuiteSized", 500, 250);


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

def matchCard(Card, trainRanks, trainSuits):

	bestRankMatchDiff = 10000
	bestSuitMatchDiff = 10000
	bestRankMatchName = "Unknown"
	bestSuitMatchName = "Unknown"
	i = 0

	if (len(Card.rankImg) != 0) and (len(Card.suitImg) != 0):

		print 1
		
		for Trank in trainRanks:

				diffImg = cv2.absdiff(Card.rankImg, Trank.img)
				rankDiff = int(np.sum(diffImg)/255)
				
				if rankDiff < bestRankMatchDiff:
					bestRankDiffImg = diffImg
					bestRankMatchDiff = rankDiff
					bestRankName = Trank.name

		for Tsuit in trainSuits:
				
				diffImg = cv2.absdiff(Card.suitImg, Tsuit.img)
				suitDiff = int(np.sum(diffImg)/255)
				
				if suitDiff < bestSuitMatchDiff:
					bestSuitDiffImg = diffImg
					bestSuitMatchDiff = suitDiff
					bestSuitName = Tsuit.name

	if (bestRankMatchDiff < RANK_DIFF_MAX):
		bestRankMatchName = bestRankName

	if (bestSuitMatchDiff < SUIT_DIFF_MAX):
		bestSuitMatchName = bestSuitName

	return bestRankMatchName, bestSuitMatchName, bestRankMatchDiff, bestSuitMatchDiff


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





	