import cv2
import numpy as np
import os
import Cards


def main():

	freq = cv2.getTickFrequency()
	frameRateCalc = 1

	webcam = cv2.VideoCapture(0)

	path = os.path.dirname(os.path.abspath(__file__))
	trainRanks = Cards.loadRanks( path + '/Img/')
	trainSuits = Cards.loadSuits( path + '/Img/')

	while True:
		
		t1 = cv2.getTickCount()

		img = webcam.read()[1]
		contoursSort, contourIscard = Cards.findCards(img)

		cards = []
		if len(contoursSort) != 0:

			
			k = 0

			for i in range(len(contoursSort)):
				if (contourIscard[i] == 1):
					cards.append(Cards.preprocessCard(contoursSort[i], img))
					cards[k].bestRankMatch, cards[k].bestSuitMatch, cards[k].rankDiff, cards[k].suitDiff = Cards.matchCard(cards[k],trainRanks,trainSuits)
					#print cards[k].bestRankMatch
					#print cards[k].bestSuitMatch
					#print cards[k].rankDiff
					#print cards[k].suitDiff
					img = Cards.drawResults(img, cards[k])
					k = k + 1

		Cards.drawContours(img, cards)
		Cards.drawFrameRate(img, frameRateCalc)

		cv2.imshow('CardDetector', img)

		t2 = cv2.getTickCount()
		time1 = (t2-t1)/freq
		frameRateCalc = 1/time1

		if cv2.waitKey(1) == 27:
			break  # esc to quit
		cv2.destroyAllWindows()



if __name__ == '__main__':
	main()