import cv2
import numpy as np
import os
import Cards


def main():

	webcam = cv2.VideoCapture(0)

	path = os.path.dirname(os.path.abspath(__file__))
	trainRanks = Cards.loadRanks( path + '/Img/')
	trainSuits = Cards.loadSuits( path + '/Img/')

	while True:
		
		img = webcam.read()[1]
		contoursSort, contourIscard = Cards.findCards(img)

		cards = []
		if len(contoursSort) != 0:

			
			k = 0

			for i in range(len(contoursSort)):
				if (contourIscard[i] == 1):
					cards.append(Cards.preprocessCard(contoursSort[i], img))
					#cv2.imshow('Card', cards[k].warp)
					#cv2.waitKey(1)
					k = k + 1
		#img = Cards.draw(img, cards)
		#cv2.imshow('Final', img)

		if cv2.waitKey(1) == 27:
			break  # esc to quit
		cv2.destroyAllWindows()



if __name__ == '__main__':
	main()