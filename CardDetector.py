import cv2
import numpy as np
import Cards


def main():

	webcam = cv2.VideoCapture(0)

	while True:
		
		img = webcam.read()[1]
		img = cv2.flip(img, 1)
		contoursSort, contourIscard = Cards.findCards(img)

		if len(contoursSort) != 0:

			cards = []
			k = 0

			for i in range(len(contoursSort)):
				if (contourIscard[i] == 1):
					cards.append(contoursSort[i])

		img = Cards.draw(img, cards)
		cv2.imshow('my webcam', img)
		if cv2.waitKey(1) == 27:
			break  # esc to quit
		cv2.destroyAllWindows()



if __name__ == '__main__':
	main()