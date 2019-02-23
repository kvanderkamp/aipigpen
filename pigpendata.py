# import the necessary packages
import numpy as np
import argparse
import cv2
from random import randint
# pylint: disable=E1101


class PigPenData():
    def __init__(self, train: int, test: int, validate: int, wonkiness: int = 8):
        self.train = train
        self.test = test
        self.validate = validate
        self.wonkiness = wonkiness
        self.cipher = {}
        for char in range(65, 91):
            img = cv2.imread('char(' + repr(char) + ').png')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rows, cols = gray.shape
            M = np.float32([[1, 0, 6], [0, 1, 0]])
            dst = cv2.warpAffine(gray, M, (cols, rows), borderValue=255)
            self.cipher[char-65] = dst

    def _four_point_transform(self, image: np.array, pts: np.array):
        # obtain a consistent order of the points and unpack them
        # individually
        (tl, tr, br, bl) = pts

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(
            image, M, (maxWidth, maxHeight), borderValue=255)

        # return the warped image
        return warped

    def _wonkify(self, image: np.array, wonky: int):
        imx = image.shape[0]
        imy = image.shape[1]
        # wonky = 8
        tl = (randint(0-wonky, 0+wonky), randint(0-wonky, 0+wonky))
        tr = (randint(imy-wonky, imy+wonky), randint(0-wonky, 0+wonky))
        br = (randint(imy-wonky, imy+wonky), randint(imx-wonky, imx+wonky))
        bl = (randint(0-wonky, 0+wonky), randint(imx-wonky, imx+wonky))
        pts = np.array([tl, tr, br, bl], dtype="float32")

        # apply the four point tranform to obtain a "birds eye view" of
        # the image
        warped = self._four_point_transform(image, pts)
        return warped

    def _produce_set(self, length: int, wonkiness: int = 8):
        images = []
        characters = []
        for _ in range(length):
            char = randint(0, 25)
            onehot = [0 for _ in range(26)]
            onehot[char] = 1
            characters.append(onehot)
            gray = self.cipher[char]
            wonk = self._wonkify(gray, wonkiness)
            wonk = cv2.resize(wonk, gray.shape, interpolation=cv2.INTER_CUBIC)
            images.append(wonk)
        return (images, characters)

    def load_data(self):
        train = self._produce_set(self.train, self.wonkiness)
        test = self._produce_set(self.test, self.wonkiness)
        validate = self._produce_set(self.validate, self.wonkiness)
        return train, test, validate
