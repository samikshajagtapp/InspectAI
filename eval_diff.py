import cv2
import numpy as np

good = cv2.imread('mvtec_dataset/bottle/test/good/000.png', cv2.IMREAD_GRAYSCALE)
bad = cv2.imread('mvtec_dataset/bottle/test/broken_large/000.png', cv2.IMREAD_GRAYSCALE)

diff = cv2.absdiff(good, bad)
diff = cv2.GaussianBlur(diff, (21, 21), 0)
print("Max diff:", diff.max(), "Mean diff:", diff.mean())
