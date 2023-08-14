import cv2
import matplotlib.pyplot as plt
import sys
from Panorama import Panorama

if len(sys.argv) < 3:
  print("Need 2 images to work.")
  print("Usage: main.py image1.jpg image2.jpg")
  exit()

img_left = cv2.imread(sys.argv[1])
img_right = cv2.imread(sys.argv[2])

panorama_image = Panorama().create(img_left, img_right, True)

plt.title("Panorama")
plt.imshow(panorama_image[:,:,::-1].astype(int))
plt.show()
cv2.imwrite("panorama.jpg", panorama_image)
