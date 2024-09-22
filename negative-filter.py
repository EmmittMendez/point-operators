import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to the input image")
args = vars(parser.parse_args())

image_BGR = cv2.imread(args["image"])
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)

max = image_RGB.max()   # Obtenemos el valor máximo de los 3 canales de la imagen
image_RGB_negative = max - image_RGB    # Restamos el maximo (escalar) con la imagen (cubo de datos, 3 canales)

fig = plt.figure(figsize=(9,6))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,3,5)

ax1.imshow(image_RGB)
ax1.set_title("Original Image")

ax2.imshow(image_RGB_negative)
ax2.set_title("Negative Image")

x = np.linspace(0, 255, 255)
y1 = x
y2 = np.array(255 - x)
ax3.plot(x, y1, color="r", linewidth=1, label = "Id. Func.")
msg = "Negative Func."
ax3.plot(x, y2, color="b", linewidth=1, label = msg)
ax3.legend()
ax3.set_title("Negative Function")
plt.show()
