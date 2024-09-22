import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Parametros de entrada
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to the input image") # Imagen de entrada
parser.add_argument("-g", "--gamma", required=True, help="Gamma value") # Valor de gamma
args = vars(parser.parse_args())

# Valor de gamma entre 0 y 1 aclara la imagen
# Valor de gamma mayor a 1 oscurece la imagen

image_BGR = cv2.imread(args["image"])
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
gamma = float(args["gamma"])

# Se aplica el filtro gamma a la imagen
image_RGB_gamma = np.array(255 * (image_RGB / 255) ** gamma, dtype = 'uint8')

#Definimos el tamaño de la ventana y los subplots
fig = plt.figure(figsize=(9,6))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,3,5)

ax1.imshow(image_RGB)
ax1.set_title("Original Image")

ax2.imshow(image_RGB_gamma)
ax2.set_title("Gamma Correction")

# Se dibuja la funcion identidad
x = np.linspace(0, 255, 255)
y1 = x

# Se dibuja la funcion gamma
y2 = np.array(255 * (x / 255) ** gamma)
ax3.plot(x, y1, color="r", linewidth=1, label = "Id. Func.")
msg = "Gamma Func.(" + str(gamma) + ")"
ax3.plot(x, y2, color="b", linewidth=1, label = msg)
ax3.legend()
ax3.set_title("Gamma correction function")
plt.show()
