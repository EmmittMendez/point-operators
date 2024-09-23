import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to the input image")
args = vars(parser.parse_args())

image_BGR = cv2.imread(args["image"])
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)

# Aplicamos el filtro de aclarado del seno
r = image_RGB.max() # Obtenemos el valor máximo de los 3 canales de la imagen
k =  255 / np.sin(np.pi/2)
image_RGB_log = np.array(k * np.sin(image_RGB*np.pi/(2*r)), dtype = 'uint8')

#Definimos el tamaño de la ventana y los subplots
fig = plt.figure(figsize=(9,6))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,3,5)

ax1.imshow(image_RGB)
ax1.set_title("Original Image")

ax2.imshow(image_RGB_log)
ax2.set_title("Sine filter")

# Se dibuja la funcion identidad
x = np.linspace(0, 255, 255)
y1 = x

# Se dibuja la funcion identidad y del seno
c2 = 255 / np.sin(np.pi/2)
y2 = np.array(c2 * np.sin(x*np.pi/(2*255)))
ax3.plot(x, y1, color="r", linewidth=1, label = "Id. Func.")
msg = "Sine Func."
ax3.plot(x, y2, color="b", linewidth=1, label = msg)
ax3.legend()
ax3.set_title("Sine Function")
plt.show()
