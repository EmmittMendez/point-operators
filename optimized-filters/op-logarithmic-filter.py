import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to the input image")
args = vars(parser.parse_args())

image_BGR = cv2.imread(args["image"])
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)

# Calculamos el mapeo de la funcion logaritmica y se almacena el resultado en un arreglo "y2"
#r = 255
r = image_RGB.max() # Obtenemos el valor máximo de los 3 canales de la imagen
c = 255 / np.log(1.0 + r)

# Construimos un vector X y Y para almacenar los valores
# del operador logaritmico . El vector Y servira para la transformación de 
# los valores de la imagen, usando programación dinamica
x = np.linspace(0, 255,256)
y2 = np.array(c * np.log(1 + x), dtype='uint8')

# Definimos una función lambda para la función logaritmica
logarithmic = lambda m: y2[m]

# Se aplica la función lambda a cada pixel de la imagen. Se vectoriza la
# matriz para que pueda ser procesado por la función lambda
image_RGB_log = np.array(np.vectorize(logarithmic)(image_RGB), dtype = 'uint8')

#Definimos el tamaño de la ventana y los subplots
fig = plt.figure(figsize=(9,6))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,3,5)

ax1.imshow(image_RGB)
ax1.set_title("Original Image")

ax2.imshow(image_RGB_log)
ax2.set_title("Logarithmic Correction")

# Se dibuja la funcion identidad
x = np.linspace(0, 255, 255)
y1 = x

# Se dibuja la funcion identidad y logaritmica
c2 = 255 / np.log(256)
y2 = np.array(c * np.log(1 + x))
ax3.plot(x, y1, color="r", linewidth=1, label = "Id. Func.")
msg = "Logarithmic Func."
ax3.plot(x, y2, color="b", linewidth=1, label = msg)
ax3.legend()
ax3.set_title("Logarithmic Function")
plt.show()
