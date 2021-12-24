"""
import matplotlib.pyplot as plt
import cv2
import numpy as np


def classify_handwriting(image):
    # convert to numpy array image
    img = image
    show_shape(img, "Shape")
    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    show_image(gray, 'Gray Image')
    # apply thresholding
    ret, th = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)
    show_image(th, 'Threshold')
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    i = 0
    for cnt in contours:
        
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        show_image(cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0, 2)))
        top = int(0.5 * th.shape[0]); bottom = top
        left = int(th.shape[1] * 0.5); right = left
        th_up = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)
        roi = th[y -top: y + h + bottom, x - left: x + w + right]
        

def show_image(image, title = ''):
    plt.figure(figsize = (8, 5))
    plt.imshow(image, interpolation = None)
    plt.title(title)
    plt.show()

def show_shape(image, label):
    print(label + " shape: ", image.shape)

path = r"C:\DJ\digit_django\digit_system\digit_app\static\images\other\5.png"
image = cv2.imread(path)
classify_handwriting(image)


from keras.datasets import mnist
def save_some_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    for i in range(100):
        num = np.random.randint(0, 10000)
        filename = "_" + str(i) + ".jpg"
        cv2.imwrite(str(y_test[num]) + filename, np.asarray(x_test[num]))
    print("Done!")

#save_some_mnist()
"""