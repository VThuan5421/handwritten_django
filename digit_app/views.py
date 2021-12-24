from django.shortcuts import render, redirect, HttpResponse

from digit_app.apps import DigitAppConfig
from .models import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
from rest_framework.decorators import api_view
from django.http import JsonResponse
import cv2
# Create your views here.
def index(request):
    # convert to gray scale
    # image = image.convert('L')
    return render(request, 'index.html')

def canvas_digit(request):
    return render(request, 'canvas_digit.html')

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
    10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',
    20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}

def image_digit(request):
    if request.method == "POST":
        image = request.FILES.get('file')
        image_bytes = image.read()
        # decode file request to image array
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        digit, acc = 0, 0
        if len(img.shape) == 2:
            digit, acc = classify_handwriting_mnist(img)
        elif len(img.shape) == 3:
            digit, acc = classify_digit_test(img)
        
        return render(request, 'image_digit.html', {"digit": str(digit), "acc": str(acc)})

    return render(request, 'image_digit.html')

def image_alphabet(request):
    if request.method == "POST":
        image = request.FILES.get('file')
        image_bytes = image.read()
        # decode file request to image array
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        alphabet, acc = 0, 0
        if len(img.shape) == 2:
            alphabet, acc = classify_handwritting_alphabet(img)
        elif len(img.shape) == 3:
            alphabet, acc = classify_alphabet_test(img)
        return render(request, 'image_alphabet.html', {'word': str(alphabet), 'acc': str(acc)})

    return render(request, 'image_alphabet.html')

    
# ============================ For randomly digit image =====================
def classify_digit_test(img):
    height = img.shape[0]
    width = img.shape[1]
    cropped_image = img
    if height > 400 and width > 400:
        cropped_image = img[int(height/2-height/3): int(height/2+height/3), int(width/2-width/3): int(width/2+width/3)]
        #show_image_plt(cropped_image, 'Cropped Image')
    # converting to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # apply thresholding
    ret, th = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)
    # find contour
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #get_shape_value_image(th, show = False)
    #show_image_plt(th, 'Th Image')
    digit, acc = predict_digit(th)
    return digit, acc

def predict_digit(img):
    # resize image to 28 x 28 pixels
    img = np.asarray(img)
    img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
    img = img.reshape(1, 28, 28, 1)
    # normalizing the image to support our model input
    img = img / 255.0
    # predicting the class
    res = DigitAppConfig.digitmodel.predict([img])[0]
    return np.argmax(res), max(res)

# ================= FOR randomly alphabet images ===============
def classify_alphabet_test(img):
    #get_shape_value_image(img, show = False)
    height = img.shape[0]
    width = img.shape[1]
    cropped_image = img
    if height > 400 and width > 400:
        cropped_image = img[int(height/2-height/3): int(height/2+height/3), int(width/2-width/3): int(width/2+width/3)]
        #show_image_plt(cropped_image, 'Cropped image')
    # Converting to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    #show_image_plt(gray, 'Gray image')
    # apply thresholding
    ret, th = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)
    # find contour
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #get_shape_value_image(th, show = False)
    #show_image_plt(th, 'Th Image')
    digit, acc = predict_alphabet_test(th)
    return digit, acc

def predict_alphabet_test(img):
    img = np.asarray(img)
    img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
    #show_image_plt(img, 'resize 28 28')
    img = img.reshape(1, 28, 28, 1)
    res = DigitAppConfig.alphabetmodel.predict([img])[0]
    return word_dict[np.argmax(res)], max(res)

# ===================== FOR image in mnist ======================
def classify_handwriting_mnist(image):
    digit, acc = predict_digit_mnist(image)
    return digit, acc

def predict_digit_mnist(img):
    img = np.asarray(img)
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    res = DigitAppConfig.digitmodel.predict([img])[0]
    return np.argmax(res), max(res)

# ====================== FOR alphabet image in dataset =============
def classify_handwritting_alphabet(image):
    alphabet, acc = predict_alphabet(image)
    return alphabet, acc

def predict_alphabet(image):
    img = np.asarray(image)
    img = img.reshape(1, 28, 28, 1)
    res = DigitAppConfig.alphabetmodel.predict([img])[0]
    return word_dict[np.argmax(res)], max(res)


# ======================== TEST CODE ==========================
def classify_handwriting(image):
    #img = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
    img = image
    # converting to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply thresholding
    ret, th = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)
    # find the contours
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    for cnt in contours:
        # get bounding box and exact region of interest
        x, y, w, h = cv2.boundingRect(cnt)
        # create rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0,0), 1)
        top = int(0.05 * th.shape[0])
        bottom = top
        left = int(0.05 * th.shape[1])
        right = left
        th_up = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)
        # Extract the image's region of interest
        roi = th[y - top: y + h + bottom, x - left: x + w + right]
        digit, acc = predict_digit(roi)
        return digit, acc

@api_view(['POST'])
def getimagefromrequest(request):
    image = request.FILES.get('file')
    image_bytes = image.read()
    digit, acc = classify_handwriting(image_bytes)
    print(str(digit))
    return JsonResponse({"digit": str(digit), 'accuracy': str(acc)})


def convert_files_to_imageio(img_files, extensions):
    temp = Image.open(img_files)
    byte_io = BytesIO()
    temp.save(byte_io, extensions)
    return temp

def get_shape_value_image(temp, show = True):
    image = np.asarray(temp)
    print("Image shape: ", image.shape)
    if show:
        print(image)

def show_image_plt(image, title = ""):
    plt.figure(figsize = (8, 6))
    plt.imshow(image, interpolation="none")
    plt.title(title)
    plt.show()

"""
image = request.FILES['fileToUpload']
temp = Image.open(image)
byte_io = BytesIO()
temp.save(byte_io, 'png')
files = {'fileToUpload': byte_io.getvalue() }
response = requests.post( self.URL, files=files)
print(response.status_code, response.content, response.reason)
"""

"""
if form.is_valid():
    url = form.cleaned_data['url']
    url_decoded = b64decode(url.encode())        
    content = ContentFile(url_decoded) 
    your_model.model_field.save('image.png', content)
"""