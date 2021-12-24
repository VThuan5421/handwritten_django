# Scale image

import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt


def show_image(images):
    plt.figure(figsize = (15, 8))
    # maximum 6 images
    if len(images) < 2:
        plt.imshow(images[0][0])
        plt.title(images[0][1])
    elif len(images) < 5:
        for i in range(len(images)):
            plt.subplot(2, 2, i + 1)
            plt.imshow(images[i][0])
            plt.title(images[i][1])
    else:
         for i in range(len(images)):
            plt.subplot(2, 3, i + 1)
            plt.imshow(images[i][0])
            plt.title(images[i][1])
    
    plt.show()

url = 'https://i.imgur.com/1vzDG2J.jpg'
def _downloadImage(url):
    resp = requests.get(url)
    img = np.asarray(bytearray(resp.content), dtype = "uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img

img = _downloadImage(url)
print("Original image shape: {}".format(img.shape))

# Scale image bằng cách gấp đôi width and height
h, w = img.shape[:2]
imgScale = cv2.resize(img, (int(w*2), int(h * 2)), interpolation=cv2.INTER_LINEAR)
print("Scale image shape: {}".format(imgScale.shape))

#show_image([[img, 'Original Image'], [imgScale, 'Scaled Image']])

# Dịch chuyển ảnh (Translation)
images = [[img, 'origianl image']]
rows, cols = img.shape[:2]
# Dịch chuyển hình ảnh xuống góc dưới bên phải
# Translate to bottom right
tx, ty = (200, 200)
M1 = np.array([[1, 0, tx], [0, 1, ty]], dtype = np.float32)
tran1 = cv2.warpAffine(img, M1, (cols, rows))
images.append([tran1, 'Translate to bottom right'])
# Dịch chuyển ảnh xuống dưới bên trái
# Translate to bottom left
M2 = np.array([[1, 0, -tx], [0, 1, ty]], dtype = np.float32)
tran2 = cv2.warpAffine(img, M2, (cols, rows))
images.append([tran2, 'Translate to bottom left'])
# Dịch chuyển ảnh lên góc trên bên phải
# Translate to up right
M3 = np.array([[1, 0, tx], [0, 1, -ty]], dtype = np.float32)
tran3 = cv2.warpAffine(img, M3, (cols, rows))
images.append([tran3,' Translate to up right'])
# Dịch chuyển ảnh lên góc trên bên trái
# Translate to up left
M4 = np.array([[1, 0, -tx], [0, 1, -ty]], dtype = np.float32)
tran4 = cv2.warpAffine(img, M4, (cols, rows))
images.append([tran4, 'Translate to up left'])

#show_image(images)

images = [[img, 'Origianl image']]
# Xoay ảnh (Rotation)
# Xoay ảnh kích thước 45 độ tại tâm của ảnh, độ phóng đại ảnh không đổi
M5 = cv2.getRotationMatrix2D(center = (cols/2, rows/2), angle = -45, scale = 1)
tran5 = cv2.warpAffine(img, M5, (cols, rows))
images.append([tran5, 'Rotate 45 at centroid'])
# Xoay ảnh kích thước 45 độ tại tâm của ảnh và độ phóng đại giảm 1/2
M6 = cv2.getRotationMatrix2D(center = (cols/2, rows/2), angle = -45, scale = 0.5)
tran6 = cv2.warpAffine(img, M6, (cols, rows))
images.append([tran6, 'Rotate 45 resize 0.5'])
# Xoay ảnh kích thước -45 độ tại tâm của ảnh
M7 = cv2.getRotationMatrix2D(center = (cols/2, rows/2), angle = 45, scale = 1)
tran7 = cv2.warpAffine(img, M7, (cols, rows))
images.append([tran7, 'Rotate -45 at centroid'])
# Xoay ảnh kích thước 20 độ tại góc trên bên trái
M8 = cv2.getRotationMatrix2D(center = (0, 0), angle = -20, scale = 1)
tran8 = cv2.warpAffine(img, M8, (cols, rows))
images.append([tran8, "Rotate 20 at upper left corner"])
# Xoay ảnh kích thước 20 độ tại góc dưới bên phải
M9 = cv2.getRotationMatrix2D(center = (cols, rows), angle = -20, scale = 1)
tran9 = cv2.warpAffine(img, M9, (cols, rows))
images.append([tran9, 'Rotate 20 at bottom right corner'])
#show_image(images)

# Biến đổi Affine
rows, cols, ch = img.shape
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
#pts2 = np.float32([[50, 100], [200, 50], [50, 200]])
pts2 = np.float32([[50, 300], [200, 150], [150, 400]])
M = cv2.getAffineTransform(pts1, pts2)
imageAffine = cv2.warpAffine(img, M, (cols, rows))

def plot():
    # Hiển thị hình ảnh gốc và 3 điểm ban đầu trên ảnh
    plt.subplot(121)
    plt.imshow(img)
    plt.title('Input')
    for (x, y) in pts1:
        plt.scatter(x, y, s = 50, c = 'white', marker ='x')
    # Hiển thị hình ảnh sau dịch chuyển và 3 điểm mục tiêu của phép dịch chuyển
    plt.subplot(122)
    plt.imshow(imageAffine)
    plt.title('Output')
    for (x, y) in pts2:
        plt.scatter(x, y, s = 50, c = 'white', marker = 'x')
    
    plt.show()

#plot()

# Biến đổi phối cảnh (Perspective Transform)
def plot1():
    pts1 = np.float32([[50, 50], [350, 50], [50, 350], [350, 350]])
    pts2 = np.float32([[0, 0], [200, 50], [50, 300], [300, 300]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (300, 300))
    plt.subplot(121)
    plt.imshow(img)
    plt.title('Input')
    for (x, y) in pts1:
        plt.scatter(x, y, s = 50, c = 'white', marker = 'x')
    plt.subplot(122)
    plt.imshow(dst)
    plt.title('Perspective Transform')
    for (x, y) in pts2:
        plt.scatter(x, y, s = 50, c = 'white', marker = 'x')
    plt.show()

#plot1()

# Crop image
pts1 = np.float32([[50, 50], [350, 50], [50, 350], [350, 350]])
M = cv2.getPerspectiveTransform(pts1, pts1)
dst = cv2.warpPerspective(img, M, (300, 300))
def plot2():
    plt.subplot(121)
    plt.imshow(img)
    plt.title('Input')
    for (x, y) in pts1:
        plt.scatter(x, y, s = 50, c = 'white', marker = 'x')
    plt.subplot(122)
    plt.imshow(dst)
    plt.title('Crop Image')
    plt.show()
#plot2()

# Làm mịn ảnh (smoothing images)
# Bộ lóc tích chập 2D (2D convolution)
kernel = np.ones((5, 5), np.float32) / 25
imgSmooth = cv2.filter2D(img, -1, kernel)
def plot4():
    plt.subplot(121)
    plt.imshow(img)
    plt.title("Original")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(imgSmooth)
    plt.title('Averaging')
    plt.xticks([])
    plt.yticks([])
    plt.show()

#plot4()
# Làm mở ảnh (Image blurring)
# 1. Trung bình (Average)
url = 'https://photo-2-baomoi.zadn.vn/w1000_r1/2019_05_10_351_30668071/06856d2daf6c46321f7d.jpg'
img = _downloadImage(url)
blur = cv2.blur(img, (5, 5))
def plot5():
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()

#plot5()

# 2. Bộ lọc Gaussian: Rất hiệu quả trong việc xóa bỏ noise khỏi hình ảnh
gaussian_img = cv2.GaussianBlur(img, (5, 5), 0)
def plot6():
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(gaussian_img), plt.title('Gaussian Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()

#plot6()

median_img = cv2.medianBlur(img, 5, 0)
bilateral_img = cv2.bilateralFilter(img, 9, 75, 75)
def plot7():
    plt.subplot(131), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(median_img), plt.title('Median Blurred')
    plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(bilateral_img), plt.title("Bilateral Blurred")
    plt.xticks([]), plt.yticks([])
    plt.show()

#plot7()

# Phương pháp Canny phát hiện Edge
url = 'https://i.pinimg.com/736x/6d/9c/e0/6d9ce08209b81b28c6ea64012e070003.jpg'
img = _downloadImage(url)
edges = cv2.Canny(img, 100, 200)
def plot8():
    plt.subplot(121), plt.imshow(img, cmap = "gray")
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap = 'gray')
    plt.title("Edge Image"), plt.xticks([]), plt.yticks([])

    plt.show()

#plot8()

#Contour - Xác định các contour
url = 'https://c4.wallpaperflare.com/wallpaper/279/111/762/close-up-pile-pipes-round-wallpaper-preview.jpg'
resp = requests.get(url)
img = np.asarray(bytearray(resp.content), dtype = "uint8")
img = cv2.imdecode(img, cv2.IMREAD_COLOR)
# Lọc ảnh nhị phân bằng thuật toán canny
imgCanny = cv2.Canny(img, 100, 200)
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Vì drawContours sẽ thay đổi ảnh gốc nên cần lưu ảnh sang một biến mới
imgOrigin = img.copy()
img1 = img.copy()
img2 = img.copy()
# Vẽ toàn bộ contours trên hình ảnh gốc
cv2.drawContours(img1, contours, -1, (0, 255, 0), 3)
# Vẽ chỉ contour thứ 4 trên hình ảnh gốc
cv2.drawContours(img2, contours, 100, (0, 255, 0),3)
def plot9():
    plt.figure(figsize = (12, 3))
    plt.subplot(141), plt.imshow(imgOrigin), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(142), plt.imshow(imgCanny), plt.title('Canny Binary Image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(img1), plt.title('All Contours')
    plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(img2), plt.title('Contour 4')
    plt.xticks([]), plt.yticks([])
    plt.show()

#plot9()

url = 'https://image.flaticon.com/icons/png/512/130/130188.png'
# Các đặc trưng của contour
img = _downloadImage(url)
# Khởi tạo các ảnh nhị phân canny
imgCanny = cv2.Canny(img, 100, 255)
# Tìm kiếm contours trên ảnh nhị phân từ bộ lọc canny
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Vẽ contour 0 trên hình ảnh gốc
img1 = img.copy()
cv2.drawContours(img1, contours, 0, (0, 255, 0), 5)
def plot10():
    plt.imshow(img1)
    plt.show()

#plot10()
# Lấy ra contour 0
cnt = contours[0]
M = cv2.moments(cnt)
#print('Moment values of contour 0: {}'.format(M))
# Tính toán tâm của contour 0
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
#print('Centroid position ({}, {})'.format(cx, cy))
# Vẽ biểu đồ tâm contour 0
def plot11():
    plt.imshow(img1)
    plt.scatter(cx, cy, s = 50, c = 'red', marker = 'o')
    plt.show()

#plot11()
area = cv2.contourArea(cnt)
print('area of contour 0: {}'.format(area))

# Bounding box
# Bounding box hình chữ nhật
img = _downloadImage('https://i0.wp.com/cdn-images-1.medium.com/max/2400/1*LmxW8FDfXZJl5yvESvjP7Q.jpeg?resize=660%2C373&ssl=1')
# Chuyển đổi sang ảnh gray
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Chuyển hình ảnh sang nhị phân
_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
def plot11():
    plt.imshow(img)
    plt.show()

# Tìm kiếm contours trên ảnh nhị phân
#plot11()
# Sử dụng hàm cv2.contourArea() để tìm ra diện tích các contours và sắp xếp thứ tự diện tích từ cao xuống thấp
# Tìm diện tích của toàn bộ các contour
area_cnt = [cv2.contourArea(cnt) for cnt in contours]
area_sort = np.argsort(area_cnt)[::-1]
# Top 5 contour có diện tích lớn nhất
print(area_sort[:5])

# Vẽ bounding box cho contours có diện tích lớn thứ 2
cnt = contours[area_sort[1]]
x, y, w, h = cv2.boundingRect(cnt)
print('CentroudL ({}, {}), (width, height): ({}, {})'.format(x, y, w, h))
img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
def plot12():
    plt.imshow(img)
    plt.show()

#plot12()

# Vẽ bounding box hình chữ nhật xoay diện tích nhỏ nhất bao quan một contour
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
img = cv2.drawContours(img, [box], 0, (100, 100, 100), 2)
def plot13():
    plt.imshow(img)
    plt.show()

#plot13()
# Vẽ hình tròn
(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
img = cv2.circle(img, center, radius, (0, 255, 0), 2)
# Hình Ellipse
# Lấy contour có diện tích lớn thứ 3
cnt = contours[area_sort[2]]
ellipse = cv2.fitEllipse(cnt)
img = cv2.ellipse(img, ellipse, (0, 255, 0), 2)

# Đường thẳng
rows, cols = img.shape[:2]
[vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x * vy/vx) + y)
righty = int(((cols - x)*vy/vx) + y)
img = cv2.line(img, (cols-1, righty), (0, lefty), (0, 255, 0), 2)

def plot14():
    plt.imshow(img)
    plt.show()

#plot14()

# Các thuộc tính của contour
# 1. Tỷ lệ cạnh (aspect ratio)
x, y, w, h = cv2.boundingRect(cnt)
aspect_ratio = float(w) / h
print("Aspect ratio: {}".format(aspect_ratio))
# 2. Độ phủ (extent)
# Tính diện tích của contour
area = cv2.contourArea(cnt)
# Tính diện tích bounding box
x, y, w, h = cv2.boundingRect(cnt)
rect_area = w * h
# Tính độ phủ
extent = float(area) / rect_area
print('ExtentL {}'.format(extent))

# 3. Độ cô đặc (Solidity)
area = cv2.contourArea(cnt)
# Khởi tạo một đa diện lồi bao quanh contour
hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)
# Tính toán độ cô đặc
solidity = float(area) / hull_area
print('Solidity: {}'.format(solidity))
# Đường kính tương đương (Equivalent Diameter)
area = cv2.contourArea(cnt)
equi_diameter = np.sqrt(4 * area / np.pi)
print('Equivalent Diameter: {}'.format(equi_diameter))

# Hướng Orientation

