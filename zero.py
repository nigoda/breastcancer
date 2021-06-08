import os
import cv2
import torch, matplotlib
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
# from matplotlib.offsetbox import AnchoredText
matplotlib.rc('image', cmap = 'gray')
import imageio
import pylab as pl
from PIL import Image
#from skimage import data
import pandas as pd

add_df = pd.DataFrame(columns = ['Id','diagnosis','radius_mean','perimeter_mean','area_mean','compactness','concavity_points','cx','cy','area','perimeter','concavity_mean','concave_points','fractal_dimension','x_mean','y_mean','radius_c','contours_circle','mean','std','smoothness'])

id = os.listdir("C:\\Users\\aksha\\PycharmProjects\\breastcancer\\IDC_regular_ps50_idx5")
file ="b1.csv"
for m in range(len(id)):
    Ids = id[m]
    d = "1"
    print(Ids)
    archive = os.listdir("C:\\Users\\aksha\\PycharmProjects\\breastcancer\\IDC_regular_ps50_idx5\\" + Ids + "\\" + d + "")
    m = 0
    for i in range(len(archive)):
        print(archive[i])
        image = archive[i]
        path = "C:\\Users\\aksha\\PycharmProjects\\breastcancer\\IDC_regular_ps50_idx5\\" + Ids + "\\" + d + "\\" + image + ""

        img = imageio.imread(path)
        # Transform into pytorch tensor.
        img = torch.tensor(img, dtype=torch.float) / 255.0

        # Show the image size.
        # print('Image size: ', img.shape)
        # display_image(img)

        # get some image
        # image = data.coins()
        image = img[:, 0:303]

        # create array of radii
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        R = np.sqrt(x ** 2 + y ** 2)

        # calculate the mean
        f = lambda r: image[(R >= r - 0.5) & (R < r + 0.5)].mean()
        r = np.linspace(1, 302, num=302)
        mean = np.vectorize(f)(r)

        # plot it
        # fig,ax=plt.subplots()
        # ax.plot(r,mean)
        # plt.show()

        x = np.array(mean)
        mean = x[np.logical_not(np.isnan(x))]
        # print(mean)
        # print(len(mean))
        radius_mean = 0
        for i in range(len(mean)):
            radius_mean = radius_mean + mean[i]
        radius_mean = radius_mean / len(mean)
        perimeter_mean = 2 * radius_mean * 3.14
        area_mean = 3.14 * radius_mean * radius_mean
        compactness = ((perimeter_mean * perimeter_mean) / (area_mean - 1.0))
        # print('radius_mean :',radius_mean)
        # print('perimeter_mean :' ,perimeter_mean)
        # print('area_mean :' ,area_mean)
        # print('compactness :' ,compactness)

        #####################################
        img = cv2.imread(path, 0)
        ret, thresh = cv2.threshold(img, 70, 200, 0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        # display_image(img)
        cnt = contours[0]
        M = cv2.moments(cnt)
        # print(M)
        concavity_points = len(M)
        # print("number of covcavity points : ",concavity_points)

        if M['m00'] == 0:
            M['m00'] = 0.001

        cx = (M['m10'] / M['m00'])
        cy = (M['m01'] / M['m00'])
        # print('cx : cy =',cx,':',cy)

        concavity_mean = (cx + cy) / 2
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        # print('area and perimeter of concavity ;',area,":", perimeter)
        # print('concavity_mean :',concavity_mean)

        ##############################

        # Load the image
        img1 = cv2.imread(path)
        # Convert it to greyscale
        img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # Threshold the image
        ret, thresh = cv2.threshold(img, 150, 255, 0)
        # Find the contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # For each contour, find the convex hull and draw it
        # on the original image.

        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            cv2.drawContours(img1, [hull], -1, (255, 0, 0), 2)

        # Display the final convex hull image
        # cv2.imshow('ConvexHull', img1)
        # cv2.waitKey(0)
        # def display_image(img):
        #   plt.figure(); plt.imshow(img)
        #   plt.grid(False);  plt.axis('off'); plt.show()
        # display_image(img1)
        concave_points = len(contours)


        # print('concave points:',concave_points)

        ###############################
        def rgb2gray(rgb):
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray


        image = rgb2gray(pl.imread(path))

        # finding all the non-zero pixels
        pixels = []
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] > 0:
                    pixels.append((i, j))

        Lx = image.shape[1]
        Ly = image.shape[0]
        # print(Lx, Ly)
        pixels = pl.array(pixels)
        # print(pixels.shape)

        # computing the fractal dimension
        # considering only scales in a logarithmic list
        scales = np.logspace(0.01, 1, num=10, endpoint=False, base=2)
        Ns = []
        # looping over several scales
        for scale in scales:
            # print("======= Scale :", scale)
            # computing the histogram
            H, edges = np.histogramdd(pixels, bins=(np.arange(0, Lx, scale), np.arange(0, Ly, scale)))
            Ns.append(np.sum(H > 0))

        # linear fit, polynomial of degree 1
        coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)

        # pl.plot(np.log(scales), np.log(Ns), 'o', mfc='none')
        # pl.plot(np.log(scales), np.polyval(coeffs, np.log(scales)))
        # pl.xlabel('log $\epsilon$')
        # pl.ylabel('log N')
        # pl.savefig('sierpinski_dimension.pdf')
        fractal_dimension = -coeffs[0]
        # print("The Hausdorff dimension is",
        #       fractal_dimension)  # the fractal dimension is the OPPOSITE of the fitting coefficient
        # np.savetxt("scaling.txt", list(zip(scales, Ns)))

        #############################
        thresh = cv2.imread(path, 0)
        # thresh = thresh[:,0:303]
        contours, hierarchy = cv2.findContours(thresh, 2, 1)

        cnt = contours
        x_mean = 0
        y_mean = 0
        radius_c = 0
        for i in range(len(cnt)):
            (x, y), radius = cv2.minEnclosingCircle(cnt[i])
            c = (int(x), int(y))
            r = int(radius)
            x_mean = x_mean + int(x)
            y_mean = y_mean + int(y)
            radius_c = radius_c + int(radius)

            # print('Circle: ' + str(i) + ' - Center: ' + str(c) + ' - Radius: ' + str(r))
        x_mean = x_mean / len(cnt)
        y_mean = y_mean / len(cnt)
        radius_c = radius_c / len(cnt)
        center = (int(x_mean), int(y_mean))
        radius = int(radius_c)
        cv2.circle(thresh, center, radius, (0, 255, 0), 2)
        # plt.text(x_mean - 10, y_mean + 6, '+', fontsize=25, color='red')
        # plt.text(10, -10, 'Centro: ' + str(center), fontsize=11, color='red')
        # plt.text(340, -10, 'Diametro: ' + str((radius * 2) / 100) + 'mm', fontsize=11, color='red')
        # plt.Circle((10, -10), 7.2, color='blue')
        # plt.imshow(thresh, cmap='gray')
        # # plt.savefig(IMG+'-diam.png')
        # plt.show()
        # print(x_mean, y_mean, radius_c)
        contours_circle = len(contours)
        # print(contours_circle)
        ##########################
        # load image
        image = Image.open(path)
        pixels = asarray(image)
        # convert from integers to floats
        pixels = pixels.astype('float32')
        # calculate global mean and standard deviation
        mean, std = pixels.mean(), pixels.std()
        # print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
        # global standardization of pixels
        # pixels = (pixels - mean) / std
        # confirm it had the desired effect
        # mean, std = pixels.mean(), pixels.std()
        # print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
        # display_image(image)
        ##########################
        import numpy as np
        import scipy.ndimage as ndi
        import imageio

        # path = 'C:\\Users\\aksha\\PycharmProjects\\breastcancer\\8863_idx5_x151_y1351_class0.png'
        smoothness = np.average(np.absolute(ndi.filters.laplace(imageio.imread(path).astype(float) / 255.0)))

        ########

        # print('Image size                        :', img.shape[0], "*", img.shape[1])
        # resolution = img.shape[0] * img.shape[1]
        # print('Image_resolution                  :', resolution)
        # print('radius_mean                       :', radius_mean)
        # print('perimeter_mean                    :', perimeter_mean)
        # print('area_mean                         :', area_mean)
        # print('compactness                       :', compactness)
        # print('number of covcavity points        :', concavity_points)
        # print('cx : cy                           :', cx, ':', cy)
        # print('area and perimeter of concavity   :', area, ":", perimeter)
        # print('concavity_mean                    :', concavity_mean)
        # print('concave points                    :', concave_points)
        # print('The Hausdorff dimension is        :', fractal_dimension)
        # print('contours of x : y : radius        :', x_mean, ":", y_mean, ":", radius_c)
        # print('number of contours cirlce         :', contours_circle)
        # print('Mean texture: %.3f              Standard Deviation texture: %.3f' % (mean, std))
        # print('Smoothness value                  :', smoothness)

        data = [[Ids, d, radius_mean, perimeter_mean, area_mean, compactness, concavity_points, cx, cy, area, perimeter,
                 concavity_mean, concave_points, fractal_dimension, x_mean, y_mean, radius_c, contours_circle, mean,
                 std,
                 smoothness]]
        df = pd.DataFrame(data, columns=['Id', 'diagnosis', 'radius_mean', 'perimeter_mean', 'area_mean', 'compactness',
                                         'concavity_points', 'cx', 'cy', 'area', 'perimeter', 'concavity_mean',
                                         'concave_points', 'fractal_dimension', 'x_mean', 'y_mean', 'radius_c',
                                         'contours_circle', 'mean', 'std', 'smoothness'])

        frames = [add_df, df]
        add_df = pd.concat(frames)
        add_df = add_df.reset_index(drop=True)
        add_df.to_csv(file)
    print(add_df.head())

