import cv2
from PIL import Image
from PIL.ExifTags import TAGS
import glob
import sys
import numpy
import random
import math
import time
import pdb

### for plotting
import matplotlib.pyplot as plt


#point1 = numpy.loadtxt('points_pick')     ### point coordinate
#point2 = numpy.loadtxt('points')
#point = numpy.append(point1, point2, axis=0)
#points = len(point)                 ### points in each images

smoothness = math.sqrt(10000)        ### curve smoothness
points = 100

### random points gen
def point_gen(r_max, c_max, points):
    out = numpy.zeros((points, 2))
    for i in range(points):
         row = int(random.random()*r_max)
         col = int(random.random()*c_max)
         out[i] = (row, col)
    return out


### weight function 
def w(z):
    if z<0 : return 0
    if z==0 : return 0.01
    if z>255 : return 255
    if z==255: return 0.01
    if z>128 : return 255-z
    return z

### divide function ( prevent weight sum = 0 )
def divide(rad, w_sum):
    if w_sum == 0:
        return rad
    return rad/w_sum

### simple tone mapping
def sim_tone_map(img, m_val):
    ### from (min, max) -> (0, max-min)
    min_val = numpy.amin(img)
    img -= min_val
    ### from (0, max-min) -> (0, 255)
    max_val = numpy.amax(img)
    img /= max_val 
    img *= m_val 
    ### truncate to int
    img.astype(numpy.int64)

    return numpy.array(img) 

### save HDR image
def saveHDR(image, filename):
    f = open(filename, "wb")
    f.write("-Y {0} +X {1}\n".format(image.shape[0], image.shape[1]))

    brightest = numpy.maximum(numpy.maximum(image[...,0], image[...,1]), image[...,2])
    mantissa = numpy.zeros_like(brightest)
    exponent = numpy.zeros_like(brightest)
    numpy.frexp(brightest, mantissa, exponent)
    scaled_mantissa = mantissa * 256.0 / brightest
    rgbe = numpy.zeros((image.shape[0], image.shape[1], 4), dtype=numpy.uint8)
    rgbe[...,0:3] = numpy.around(image[...,0:3] * scaled_mantissa[...,None])
    rgbe[...,3] = numpy.around(exponent + 128)

    rgbe.flatten().tofile(f)
    f.close()


### solve non-linear curve
### images should be in path with jpg format
def solveCurve( cv_imgs, exp_time ):
    num = len(cv_imgs)
    cv_imgs = numpy.array(cv_imgs)
    cv_imgs = numpy.swapaxes(cv_imgs, 2, 3)
    cv_imgs = numpy.swapaxes(cv_imgs, 1, 2)
    row_max = len(cv_imgs[0][0])
    col_max = len(cv_imgs[0][0][0])
    point = point_gen(row_max, col_max, points) 
    print 'solving curve using ', num, ' pics..., points = ', points, ' lambda = ', smoothness
   
    ### empty matrix
    result = [ numpy.zeros((255+points, 1)), numpy.zeros((255+points, 1)), numpy.zeros((255+points, 1)) ]
    tmp = numpy.zeros((256+points, 1))
    
    ### fill in the coefficients
    for color in [0, 1, 2]:
        mat_a = numpy.zeros((num*points+255, 256+points))
        mat_b = numpy.zeros((num*points+255, 1))
        for pic in range(0, num):
            for idx, pt in enumerate(point):
                z = cv_imgs[pic][color][pt[0]][pt[1]]
                weight = math.sqrt(w(z))
                mat_a[pic*points+idx][z] = weight
                mat_a[pic*points+idx][256+idx] = -1*weight
                mat_b[pic*points+idx][0] = weight*math.log(exp_time[pic])

        ### g(127) = 0
        mat_a[num*points][127] = 1

        ### for smoothness
        for smooth in range(1, 254):
            mat_a[num*points+smooth][smooth-1] = smoothness*w(smooth)
            mat_a[num*points+smooth][smooth] = -2*smoothness*w(smooth)
            mat_a[num*points+smooth][smooth+1] = smoothness*w(smooth)
            #mat_a[num*points+smooth][smooth-1] = smoothness
            #mat_a[num*points+smooth][smooth] = -2*smoothness
            #mat_a[num*points+smooth][smooth+1] = smoothness

        err, result[color] = cv2.solve(mat_a, mat_b, result[color], cv2.DECOMP_SVD )


        
    plt.plot(result[0][0:256], 'bo', result[1][0:256], 'go', result[2][0:256], 'ro')
    plt.show()

    result = numpy.array(result)
    return result[0:3, 0:256] 

### using reconstruct curve
def radianceMap(cv_imgs, exp_time, curve):
    num = len(cv_imgs)

    cv_imgs = numpy.array(cv_imgs, dtype='int64')
    cv_imgs = numpy.swapaxes(cv_imgs, 2, 3)
    cv_imgs = numpy.swapaxes(cv_imgs, 1, 2)
    row = len(cv_imgs[0][0])
    col = len(cv_imgs[0][0][0])
    rad = [  numpy.zeros((row, col)), numpy.zeros((row, col)), numpy.zeros((row, col)) ]


    for color in [0, 1, 2]:
        curve_np = numpy.asarray(curve[color])
        curve_np = curve_np.reshape(256)
        w_sum = numpy.zeros((row, col))
        for pic in range(num):
            w_vec = numpy.vectorize(w)
            ### from (col, row, 1) to (col, row)
            z_refactor = numpy.reshape(curve_np[cv_imgs[pic][color]], (row, col))
            rad[color] += w_vec(cv_imgs[pic][color])*( z_refactor - math.log(exp_time[pic]) )
            w_sum += w_vec(cv_imgs[pic][color])

        divide_vec = numpy.vectorize(divide)
        rad[color] = divide_vec(rad[color] , w_sum)


    ### merge 3 channels back
    hdr = numpy.exp(rad)
    numpy.savetxt('hdrb', hdr[0])
    numpy.savetxt('hdrg', hdr[1])
    numpy.savetxt('hdrr', hdr[2])
    return hdr

