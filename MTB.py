import cv2
import cv
import numpy as np
from PIL import Image
import os

def shift(I, bias_x, bias_y, l):
    rows, cols, ch = I.shape
    shift_I = np.zeros((I.shape))
    for color in range(3):
        if bias_x==-1 and bias_y==-1:
            shift_I[0:rows+bias_y*l, 0:cols+bias_x*l, color] = I[-bias_y*l:rows, -bias_x*l:cols, color]
        elif bias_x==0 and bias_y==-1:
            shift_I[0:rows+bias_y*l ,0:cols, color] = I[-bias_y*l:rows, 0:cols, color]
        elif bias_x==1 and bias_y==-1:
            shift_I[0:rows+bias_y*l, bias_x*l:cols, color] = I[-bias_y*l:rows, 0:cols-bias_x*l, color]
        elif bias_x==-1 and bias_y==0:
            shift_I[0:rows, 0:cols+bias_x*l, color] = I[0:rows, -bias_x*l:cols, color]
        elif bias_x==1 and bias_y==0:
            shift_I[0:rows, bias_x*l:cols, color] = I[0:rows, 0:cols-bias_x*l, color]
        elif bias_x==-1 and bias_y==1:
            shift_I[bias_y*l:rows, 0:cols+bias_x*l, color] = I[0:rows-bias_y*l, -bias_x*l:cols, color]
        elif bias_x==0 and bias_y==1:
            shift_I[bias_y*l:rows, 0:cols, color] = I[0:rows-bias_y*l, 0:cols, color]
        elif bias_x==1 and bias_y==1:
            shift_I[bias_y*l:rows, bias_x*l:cols, color] = I[0:rows-bias_y*l,0:cols-bias_x*l, color]
        else:
            shift_I[:,:, color] = I[:,:, color]
    return shift_I

def diff(a, b, bias_x, bias_y):
    rows, cols = a.shape
    y_begin = max(0, bias_y)
    y_end = min(rows, rows+bias_y-1)
    x_begin = max(0, bias_x)
    x_end = min(cols, cols+bias_x-1)
    if bias_x==-1 and bias_y==-1:
        shift_b = b[1:rows, 1:cols]
    elif bias_x==0 and bias_y==-1:
        shift_b = b[1:rows, 0:cols]
    elif bias_x==1 and bias_y==-1:
        shift_b = b[1:rows, 0:cols-1]
    elif bias_x==-1 and bias_y==0:
        shift_b = b[0:rows, 1:cols]
    elif bias_x==1 and bias_y==0:
        shift_b = b[0:rows, 0:cols-1]
    elif bias_x==-1 and bias_y==1:
        shift_b = b[0:rows-1, 1:cols]
    elif bias_x==0 and bias_y==1:
        shift_b = b[0:rows-1,0:cols]
    elif bias_x==1 and bias_y==1:
        shift_b = b[0:rows-1,0:cols-1]
    else:
        shift_b = b
    cost = np.bitwise_xor(a[y_begin:(y_end+1), x_begin:(x_end+1)], shift_b)
    cost = np.sum(cost)
    w = np.ones(( (y_end-y_begin+1), (x_end-x_begin+1) ))
    w = np.sum(w)

    return float(cost)/float(w)


def align(data_set, ref_frame, level):
    #----- Load Image
    img = []
    img_Y = []
    exp_time = []
    for filename in os.listdir(data_set):
        if filename.endswith(".jpg"):
            im = Image.open(data_set + filename)
            imgInfo = im._getexif();
            for tag, value in imgInfo.items():
                if( tag == 33434 ):
                    exp_time.append(value[0]/float(value[1]))
                    break
            image = np.array(im)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img.append(image)
    
            rows, cols, ch = image.shape
            Y = cv2.cvtColor(image,cv.CV_BGR2GRAY)
            img_Y.append(Y)
    print 'Finding median thresholding value...'
    #----- Median thresholding
    threshold = []
    for image in img_Y:
        histo = {}
        rows, cols = image.shape
        for row in range(rows):
            for col in range(cols):
                color = image[row, col]
                if color in histo:
                    histo[color] += 1
                else:
                    histo[color] = 1
        target = rows*cols/2
        acc = 0
        for color, freq in histo.iteritems():
            if (freq+acc) >= target:
                threshold.append(color)
                break
            else:
                acc += freq
    
    print 'color converting...'
    #----- BGR to BW 
    img_BW = []
    for idx in range(len(img_Y)):
        rows, cols = img_Y[idx].shape
        BW = cv2.threshold(img_Y[idx], threshold[idx], 255, cv2.THRESH_BINARY)[1]
        img_BW.append(BW)
    
    print 'Start alignment...'
    #----- Alignment
    for idx in range(len(img_BW)):
        if idx == ref_frame:
            continue
        rows, cols = img_BW[ref_frame].shape
        ref = cv2.resize(img_BW[ref_frame], (int(rows*pow(2,-(level-1))), int(cols*pow(2,-(level-1)))) )
        current = cv2.resize(img_BW[idx], (int(rows*pow(2,-(level-1))), int(cols*pow(2,-(level-1)))) )
        for l in range(level): 
            # top-left
            min_cost = diff(ref,current,-1,-1)
            direc = [-1,-1]
            # top
            tmp_cost = diff(ref,current,0,-1)
            if (tmp_cost < min_cost):
                min_cost = tmp_cost
                direc = [0,-1]
            # top-right
            tmp_cost = diff(ref,current,1,-1)
            if (tmp_cost < min_cost):
                min_cost = tmp_cost
                direc = [1,-1]
            # left
            tmp_cost = diff(ref,current,-1,0)
            if (tmp_cost < min_cost):
                min_cost = tmp_cost
                direc = [-1,0]
            # middle
            tmp_cost = diff(ref,current,0,0)
            if (tmp_cost < min_cost):
                min_cost = tmp_cost
                direc = [0,0]
            # right
            tmp_cost = diff(ref,current,1,0)
            if (tmp_cost < min_cost):
                min_cost = tmp_cost
                direc = [1,0]
            # bottom-left
            tmp_cost = diff(ref,current,-1,1)
            if (tmp_cost < min_cost):
                min_cost = tmp_cost
                direc = [-1,1]
            # bottom
            tmp_cost = diff(ref,current,0,1)
            if (tmp_cost < min_cost):
                min_cost = tmp_cost
                direc = [0,1]
            # bottom-right
            tmp_cost = diff(ref,current,1,1)
            if (tmp_cost < min_cost):
                min_cost = tmp_cost
                direc = [1,1]
            
            #shift(img_BW[idx], direc[0], direc[1])
            img[idx] = shift(img[idx], direc[0], direc[1], pow(2,level-l-1))
            
            rows, cols = ref.shape
            ref = cv2.resize(ref, (int(rows*2), int(cols*2)) )
            current = cv2.resize(current, (int(rows*2), int(cols*2)) )
            '''
            for row in range(rows):
                for col in range(cols):
                    if row-direc[0] >= rows or row-direc[0] < 0 or col-direc[1] >= cols or col-direc[1] < 0:
                        img_BW[idx][row,col] = 0
                        for color in range(3):
                            img[idx][row,col][color] = 0
                    else:
                        img_BW[idx][row,col] = img_BW[idx][row-direc[0],col-direc[1]]
                        for color in range(3):
                            img[idx][row,col][color] = img[idx][row-direc[0],col-direc[1]][color]
            '''
        # Write aligned image into JPG file
        filename = 'align_img' + str(idx+1).zfill(2) + '.jpg'
        cv2.imwrite(filename, img[idx])
    #   cv2.namedWindow("alignment")
    #   cv2.imshow("alignment", image)
    #   cv2.waitKey(0)
    #   cv2.destroyAllWindows()
    #img = np.array(img)
    #img = np.swapaxes(img, 2,3)
    #img = np.swapaxes(img, 1,2)
#   aligned_img = Image.open(filename)
#   aligned_img.save(filename, exif=imgInfo)
    return img , exp_time


def main():
    img, exp_time = align(10, 5, 5)

if __name__ == '__main__':
    main()
