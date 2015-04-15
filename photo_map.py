import cv2
import numpy
import math
import curve


### threshold value
### result are marked by negative sign when passed threshold
def threshold(result, delta, v1):
    eps = 0.05
    if result < 0: 
        return result
    elif delta > eps:
        result = -v1
    else:
        result = 0
    return result


### handle max itr and reverse negetive
def handle_max(result, v):
    if result == 0 :
        result = v
    elif result < 0:
        result = abs(result)
    else:
        print "Warning!!!! result invalid!!, result = ", result
    return result
        

### tone mapping(Photographic)
### second approach
def Photo_tone(img_bgr):
    print 'starting tone mapping...'
    print img_bgr.shape

    ### img_bgr is E
    

    ### convert BGR to Yxy
    X = 0.488718*img_bgr[2] + 0.310680*img_bgr[1] + 0.200602*img_bgr[0]
    Y = 0.176204*img_bgr[2] + 0.812985*img_bgr[1] + 0.010811*img_bgr[0]
    Z = 0.000000*img_bgr[2] + 0.010205*img_bgr[1] + 0.989795*img_bgr[0]
    img_xyz = numpy.array([X, Y, Z])
    W = img_xyz[0] + img_xyz[1] + img_xyz[2]
    img_yxy = numpy.array( [img_xyz[1], img_xyz[0]/W, img_xyz[1]/W] )

    lum_exp = img_yxy[0]
    lum = numpy.log(lum_exp+1)
    
    ### key
    key = 1.0
    phi = 8
    ### size of gaussian blur
    g_w = 41 
    g_h = 41 
    ### threshold
    ### max itr
    max_itr = 6

    row = len(img_bgr[0])
    col = len(img_bgr[0][0])

    ### scale to mid tone
    lum_avg = math.exp(numpy.sum(lum) / (row*col))
    lum_exp = lum_exp*key/lum_avg



    ### blur size parameter
    alpha_1 = 0.35
    alpha_2 = 0.51


    ### result V
    v_result = numpy.zeros((row, col))
    V1_gaussian = numpy.zeros((max_itr, row, col))
    V2_gaussian = numpy.zeros((max_itr, row, col))
    delta_V = numpy.zeros((max_itr, row, col))
    sigma = 0.7 
    for itr in range(max_itr):

        V1_gaussian[itr] = cv2.GaussianBlur(lum_exp, (g_w, g_h), sigma*alpha_1)
        V2_gaussian[itr] = cv2.GaussianBlur(lum_exp, (g_w, g_h), sigma*alpha_2)
        delta_V[itr] = (V1_gaussian[itr]-V2_gaussian[itr])/ ((2**phi)/sigma**2 + V1_gaussian[itr])
        sigma *= 1.1
            
        print "itr ", itr, 'V1max ', numpy.amax(V1_gaussian[itr]), \
                'V2Max', numpy.amax(V2_gaussian[itr]), 'delMax', \
                numpy.amax(delta_V[itr])

    threshold_vec = numpy.vectorize(threshold)
    for itr in range(max_itr):
        v_result = threshold_vec(v_result, delta_V[itr], V1_gaussian[itr])
        print 'result_max ', numpy.amin(v_result)

    ### this is for those delta_v that are not above threshold at max itr
    handle_max_vec = numpy.vectorize(handle_max)
    v_result = handle_max_vec(v_result, V1_gaussian[max_itr-1])
    print 'result_max ', numpy.amax(v_result)


    print 'lum max ', numpy.amax(lum_exp)
        
            

    ### lower constrast
    img_yxy[0] = lum_exp/(1+v_result)

    ### Yxy to BGR
    img_xyz = numpy.array([img_yxy[1]*img_yxy[0]/img_yxy[2], img_yxy[0], (1-img_yxy[1]-img_yxy[2])*(img_yxy[0]/img_yxy[2])])
    B =  0.005298*img_xyz[0] - 0.014695*img_xyz[1] + 1.009397*img_xyz[2]
    G = -0.513885*img_xyz[0] + 1.425304*img_xyz[1] + 0.088581*img_xyz[2]
    R =  2.370674*img_xyz[0] - 0.900041*img_xyz[1] - 0.470634*img_xyz[2] 
    img_bgr = numpy.array([B, G, R])
    for i in range(3):
        print 'BGR max ', numpy.amax(img_bgr[i]), numpy.average(img_bgr[i])

    ### clamp image
    img_bgr[img_bgr>1] = 1 
    img_bgr[img_bgr<0] = 0

    ### test
    img_bgr = numpy.power(img_bgr, 1/1.5)
    img_bgr *= 255 


    cv2.imwrite('result_photo.jpg', cv2.merge(img_bgr))
