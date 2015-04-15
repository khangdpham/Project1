import sys
from MTB import *
from curve import *
from tone_mapping import *
import timeit
from photo_map import *
#from photo_map import *

#------ Main function
# argv = ref_frame, level, radius, sigma_s, sigma_r, contrast, dataset
if( len(sys.argv) != 8 ) :
    print 'hdr <ref_frame> <MTB_level> <radius> <sigma_s> <sigma_r> <contrast: suggest 50~200> <dataset_path>'
    quit()
start = timeit.default_timer()
ref_frame = int(sys.argv[1])
level = int(sys.argv[2])
radius = int(sys.argv[3])
sigma_s = float(sys.argv[4])
sigma_r = float(sys.argv[5])
contrast = float(sys.argv[6])
data_set = sys.argv[7]
bf_output = 'bilateral_HDR_' + str(contrast) + '.jpg'

img, exp_time = align(data_set, ref_frame, level)
result = solveCurve(img, exp_time)
E = radianceMap(img, exp_time, result)
#Photo_tone(E)
# Our Bilateral Filter
# tone_map(E, radius, sigma_s, sigma_r, direct_BF, contrast, bf_output)
# opencv Bilateral Filter
tone_map(E, radius, sigma_s, sigma_r, opencv_BF, 50, 'bilateral_HDR_50.jpg')
tone_map(E, radius, sigma_s, sigma_r, opencv_BF, 100, 'bilateral_HDR_100.jpg')
tone_map(E, radius, sigma_s, sigma_r, opencv_BF, 150, 'bilateral_HDR_150.jpg')
tone_map(E, radius, sigma_s, sigma_r, opencv_BF, 200, 'bilateral_HDR_200.jpg')

end = timeit.default_timer()
print 'time = ', end - start
