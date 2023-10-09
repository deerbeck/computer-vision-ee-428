import rawpy
from raw import *
import argparse
import imageio
import os

parser = argparse.ArgumentParser()
parser.add_argument('path',help='path to DNG file')
args = parser.parse_args()

# this is the path to the output JPEG
path_out_JPG = os.path.basename(args.path).split('.')[0]+'.JPG'

# this is the path to the output PNG
path_out_PNG = os.path.basename(args.path).split('.')[0]+'.PNG'


# just for testing:
# path = "L1004220.DNG"
# path_out1 = "L100422011.PNG"
# path_out2 = "L100422011.JPG"


with rawpy.imread(args.path) as raw:
    # raw_image contains the raw image data in 16-bit integer format.
    raw_image = raw.raw_image

    # demosaic raw image
    demosaic_im = demosaic(raw_image)

    # white balance demosaiced image
    white_balance_im = white_balance(demosaic_im)

    #apply gamma, and quantize white balanced image
    curve_quan_im = curve_and_quantize(white_balance_im)

    #safe picture as PNG and JPG
    imageio.imwrite(path_out_JPG, curve_quan_im)
    imageio.imwrite(path_out_PNG, curve_quan_im)
