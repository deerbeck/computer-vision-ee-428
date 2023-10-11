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


with rawpy.imread(args.path) as raw:
    # raw_image contains the raw image data in 16-bit integer format.
    raw_image = raw.raw_image

    # demosaic raw image
    demosaic_image = demosaic(raw_image)

    # white balance demosaiced image
    white_balance_image = white_balance(demosaic_image)

    #apply gamma, and quantize white balanced image
    curve_quan_image = curve_and_quantize(white_balance_image)

    #safe picture as PNG and JPG
    imageio.imwrite(path_out_JPG, curve_quan_image)
    imageio.imwrite(path_out_PNG, curve_quan_image)
