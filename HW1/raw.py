import numpy as np
import matplotlib.pyplot as plt
import cv2

def demosaic(raw):
    """ Demosaics a raw 16-bit image captured using a Bayer pattern.
        Arguments:
          raw: the input raw data in 16-bit integer [HxW]
        Returns:
          The demosaiced image in 32-bit floating point [HxWx3]
    """
    # Step 1 convert to floating point and normalize image with max val of 16-bit integer
    demosaic_im = raw.astype('float')/((2**16)-1)

    # # # SAMPLE ARRAY
    demosaic_im = np.array(
        [[0.2, 0.1, 0.3, 0.2,0.1], [0.12, 0.8, 0.4,0.3,0.1], [0.4, 0.38, 0.6,0.2,0.1], [0.8, 0.28, 0.76,0.1,0.2]])

    # Step 2 Demosaic the red green and blue channel
    # Pad the current raw_image
    demosaic_im = np.pad(demosaic_im, pad_width=(1, 1), mode="reflect")

    # Compute green channel
    green_ch = np.copy(demosaic_im)
    # loop through each column/row and calculate the missing green pixels as a interpolation of the bayer pattern
    for row in range(1, green_ch.shape[0]-1):
        # for odd rows start with column index 1
        if row % 2 != 0:
            for col in range(1, green_ch.shape[1]-1, 2):
                green_ch[row][col] = (demosaic_im[row-1][col] + demosaic_im[row+1]
                                      [col] + demosaic_im[row][col-1] + demosaic_im[row][col+1])/4
        # for even rows start with column index 0
        else:
            for col in range(0, green_ch.shape[1]-1, 2):
                green_ch[row][col] = (demosaic_im[row-1][col] + demosaic_im[row+1]
                                      [col] + demosaic_im[row][col-1] + demosaic_im[row][col+1])/4
    # cut off edges because no longer needed
    green_ch = green_ch[1:-1, 1:-1]

    # Compute red channel
    red_ch = np.copy(demosaic_im)
    # interpolate missing columns first
    for row in range(1, red_ch.shape[0], 2):
        for col in range(2, red_ch.shape[1]-1, 2):
            red_ch[row][col] = (demosaic_im[row][col-1] +
                                demosaic_im[row][col+1])/2
    # now use fully filled rows to interpolate the missing rows
    for row in range(2, red_ch.shape[0]-1, 2):
        for col in range(0, red_ch.shape[1]-1):
            red_ch[row][col] = (red_ch[row-1][col] + red_ch[row+1][col])/2
    # cut off edges because no longer needed
    red_ch = red_ch[1:-1, 1:-1]

    # Compute blue channel
    blue_ch = np.copy(demosaic_im)
    # interpolate missing columns first
    for row in range(0, blue_ch.shape[0], 2):
        for col in range(1, blue_ch.shape[1]-1, 2):
            blue_ch[row][col] = (demosaic_im[row][col-1] +
                                 demosaic_im[row][col+1])/2
    # now use fully filled rows to interpolate the missing rows
    for row in range(1, blue_ch.shape[0]-1, 2):
        for col in range(0, blue_ch.shape[1]-1):
            blue_ch[row][col] = (blue_ch[row-1][col] + blue_ch[row+1][col])/2
    # cut off edges because no longer needed
    blue_ch = blue_ch[1:-1, 1:-1]

    return np.dstack([red_ch, green_ch, blue_ch])   # np stack axis = -1 is the same


def white_balance(image):
    """ White balanaces a 32-bit floating point demosaiced image.
        This is done by simply scaling each channel so that its mean = 0.5.
        Arguments:
          image: the input image in 32-bit floating point [HxWx3]
        Returns:
          The white balanced image in 32-bit floating point [HxWx3]
    """   
    # splitting the image into the three colour channels
    channels = cv2.split(image)

    # Step 3 Apply white balancing by scaling each channel so that its mean is 0.5
    # white balancing red channel
    red_ch = channels[0] * 0.5/(np.mean(channels[0]))
    
    # white balancing green channel
    green_ch = channels[1] * 0.5/(np.mean(channels[1]))
    
    # white balancing blue channel
    blue_ch = channels[2] * 0.5/(np.mean(channels[2]))
    
    return np.dstack([red_ch, green_ch, blue_ch]) # np stack axis = -1 is the same


def curve_and_quantize(image, inv_gamma=0.85):
    """ Applies inverse gamma function and quantizes to 8-bit.
        Arguments:
          image: the input image in 32-bit floating point [HxWx3]
          inv_gamma: the value of 1/gamma
        Returns:
          The curved and quantized image in 8-bit unsigned integer [HxWx3]
    """
    # splitting the image into the three colour channels
    channels = cv2.split(image)
    
    # apply gamma curve to red channel, clip it, scale it with 255 and safe as uint8
    red_ch= (np.clip(channels[0]**(inv_gamma),0,1)*255).astype('uint8')

    # apply gamma curve to green channel, clip it, scale it with 255 and safe as uint8
    green_ch = (np.clip(channels[1]**(inv_gamma),0,1)*255).astype('uint8')

    # apply gamma curve to blue channel, clip it, scale it with 255 and safe as uint8
    blue_ch = (np.clip(channels[2]**(inv_gamma),0,1)*255).astype('uint8')
    
    return np.dstack([red_ch, green_ch, blue_ch]) # np stack axis = -1 is the same
