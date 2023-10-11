import numpy as np
import matplotlib.pyplot as plt

def demosaic(raw):
    """ Demosaics a raw 16-bit image captured using a Bayer pattern.
        Arguments:
          raw: the input raw data in 16-bit integer [HxW]
        Returns:
          The demosaiced image in 32-bit floating point [HxWx3]
    """
    # Step 1 convert to floating point and normalize image with max val of 16-bit integer
    mosaic_image = raw.astype('float')/((2**16)-1)

    # Step 2 Demosaic the red green and blue channel
    # Pad the current raw_image
    mosaic_image = np.pad(mosaic_image, pad_width=(1, 1), mode="reflect")

    # Compute green channel
    green_channel = np.copy(mosaic_image)
    # loop through each column/row and calculate the missing green pixels as a average of the neigboring pixels
    # according ti the bayer pattern
    for row in range(1, green_channel.shape[0]-1):
        # for odd rows start with column index 1
        if row % 2 != 0:
            for col in range(1, green_channel.shape[1]-1, 2):
                green_channel[row][col] = (mosaic_image[row-1][col] + mosaic_image[row+1]
                                      [col] + mosaic_image[row][col-1] + mosaic_image[row][col+1])/4
        # for even rows start with column index 0
        else:
            for col in range(0, green_channel.shape[1]-1, 2):
                green_channel[row][col] = (mosaic_image[row-1][col] + mosaic_image[row+1]
                                      [col] + mosaic_image[row][col-1] + mosaic_image[row][col+1])/4
    # cut off edges to undo the padding
    green_channel = green_channel[1:-1, 1:-1]

    # Compute red channel
    red_channel = np.copy(mosaic_image)
    # interpolate missing columns first
    for row in range(1, red_channel.shape[0], 2):
        for col in range(2, red_channel.shape[1]-1, 2):
            red_channel[row][col] = (mosaic_image[row][col-1] +
                                mosaic_image[row][col+1])/2
    # now use fully filled rows to interpolate the missing rows
    for row in range(2, red_channel.shape[0]-1, 2):
        for col in range(0, red_channel.shape[1]-1):
            red_channel[row][col] = (red_channel[row-1][col] + red_channel[row+1][col])/2
    # cut off edges to undo the padding
    red_channel = red_channel[1:-1, 1:-1]

    # Compute blue channel
    blue_channel = np.copy(mosaic_image)
    # interpolate missing columns first
    for row in range(0, blue_channel.shape[0], 2):
        for col in range(1, blue_channel.shape[1]-1, 2):
            blue_channel[row][col] = (mosaic_image[row][col-1] +
                                 mosaic_image[row][col+1])/2
    # now use fully filled rows to interpolate the missing rows
    for row in range(1, blue_channel.shape[0]-1, 2):
        for col in range(0, blue_channel.shape[1]-1):
            blue_channel[row][col] = (blue_channel[row-1][col] + blue_channel[row+1][col])/2
    # cut off edges to undo the padding
    blue_channel = blue_channel[1:-1, 1:-1]

    # stack image to return as 3D-array
    # np stack axis = -1 is the same
    return np.dstack([red_channel, green_channel, blue_channel])


def white_balance(image):
    """ White balanaces a 32-bit floating point demosaiced image.
        This is done by simply scaling each channel so that its mean = 0.5.
        Arguments:
          image: the input image in 32-bit floating point [HxWx3]
        Returns:
          The white balanced image in 32-bit floating point [HxWx3]
    """
    # splitting the image into the three colour channels
    channels = np.split(image, image.shape[-1], axis=2)

    # Step 3 Apply white balancing by scaling each channel so that its mean is 0.5
    # white balancing red channel
    red_channel = channels[0] * 0.5/(np.mean(channels[0]))

    # white balancing green channel
    green_channel = channels[1] * 0.5/(np.mean(channels[1]))

    # white balancing blue channel
    blue_channel = channels[2] * 0.5/(np.mean(channels[2]))

    # stack image to return as 3D-array
    # np stack axis = -1 is the same
    return np.dstack([red_channel, green_channel, blue_channel])


def curve_and_quantize(image, inv_gamma=0.85):
    """ Applies inverse gamma function and quantizes to 8-bit.
        Arguments:
          image: the input image in 32-bit floating point [HxWx3]
          inv_gamma: the value of 1/gamma
        Returns:
          The curved and quantized image in 8-bit unsigned integer [HxWx3]
    """
    # copy image to not mess with source array
    image = np.copy(image)

    # Step 4 Applying the gamma curve
    image = np.power(image, inv_gamma)

    # Step 5 Clipping the image, scale it with 255 and safe as unit8
    image = (np.clip(image,0,1)*255).astype('uint8')

    return image
