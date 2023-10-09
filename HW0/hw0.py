import cv2
import numpy as np

# PART I

# Step 1 load image 0 and show it
image_zero = cv2.imread('frames/000000.jpg')
cv2.imshow('Image 0', image_zero)
# waits for user to press any key and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 2 print out shape of the image array
# shape of 240x320x3 means the image has a pixel dimension of 240x320 and 3 channels which contain the blue, green
# and red channel of the picture
print(image_zero.shape)

# Step 3 print out the image itself
# It prints out 3 arrays which each have a dimension of 240x320 as mentioned before
# and each number in the array represents the brightness of each pixel for each channel in the picture ranging
# from 0 to 255 for the blue, green, and red channel.
# With 0 as the darkest brightness and 255 the brightest brightness
print(image_zero)

# Step 4 convert the image to grayscale and show it
image_zero_gray = cv2.cvtColor(image_zero, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', image_zero_gray)
# waits for user to press any key and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 5 save grayscale image
cv2.imwrite('000000_gray.png', image_zero_gray)


# Part II
# Step 1 setting up a video capture
vid_cap = cv2.VideoCapture('frames/%06d.jpg')

# Step 2 show each picture of the video capture
# Iterate through frames and display each frame
# prep for Step 3:
images = []

while True:
    # Read a frame
    ret, frame = vid_cap.read()

    # Break the loop if the end of the video is reached
    if not ret:
        break

    # append grayscaled frame for Step 3
    images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    # Display the frame
    cv2.imshow('Movie', frame)

    # Wait for a short duration and check for user input to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# waits for user to press any key and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 3 compute the average video frame
# convert list to numpy array and use np.mean() for averaging video
images = np.array(images)
background_image = np.mean(images, axis=0).astype('uint8')

# Step 4 Show the background Image
# Because we are averaging out each pixel, as the cameras perspective is stationary the background is
# present in each picture (and thus the corresponding pixel value for the background),
# so the average for each pixel is at least the background.
# The cars only appear in some of the many images and therefore do not contribute much to the average value.
cv2.imshow('Background Image', background_image)
# waits for user to press any key and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 5 Save Image as PNG
cv2.imwrite('background_gray.png', background_image)


# PART III
# Step 1 load images from Part I and II as grayscale images
image_0 = cv2.imread('000000_gray.png', cv2.IMREAD_GRAYSCALE)
background_image = cv2.imread('background_gray.png', cv2.IMREAD_GRAYSCALE)

# Step 2 calculate the absolute difference of the background and image from  PART I
im_dif = cv2.absdiff(image_0, background_image)
# Show the result
cv2.imshow('Absolute Difference Image', im_dif)
# waits for user to press any key and close all windows,
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 3 Threshold the absolute difference picture
# threshold variable to change manually
thresh_val = 45
ret2, im_bin_mask = cv2.threshold(im_dif, thresh_val, 255, cv2.THRESH_BINARY)
cv2.imshow('Binary Image - Manual', im_bin_mask)
# waits for user to press any key and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 4 Threshold with Otsu's method
ret3, im_bin_mask_otsu = cv2.threshold(
    im_dif, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('Binary Image - Otsu', im_bin_mask_otsu)
# waits for user to press any key and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 5 Eval
# With the Otsu method the Threshold Value is automatically determined
# With enough playing around on the normal threshold method you can get a similar result
# Improving the output:
# Preprocessing: Prior image preprocessing (e.g., noise reduction, image enhancement) might
# improve the performance of Otsu's method by ensuring a clearer separation between foreground and background.


# BONUS
# Step 1 Run Image Threshholding over each frame
# images still available in images array

# Array for binary images:
binary_images = []
# apply threshol on each frame and display frames to show as a movie
for frame in images:
    # subtract background from each frame
    frame = cv2.absdiff(frame, background_image)

    # apply threshold on current frame
    ret4, frame_binary_mask = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # append to binary image array
    binary_images.append(frame_binary_mask)

    # Display the frame
    cv2.imshow('Binary Movie', frame_binary_mask)

    # Wait for a short duration and check for user input to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# waits for user to press any key and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()


# convert to numpy array:
binary_images = np.array(binary_images)

# Step 2 Detect bounding Box on each Car and show as a movie
for frame_binary_mask, frame in zip(binary_images, images):
    # get contours for each frame
    contours, hierarchy = cv2.findContours(
        frame_binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over contours and draw bounding boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)

    # Display the image with bounding boxes
    cv2.imshow('Bounding Boxes', frame)

    # Wait for a short duration and check for user input to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# waits for user to press any key and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()