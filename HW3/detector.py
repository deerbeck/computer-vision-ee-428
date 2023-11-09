import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter


class FeatureDetector:
    def __init__(self, sigma=1, nblur=10, thresh=0.05):
        """ Initializes the FeatureDetector object.

            The feature detector detects difference-of-Gaussian (DOG) features.

            Features are detected by finding local minima / maxima in the DOG
            response stack.

            Arguments:
              sigma: base sigma value for Gaussian filters 
              nblur: number of Gaussian filters 
              thresh: minimum absolute response value for a feature detection
        """
        self.__sigma = sigma
        self.__nblur = nblur
        self.__thresh = thresh

    def get_dog_response_stack(self, image):
        """ Build the DOG response stack.

            The image is first converted to grayscale, floating point on [0 1]
            range. Then a difference-of-Gaussian response stack is built.

            Let I be the original (grayscale) image. Let G[i] be the result of
            applying a Gaussian with sigma s*((1.5)^i) to I, where s is the
            base sigma value.

            Layer i in the stack is computed as G[i+1]-G[i].

            Arguments:
                image: 8-bit BGR input image
            Returns:
                DOG response stack [nblur,H,W]
        """
        # initialize Gaussian blur array
        G = []

        # convert image to grayscale and typecast to float as well as normalize
        # it to [0 1] scale so no information will be lost when subtracting
        # images
        I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype("float32")/(2**8-1)
        # I = np.random.randint(0, 256, size=(10, 10))

        # get number of blured images and put them in G
        for i in range(self.__nblur+1):
            G.append(gaussian_filter(I, self.__sigma*((1.5)**i)))

        # build difference-of-gaussian response stack
        DOG = []
        for i in range(self.__nblur):
            DOG.append(G[i+1] - G[i])

        return np.array(DOG)

    def find_features(self, responses):
        """ Find features in the DOG response stack.

            Features are detected using non-minima / non-maxima supression over
            a 3x3x3 window.

            To do this, compute the local minimum / maximum at each location
            using skimage.ndimage.minimum_filter / maximum_filter.

            Then find locations where the response value is equal to the local
            minimum/ maximum, and the absolute response value exceeds thresh.

            See np.argwhere for a fast way to to do this.

            Arguments:
                response: DOG response stack
            Returns:
                List of features (level,y,x)
        """
        # get non maxima and non minima supression over a 3x3x3 window using
        # the maximum_filter/minimum_filter from scipy
        max_response = maximum_filter(responses, size=(3, 3, 3))
        min_response = minimum_filter(responses, size=(3, 3, 3))

        # apply the threshholds as well as the reponses to get a boolean array
        # and use argwhere on them to get the location of the responses
        feature_locations = np.argwhere(((responses == max_response) &
                                         (responses > self.__thresh)) |
                                        ((responses == min_response) &
                                         (responses < -self.__thresh)))
        # just for cleaniness turn the lists in the features array into tuples
        features = [(level, y, x) for level, y, x in feature_locations]

        return features

    def draw_features(self, image, features, color=[0, 0, 255]):
        """ Draw features on an image.

            For each feature, draw it as a dot and a circle.

            The radius of the circle should be equal to the sigma value at that
            level.

            Arguments:
                image: input 8-bit BGR image features: list of (level,y,x)
                features color: color in which to draw
            Returns:
                Image with features drawn
        """
        for feature in features:
            # extract necessary data from feature
            x = feature[2]
            y = feature[1]
            level = feature[0]

            # calculate radius with given formula
            r = self.__sigma*((1.5)**level)

            # draw circle in image with calculated radius 
            cv2.circle(image, (int(x), int(y)), int(r), color, 2)
            
            # draw middle of circle (linethickness -1 fills out circle)
            cv2.circle(image, (int(x), int(y)), 0, color, -1)

        return image
