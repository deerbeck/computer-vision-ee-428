import cv2
import numpy as np

class MagicWand:
    def __init__(self,calibration_path,R):
        """ Loads calibration from file and stores ball radius.
            Arguments:
                calibration_path: path to calibration file
                R: ball radius in cm
        """
        self.focal, self.centerx, self.centery = np.loadtxt(calibration_path,delimiter=' ')
        self.R = R

    def detect_ball(self,image):
        """ Detect one or more balls in image.
            Arguments:
                image: RGB image in which to detect balls
            Returns:
                list of tuples (x, y, radius)
        """
        # Pre-Processing of the image to use the ball detection on it
        # Start with converting frame into gray_scale.
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Applying gaussian blur on it
        # Kernel size of 9x9 and sigma = 2
        kernel_size = (9,9)
        sigma = 2
        image_gauss = cv2.GaussianBlur(image_gray,kernel_size,sigma)

        # Run HughCircles to detect circles
        circles = cv2.HoughCircles(image_gauss, method=cv2.HOUGH_GRADIENT, dp=2, minDist = 10)

        # rearrange list of circles to match desired return type
        circles = list(map(tuple,circles[0, :]))

        return circles

    def calculate_ball_position(self,x,y,r):
        """ Calculate ball's (X,Y,Z) position in world coordinates
            Arguments:
                x,y: 2D position of ball in image
                r: radius of ball in image
            Returns:
                X,Y,Z position of ball in world coordinates
        """
        # Applying projection equationws

        pass

    def draw_ball(self,image,x,y,r,Z):
        """ Draw circle on ball and write depth estimate  in center
            Arguments:
                image: image on which to draw
                x,y,r: 2D position and radius of ball
                Z: estimated depth of ball
        """

        cv2.circle( image, (int(x),int(y)), int(r), (0,0,255),2)
        cv2.putText( image, str(int(Z)) + ' cm', (int(x),int(y)), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
    
    def project(self,X,Y,Z):
        """ Pinhole projection.
            Arguments:
                X,Y,Z: 3D point
            Returns:    
                (x,y) 2D location of projection in image
        """
        pass

    def draw_line(self,image,X1,Y1,Z1,X2,Y2,Z2):
        """ Draw a 3D line
            Arguments:
                image: image on which to draw
                X1,Y1,Z1: 3D position of first line endpoint
                X2,Y2,Z2: 3D position of second line endpoint
        """
        pass

    def draw_bounding_cube(self,image,X,Y,Z):
        """ Draw bounding cube around 3D point, with radius R
            Arguments:
                image: image on which to draw
                X,Y,Z: 3D center point of cube
        """
        pass
    
    def process_frame(self,image):
        """ Detect balls in frame, estimate 3D positions, and draw on image
            Arguments:
                image: image to be processed
            Returns:
                list of (X,Y,Z) 3D points of detected balls
        """
        self.detect_ball(image)
        pass

