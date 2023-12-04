import cv2
import numpy as np
import time


class RANSAC:
    def __init__(self, prob_success=0.99, init_outlier_ratio=0.7, inlier_threshold=5.0):
        """ Initializes a RANSAC estimator.
            Arguments:
                prob_success: probability of success
                init_outlier_ratio: initial outlier ratio
                inlier_threshold: maximum re-projection error for inliers
        """
        self.prob_success = prob_success
        self.init_outlier_ratio = init_outlier_ratio
        self.inlier_threshold = inlier_threshold

        # sample size = 4 for homography
        self.sample_size = 4

    def compute_num_iter(self, outlier_ratio):
        """ Compute number of iterations given the current outlier ratio estimate.

            The number of iterations is computed as:

                N = ceil( log(1-p)/log(1-(1-e)^s) )

            where p is the probability of success,
                  e is the outlier ratio, and
                  s is the sample size.

            Arguments:
                outlier_ratio: current outlier ratio estimate
            Returns:
                number of iterations
        """
        n = np.ceil(np.log(1-self.prob_success) /
                    np.log(1-(1-outlier_ratio)**self.sample_size))
        return n

    def compute_inlier_mask(self, H, ref_pts, query_pts):
        """ Determine inliers given a homography estimate.

            A point correspondence is an inlier if its re-projection error is
            below the given threshold.

            Arguments:
                H: homography to be applied to the reference points [3,3]
                ref_pts: reference points [N,1,2]
                query_pts: query points [N,1,2]
            Returns:
                Boolean array where mask[i] = True means the point i is an inlier. [N]
        """

        # first use homography on reference points to compare against query after
        comp_query_pts = cv2.perspectiveTransform(ref_pts, H)

        # calculate the re-projection error
        # Calculate Euclidean distance between transformed reference points and query points
        errors = np.linalg.norm(comp_query_pts - query_pts, axis=2)

        # now return inlier mask (its re-projection error is below the given threshold.)
        return errors < self.inlier_threshold

    def find_homography(self, ref_pts, query_pts):
        """ Compute a homography and determine inliers using the RANSAC algorithm.

            The homography transforms the reference points to match the query points, i.e.

            query_pt ~ H * ref_pt

            Arguments:
                ref_pts: reference points [N,1,2]
                query_pts: query points [N,1,2]
            Returns:
                H: the computed homography estimate [3,3]
                mask: the Boolean inlier mask [N]
        """
        # iter variable to keep track of number of iterations
        iter = 0

        # initialize first outlier_ratio
        outlier_ratio = self.init_outlier_ratio

        # final homography and inlier mask
        final_H = np.array([])
        final_inlier_mask = np.array([0])

        # perform RANSAC iterations
        while iter < self.compute_num_iter(outlier_ratio):

            # choose 4 random points ot of the reference and the query points
            rand_ind = np.random.choice(ref_pts.shape[0],
                                        size=4, replace=False)

            selected_ref_points = ref_pts[rand_ind]
            selected_query_points = query_pts[rand_ind]

            # calculate Homography with those 4 points
            H, inliers = cv2.findHomography(selected_ref_points,
                                            selected_query_points, method=0)

            # compute the inlier mask
            inlier_mask = self.compute_inlier_mask(H, ref_pts, query_pts)


            # check if inliers improved
            if np.count_nonzero(inlier_mask) > np.count_nonzero(final_inlier_mask):

                # update final mask and Homographie
                final_H = H
                final_inlier_mask = inlier_mask

                # calculate new outlier ratio
                outlier_ratio = (
                    (len(final_inlier_mask)-np.count_nonzero(final_inlier_mask))/len(final_inlier_mask))
            # increment iter var
            iter += 1

        # finally recompute the homography with the inliers found
        final_H, inliers = cv2.findHomography(
            ref_pts[final_inlier_mask], query_pts[final_inlier_mask], method=0)

        return final_H, final_inlier_mask


class Tracker:
    def __init__(self, reference, overlay, min_match_count=10, inlier_threshold=5):
        """ Initializes a Tracker object.

            During initialization, this function will compute and store SIFT keypoints
            for the reference image.

            Arguments:
                reference: reference image
                overlay: overlay image for augmented reality effect
                min_match_count: minimum number of matches for a video frame to be processed.
                inlier_threshold: maximum re-projection error for inliers in homography computation
        """
        # store reference, overlay, min_match_count and inlier_threshold
        self.ref = reference
        self.overlay = overlay
        self.min_match_count = min_match_count
        self.inlier_threshold = inlier_threshold

        # grayscale image
        self.gray_ref = cv2.cvtColor(self.ref, cv2.COLOR_BGR2GRAY)
        # create sift object
        sift = cv2.SIFT_create()

        # extract keypoints and store them
        self.trgt_keypoints, self.trgt_descriptors = sift.detectAndCompute(
            self.gray_ref, None)

    def compute_homography(self, frame, ratio_thresh=0.7):
        """ Calculate homography relating the reference image to a query frame.

            This function first finds matches between the query and reference
            by matching SIFT keypoints between the two image.  The matches are
            filtered using the ratio test.  A match is accepted if the first
            nearest neighbor's distance is less than ratio_thresh * the second
            nearest neighbor's distance.

            RANSAC is then applied to matches that pass the ratio test, to determine
            inliers and compute a homography estimate.

            If less than min_match_count matches pass the ratio test,
            the function returns None.

            Arguments:
                frame: query frame from video
            Returns:
                the estimated homography [3,3] or None if not enough matches are found
        """
        # first find keypoints in query frame
        # same procedure as in init
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        frame_keypoints, frame_descriptors = sift.detectAndCompute(
            gray_frame, None)

        # initiate BF Matcher object
        matcher = cv2.BFMatcher()

        # find two nearest neighbors
        matches = matcher.knnMatch(
            self.trgt_descriptors, frame_descriptors, k=2)

        # apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        # return None if numnber of good matches is below threshold
        if len(good_matches) <= self.min_match_count:
            return None

        # extract target and frame points to find Homography
        trgt_pts = np.float32(
            [self.trgt_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        frame_pts = np.float32(
            [frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)


        # compute homography and inliers
        # create RANSAC object
        ransac = RANSAC()
        H, inliers = ransac.find_homography(trgt_pts, frame_pts)

        # in case I want to use the cv2 functionality
        # H, inliers = cv2.findHomography(trgt_pts, frame_pts, cv2.RANSAC, self.inlier_threshold)


        return H

    def augment_frame(self, frame, H):
        """ Draw overlay image on video frame.

            Arguments:
                frame: frame to be drawn on [H,W,3]
                H: homography [3,3]
            Returns:
                augmented frame [H,W,3]
        """
        # get height and width from frame (output image)
        height = frame.shape[0]
        width = frame.shape[1]

        # get warped perspective of overlay and scale it down for alphablend
        warped = cv2.warpPerspective(
            self.overlay, H, (width, height)).astype("float") / 255
        frame = frame.astype("float") / 255

        # get alpha image out of ones and scale up to full brightness
        overlay_height = self.overlay.shape[0]
        overlay_width = self.overlay.shape[1]
        mask = np.ones((overlay_height, overlay_width, 3))
        alpha = cv2.warpPerspective(mask, H, (width, height))

        # scale back up (not necessary for cv2.imshow but)
        augmented_frame = ((np.multiply(warped, alpha) +
                           np.multiply((1-alpha), frame))*255).astype("uint8")

        return augmented_frame
