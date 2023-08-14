import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


class Panorama:
    def __init__(self):
        self.detector = cv2.ORB_create(nfeatures=1500)

    def create(self, img_left, img_right, draw_matches=False):
        '''
            Create a panoramic image given two images
        '''

        gray_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        kps_l, features_l = self.detector.detectAndCompute(gray_l, None)
        kps_r, features_r = self.detector.detectAndCompute(gray_r, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
        matches = sorted(bf.match(features_l, features_r),
                         key=lambda x: x.distance)

        if draw_matches:
            matches_img = cv2.drawMatches(gray_l, kps_l, gray_r, kps_r, np.random.choice(
                matches, 100), None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(matches_img)
            plt.show()

        H = self.ransac(kps_l, kps_r, matches)

        warp_img = self.warp(img_left, img_right, H)

        return warp_img

    def homography(self, img_l, img_r):
        A = []
        for r in range(len(img_l)):
            A.append([-img_l[r, 0], -img_l[r, 1], -1, 0, 0, 0, img_l[r, 0]
                     * img_r[r, 0], img_l[r, 1] * img_r[r, 0], img_r[r, 0]])
            A.append([0, 0, 0, -img_l[r, 0], -img_l[r, 1], -1, img_l[r, 0]
                     * img_r[r, 1], img_l[r, 1] * img_r[r, 1], img_r[r, 1]])
        u, s, vt = np.linalg.svd(A)
        H = np.reshape(vt[8], (3, 3))
        H = (1/H.item(8)) * H
        return H

    def ransac(self, kps_l, kps_r, matches):
        '''
            Fit the best homography model with RANSAC algorithm
        '''

        dstPoints = np.float32([kps_l[m.queryIdx].pt for m in matches])
        srcPoints = np.float32([kps_r[m.trainIdx].pt for m in matches])

        num_sample = len(matches)
        threshold = 5.0
        num_iter = 2000
        sub_sample = 4
        max_inlier = 0
        best_H = None
        k = 0
        while (k < num_iter):
            sub_sample_index = random.sample(range(num_sample), sub_sample)
            H = self.homography(
                srcPoints[sub_sample_index], dstPoints[sub_sample_index])
            num_inlier = 0
            for i in range(num_sample):
                if i not in sub_sample_index:
                    concateCoor = np.hstack((srcPoints[i], [1]))
                    dstCoor = H @ concateCoor.T
                    if not dstCoor[2]:
                        continue
                    dstCoor = dstCoor / dstCoor[2]
                    if np.linalg.norm(dstCoor[:2] - dstPoints[i]) < threshold:
                        num_inlier = num_inlier + 1
            if max_inlier < num_inlier:
                max_inlier = num_inlier
                best_H = H
            k += 1
        return best_H

    def warp(self, img_left, img_right, H):
        '''
           Warp image to create panoramic image
        '''
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        panoramic_img = np.zeros((max(hl, hr), wl + wr, 3), dtype="int")
        panoramic_img[:hl, :wl] = img_left
        inv_H = np.linalg.inv(H)
        for i in range(panoramic_img.shape[0]):
            for j in range(panoramic_img.shape[1]):
                coor = np.array([j, i, 1])
                img_right_coor = inv_H @ coor
                img_right_coor /= img_right_coor[2]
                y, x = int(round(img_right_coor[0])), int(
                    round(img_right_coor[1]))
                if x >= 0 and x < hr and y >= 0 and y < wr:
                    panoramic_img[i, j] = img_right[x, y]

        panoramic_img = self.removeBlackBorder(panoramic_img)
        return panoramic_img

    def removeBlackBorder(self, img):
        '''
        Remove the black border from the image
        '''
        h, w = img.shape[:2]
        reduced_h, reduced_w = h, w
        for col in range(w - 1, -1, -1):
            all_black = True
            for i in range(h):
                if np.count_nonzero(img[i, col]) > 0:
                    all_black = False
                    break
            if all_black:
                reduced_w = reduced_w - 1

        for row in range(h - 1, -1, -1):
            all_black = True
            for i in range(reduced_w):
                if np.count_nonzero(img[row, i]) > 0:
                    all_black = False
                    break
            if all_black:
                reduced_h = reduced_h - 1

        return img[:reduced_h, :reduced_w]
