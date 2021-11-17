import mmcv
import numpy as np
import cv2
import random
from ..registry import PIPELINES

def padding(target, neighbor_align):
    height, width, _ = target.shape

    for i in range(height):
        for j in range(width):
            if sum(neighbor_align[i,j:]) == 0.0:
                neighbor_align[i,j,:] = target[i,j,:]

    return neighbor_align


def sift(target, neighbor, min_match_count = 200):
    align_neighbor = neighbor.copy()
    sift_module = cv2.xfeatures2d.SIFT_create()
    
    # FLANN matcher; KD tree
    index_params = dict(algorithm = 1, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    height, width, _ = target.shape

    # compute homography and align
    target_img_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    neighbor_img_gray = cv2.cvtColor(neighbor, cv2.COLOR_RGB2GRAY)
    
    target_img_gray = cv2.normalize(target_img_gray, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    neighbor_img_gray = cv2.normalize(neighbor_img_gray, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # find the keypoints and descriptors with sift
    neighbor_kps, neighbor_des = sift_module.detectAndCompute(neighbor_img_gray, None)
    target_kps, target_des = sift_module.detectAndCompute(target_img_gray, None)

    # use knn algorithm
    matches = None
    try:
        matches = flann.knnMatch(neighbor_des, target_des, k = 2)
    except:
        return neighbor

    # remove error matches
    good = []
    for m, n in matches:
        if m.distance <= 0.4 * n.distance:
            good.append(m)
    
    if len(good) > min_match_count:
        src_pts = np.float32([neighbor_kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([target_kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        # compute homography matrix
        homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # warp neighbor image
        try:
            align_neighbor = cv2.warpPerspective(neighbor, homography_matrix, (width, height))
            align_neighbor = padding(target, align_neighbor)
        except:
            return neighbor

    return align_neighbor


@PIPELINES.register_module()
class HomographyWithSIFT:
    """
        SIFT Implement.
    """

    def __init__(self, keys, ratio):
        self.keys = keys
        self.ratio = ratio

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        for key in self.keys:
            if isinstance(results[key], list):
                for v in self.keys:
                    if v == 'lq' and random.random() < self.ratio:
                        center_index = len(results[v]) // 2
                        center_frame = results[v][center_index].copy()
                        for idx, frame in enumerate(results[v]):
                            if idx != center_index and abs(idx - center_index) == 1:
                                results[v][idx] = sift(center_frame, frame)
                                    
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'