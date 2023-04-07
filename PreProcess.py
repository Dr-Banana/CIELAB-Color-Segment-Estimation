import cv2
import numpy as np
from sklearn.cluster import KMeans
import datetime

class im:
    def __init__(self, original):
        self.src = original
        self.rows, self.cols, self.c = original.shape
        self.equalizeGray_img = self.equalize_gray()
        self.enhance_img = self.increase_contrast_col()

    def equalize_gray(self):
        dst = cv2.GaussianBlur(self.src, (5, 5), 0)
        dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
        dst = cv2.equalizeHist(dst)
        return dst

    def increase_contrast_col(self):
        lab = cv2.cvtColor(self.src, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        dst = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return dst

    def kMeans_gray_1d(self, orig_enhance_gray, k):
        flatten_grayEq_b = orig_enhance_gray.flatten()
        kMeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(flatten_grayEq_b.reshape(-1, 1))
        centroids = kMeans.cluster_centers_
        centroid_1 = np.floor(centroids[1]).astype(np.uint8)
        out_list = np.zeros((self.rows, self.cols))
        out_list[np.where(orig_enhance_gray > centroid_1)] = 255
        return out_list.astype(np.uint8)

    def bgColExtract(self):
        enhance_gray = cv2.GaussianBlur(self.enhance_img, (5, 5), 0)
        enhance_gray = cv2.cvtColor(enhance_gray, cv2.COLOR_RGB2GRAY)
        enhance_gray = cv2.equalizeHist(enhance_gray)
        mask_1 = self.kMeans_gray_1d(enhance_gray, k=3)
        overlap = self.src.copy()
        background_color = np.mean(overlap[np.where(mask_1 == 255)], axis=0)
        return list(np.floor(background_color).astype(np.uint8))