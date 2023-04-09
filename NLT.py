import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
from skimage import color as col


class ClusterMethod:
    def __init__(self, path=None, video=None):
        self.path = path
        self.video = video
        self.src = self.readImg()
        self.src_vector = self.src.reshape(-1, 3)
        self.lab_roi = col.rgb2lab(self.src)
        self.D = 5
        self.k = self.ccse()
        self.segmentImg = self.segmentImg()

    def readImg(self):
        if self.path is not None:
            return cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2RGB)
        elif self.video is not None:
            return self.video

    def ccse(self):
        img = self.src
        # img = pre.im(img).enhance_img
        lab_roi = self.lab_roi
        # Calculate the histogram of each channel
        hist_l, _ = np.histogram(lab_roi[:, 0], bins=256, range=(0, 256))
        hist_a, _ = np.histogram(lab_roi[:, 1], bins=256, range=(0, 256))
        hist_b, _ = np.histogram(lab_roi[:, 2], bins=256, range=(0, 256))
        dis = self.D
        peaks_l, _ = find_peaks(hist_l, distance=dis)
        peaks_a, _ = find_peaks(hist_a, distance=dis)
        peaks_b, _ = find_peaks(hist_b, distance=dis)
        while np.std(hist_l) < len(peaks_l):
            dis += self.D
            peaks_l, _ = find_peaks(hist_l, distance=dis)

        num_clusters = int(np.floor((np.log(np.std(hist_l) / len(peaks_l))))) * (
            max(len(peaks_a), len(peaks_b)))+min(len(peaks_a), len(peaks_b))
        return num_clusters

    def segmentImg(self):
        img = self.src
        # Reshape the image into a 2D array of pixels
        pixels = self.src_vector

        # Perform k-means clustering on the pixels
        kmeans = KMeans(n_clusters=self.k, random_state=0, n_init="auto").fit(pixels)
        # Assign each pixel to its nearest centroid
        labels = kmeans.predict(pixels)

        # Reshape the labels back into the original image shape
        clustered = labels.reshape(img.shape[:2])
        blank = img.copy()
        for i in range(max(labels) + 1):
            r = int(np.mean(img[np.where(clustered == i)][:, 0]))
            g = int(np.mean(img[np.where(clustered == i)][:, 1]))
            b = int(np.mean(img[np.where(clustered == i)][:, 2]))
            col = [r, g, b]
            blank[np.where(clustered == i)] = col
        blank = cv2.medianBlur(blank, 3)
        return blank

    def imshow(self):
        return self.segmentImg

    def imlabel(self):
        img = self.src
        # Reshape the image into a 2D array of pixels
        pixels = self.src_vector

        # Perform k-means clustering on the pixels
        kmeans = KMeans(n_clusters=self.k, random_state=0, n_init="auto").fit(pixels)
        # Assign each pixel to its nearest centroid
        labels = kmeans.predict(pixels)
        labels = labels.copy()
        label_image = labels.reshape(img.shape[:2])
        # Create a colormap with 9 distinct colors
        hue_values = (label_image / 8 * 180).astype(np.uint8)
        hsv_image = np.zeros((*label_image.shape, 3), dtype=np.uint8)
        hsv_image[:, :, 0] = hue_values
        hsv_image[:, :, 1] = 255
        hsv_image[:, :, 2] = 255
        color_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        color_image = cv2.medianBlur(color_image, 5)
        return color_image

    def imClusters(self):
        img = self.src
        # Reshape the image into a 2D array of pixels
        pixels = self.src_vector

        # Perform k-means clustering on the pixels
        kmeans = KMeans(n_clusters=self.k, random_state=0, n_init="auto").fit(pixels)
        # Assign each pixel to its nearest centroid
        labels = kmeans.predict(pixels)

        # Reshape the labels back into the original image shape
        clustered = labels.reshape(img.shape[:2])

        binary_images = []
        colored_binary_images = []

        for i in range(self.k):
            binary_img = np.zeros(clustered.shape, dtype=np.uint8)
            binary_img[clustered == i] = 255
            binary_images.append(binary_img)
            color_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
            colored_img = cv2.bitwise_and(color_img, img)
            colored_binary_images.append(colored_img)

        return colored_binary_images

    def segmentEdge(self):
        ksize = 3
        scale = 1
        delta = 0
        img = self.segmentImg
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale, delta=delta)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale, delta=delta)
        edges = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

        def close_edges(image, kernel_size=(5, 5)):
            kernel = np.ones(kernel_size, np.uint8)
            closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            return closing

        edges = close_edges(edges)
        return edges


def crese(image, averaged_color_image):
    # Get unique colors from the averaged_color_image
    unique_colors = np.unique(averaged_color_image.reshape(-1, 3), axis=0)

    max_error = np.sqrt(255 ** 2 * 3)
    total_pixels = image.shape[0] * image.shape[1]
    max_total_error = max_error * total_pixels

    reconstruction_error = 0
    for color in unique_colors:
        mask = np.all(averaged_color_image == color, axis=-1)
        color_difference = image[mask] - color
        reconstruction_error += np.sqrt(np.sum(color_difference ** 2, axis=1)).sum()

    normalized_error = (reconstruction_error / max_total_error) * 100
    return normalized_error