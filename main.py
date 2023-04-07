import time
import NLT
import Plot
import numpy as np
import cv2
from skimage import color as col
path = "config/phase.jpg"


if __name__ == '__main__':
    NLT.video()
    # Plot.ColorSpace().function_plot()
    # random_img = Plot.generate_clustered_image(k=5)
    # method = NLT.ClusterMethod(path) # , video=random_img
    # original = method.img
    # Plot.plot_runtime()
    # Plot.plot_CRESE()
    # Plot.plot_robustness()
    # start_time = time.time()
    # k = method.GapStatistic()
    # print("gap statistic: ", '%.35f' % (time.time() - start_time), "K value:", k)
    # color_image = method.cluster_image(k, "gap statistic: ")
    # print(Plot.crese(original, color_image))
    #
    # start_time = time.time()
    # k = method.ElbowMethod()
    # print("Elbow Method: ", '%.35f' % (time.time() - start_time), "K value:", k)
    # color_image = method.cluster_image(k, "Elbow Method: ")
    # print(Plot.crese(original, color_image))
    #
    # start_time = time.time()
    # k = method.NoC()
    # print("CCSE: ", '%.35f' % (time.time() - start_time), "K value:", k)
    # color_image = method.cluster_image(k, "CCSE:")
    # print(Plot.crese(original, color_image))
    # method.label_img_coloring(k, "CCSE")