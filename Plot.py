import numpy as np
import matplotlib.pyplot as plt


class ColorSpace:
    def __init__(self):
        self.resolution = 10
        self.R, self.G, self.B = np.meshgrid(np.linspace(0, 1, self.resolution),
                                             np.linspace(0, 1, self.resolution),
                                             np.linspace(0, 1, self.resolution))

    def LAB_3D_Plot(self):
        # Flatten and stack the RGB components
        rgb_colors = np.stack((self.R.flatten(), self.G.flatten(), self.B.flatten()), axis=-1)

        # Convert RGB to L*a*b* for plotting
        import skimage.color as color
        lab_colors = color.rgb2lab(rgb_colors)

        # Create the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot L*a*b* color space points with their corresponding RGB colors
        for i, color in enumerate(rgb_colors):
            ax.scatter(lab_colors[i, 0], lab_colors[i, 1], lab_colors[i, 2], c=color.reshape(1, -1), marker='o')

        ax.set_xlabel('L*', fontsize=18)
        ax.set_ylabel('a*', fontsize=18)
        ax.set_zlabel('b*', fontsize=18)
        plt.show()

    def RGB_3D_Plot(self):
        # Flatten and stack the RGB components
        rgb_colors = np.stack((self.R.flatten(), self.G.flatten(), self.B.flatten()), axis=-1)

        # Create the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot L*a*b* color space points with their corresponding RGB colors
        for i, color in enumerate(rgb_colors):
            ax.scatter(rgb_colors[i, 0], rgb_colors[i, 1], rgb_colors[i, 2], c=color.reshape(1, -1), marker='o')

        ax.set_xlabel('R', fontsize=18)
        ax.set_ylabel('G', fontsize=18)
        ax.set_zlabel('B', fontsize=18)
        plt.show()


def generate_clustered_image(k, width=128, height=128):
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate k random RGB colors
    cluster_colors = np.random.randint(0, 256, size=(k, 3), dtype=np.uint8)

    # Calculate the height of each cluster's region
    region_height = height // k

    # Assign each region a unique cluster color
    for i in range(k):
        start = i * region_height
        end = (i + 1) * region_height if i != k - 1 else height
        image[start:end, :] = cluster_colors[i]
    return image


def plot_CRESE(image_names, ccse_errors, elbow_errors, gap_errors):
    image_names = ["Beach", "Church", "Lena", "Desk"]
    gap_errors = [1.9852748209163464, 2.2486569852110576, 2.4182489234966287, 2.855213280338615]
    elbow_errors = [2.239360122419021, 2.3886972658916874, 2.6705288772133318, 2.896647313589627]
    ccse_errors = [2.3119905686342137, 2.5449697022680193, 2.954758500593425, 3.2784352686144915]
    ccse_errors = [2.0829530256118205, 2.3708661619376272, 2.5787101659123604, 3.0073513063508273]
    x = np.arange(len(image_names))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, ccse_errors, width, label='CCSE')
    rects2 = ax.bar(x, elbow_errors, width, label='Elbow')
    rects3 = ax.bar(x + width, gap_errors, width, label='Gap Statistic')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Color Reconstruction Error(%)', fontsize=18)
    ax.set_title('CRESE by Original Image and Method', fontsize=18)
    ax.set_xticks(x)
    ax.set_ylim(0, 4)
    ax.set_xticklabels(image_names, fontsize=18)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.show()


def plot_runtime():
    image_size = ["380x220", "960x540", "1280x720", "1920x1080", "2880x1620", "3840x2160"]
    gap_time = [14.5845, 97.9925, 172.647, 355.3536, 867.3867, 1308.5578]
    elbow_time = [3.8137, 16.7425, 34.9116, 63.4547, 111.9856, 205.6381]
    ccse_time = [0.1719, 0.1719, 0.1876, 0.1876, 0.1875, 0.1876]
    x = np.arange(len(image_size))  # the label locations

    fig, ax = plt.subplots()
    rects1 = ax.plot(x, ccse_time, '-o', label='CCSE')
    rects2 = ax.plot(x, elbow_time, '-o', label='Elbow')
    rects3 = ax.plot(x, gap_time, '-o', label='Gap Statistic')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Time in seconds', fontsize=18)
    ax.set_xlabel('Image Size', fontsize=18)
    ax.set_title('Runtime Comparison', fontsize=18)
    ax.set_yscale("log")
    ax.set_xticks(x)
    # ax.set_ylim(1, 3000)
    ax.set_xticklabels(image_size, rotation=45)
    ax.legend()

    def autolabel(rects, time):
        for i, time in enumerate(time):
            ax.annotate('{:.2f}'.format(time),
                        xy=(x[i], time),
                        xytext=(-2, 3),  # 3 points vertical offset, 2 horizontal offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1, ccse_time)
    autolabel(rects2, elbow_time)
    autolabel(rects3, gap_time)

    fig.tight_layout()
    plt.show()


def plot_robustness():
    parameter = [5, 10, 15, 20, 25, 30, 35, 40]
    gap_k = [5, 10, 15, 20, 25, 30, 35, 40]
    elbow_k = [4, 10, 13, 16, 21, 23, 33, 34]
    ccse_k = [21, 10, 7, 6, 5, 4, 3, 6]
    x = np.arange(len(parameter))  # the label locations

    fig, ax = plt.subplots()
    rects1 = ax.plot(x, ccse_k, '-o', label='CCSE')
    rects2 = ax.plot(x, elbow_k, '-o', label='Elbow')
    rects3 = ax.plot(x, gap_k, '-o', label='Gap Statistic')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Estimated K', fontsize=18)
    ax.set_xlabel('D or $K_{max}$', fontsize=18)
    ax.set_title('Parameter Sensitivity', fontsize=18)
    # ax.set_yscale("log")
    ax.set_xticks(x)
    # ax.set_ylim(1, 3000)
    ax.set_xticklabels(parameter, rotation=45)
    ax.legend()

    def autolabel(rects, time):
        for i, time in enumerate(time):
            ax.annotate('{}'.format(time),
                        xy=(x[i], time),
                        xytext=(-2, 3),  # 3 points vertical offset, 2 horizontal offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1, ccse_k)
    autolabel(rects2, elbow_k)
    autolabel(rects3, gap_k)

    fig.tight_layout()
    plt.show()
