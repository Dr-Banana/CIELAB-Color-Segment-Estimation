# CIELAB Color Segment Estimation (CCSE)

## Abstract

In the field of image analysis, computer algorithms must recognize and extract essential features or patterns within visual data to accurately represent objects, scenes, or activities. Color information is a critical aspect of object classification, as color differences enable humans to identify the presence of one or more objects. This research introduces a novel method for determining the optimal number of color clusters in CIELAB color space images, using mathematical expressions and K-Means as image segmentation method. Our approach combines perceptual lightness, red, green, blue, and yellow histogram standard deviations, along with the number of local maximum points, by employing a non-linear equation to calculate the ideal number of clusters. Experimental results reveal that the proposed method surpasses widely used techniques such as the Elbow Method and Gap Statistic Method in terms of time. Moreover, our algorithm demonstrates exceptional accuracy in processing visual data within real-world environments, maintaining computational efficiency even as the number of clusters increases.

## Files

- main.py

- Input.py
  > acquire webcamera from computer

- NLT.py
  > The file used to calculate the number of cluster using CCSE function and generate the segmented images.

- Plot.py
  > The file used to plot the 3D view of RGB color space, LAB color space, runtime data, robustness, and accuracy experiment data.
  
- PreProcess.py
  > The file used to do image preprocessing such as convert image to gray scale or enhance image contrast.

## MATH

*A. Proposed Method*

The method proposed in this paper can be divided into three parts. The first part is to transform three-channel RGB color into three-channel LAB and extract the histogram of L, A and B respectively. The second part is to calculate the optimal standard deviation and number of local maxima from three channels respectively as parameters. Finally, the best number of clusters is determined using the CIELAB Color Segment Estimation (CCSE) function, as shown below:

$$
k = \lfloor ln(\frac{\sigma_{L}}{n_L})\rfloor max(n_a,n_b)+min(n_a,n_b)
$$

In this equation, $\sigma_{L}$ denotes the standard deviation of L histogram and $n_L$, $n_a$, $n_b$ represent the number of local maxima extracted from the L, a, and b histograms in the second part. $max⁡(n_a,n_b)$ refers to the larger value between n_a and n_b, while $min(n_a,n_b)$ indicates the smaller value. 
The detailed steps of the designed method can be found in the pipeline presented in Figure 3. It is important to note that this method specifically requires input images in the three-channel RGB color format.

*B. Color Histogram*

Color is usually represented by color histogram, color correlogram, color coherence vector, and color moment under a certain color space. The color histogram serves as an effective representation of the color content of an image if the color pattern is unique compared with the rest of the data set.

We extract information from the L, a, and b histograms by examining the standard deviation and the number of local maxima. These metrics provide insights into the following aspects and explain the idea behind CCSE function:

  1.	**Lightness distribution**: The L histogram's standard deviation reveals the range of lightness values in the image. A higher standard deviation signifies a broader range, while a lower value indicates a narrower range.
  2.	**Color distribution**: Standard deviations in the a and b histograms reflect the range of colors present in the image. Higher values suggest a more diverse color palette, while lower values indicate a more limited one.
  3.	**Peak separability**: The number of local maxima in the L, a, and b histograms informs the separability of different lightness and color levels, which is useful for image segmentation or object recognition. More local maxima denote distinct peaks, while fewer maxima suggest closely clustered peaks.
  4.	**Image complexity**: A combination of the standard deviation and the number of local maxima in the L, a, and b histograms provides an indication of overall image complexity in terms of lightness and color distribution. Images with higher standard deviations and more local maxima are considered more complex, while those with lower values are simpler.

In conclusion, color histogram analysis allows for a comprehensive understanding of an image's lightness and color distribution, peak separability, and overall complexity. This information can be invaluable for various image processing tasks, such as segmentation and object recognition.

*C. CCSE*

The term $\lfloor ln(\frac{\sigma_{L}}{n_L})\rfloor$ calculates the logarithm of the ratio between the standard deviation of the L histogram and the number of local maxima in the L histogram. This ratio reflects the relative spread of lightness values and the degree of separability between different lightness levels in the image. Taking the logarithm of this ratio helps normalize the scale of the values, ensuring that the final estimation is less sensitive to extreme values. logarithm is rounded down to make sure that the number of clusters is an integer. To keep normalized cluster number positive and avoid too small values of $\sigma_{L}$ because the colors in the picture all belong to the same hues.

By incorporating $max(n_a, n_b)$ into product term, the CCSE function accounts for the channel with the most distinct local maxima, which corresponds to the channel with the most dominant color variation. In contrast, using $min(n_a, n_b)$ would emphasize the channel with the least color variation, which might lead to an underestimation of the true number of color segments required for effective image segmentation.

Adding $min(n_a, n_b)$ to the expression allowing the contribution of the less dominant color channel to the overall color distribution is not entirely disregarded. While it is essential to prioritize the channel with the most significant color variation, considering the less dominant channel helps achieve a more accurate estimation of the required number of color segments.

*D. Algorithm*

![image](https://user-images.githubusercontent.com/77912454/230666521-199b02f4-3165-4116-9f4c-84eb0468dee1.png)

*E. Time Complexity*

The time complexity of the given function can be analyzed by looking at the time complexity of each of its operations. Given that the histogram size is no more than 256 bins, the time complexity of finding peaks using Find_Local_Maxima can be treated as constant. Therefore, the time complexity of the function can be simplified as follows: assigning image variable has a time complexity of O(1), converting the image from RGB to LAB color space has a time complexity of O(N), where N is the total number of pixels in the image, calculating histograms has a time complexity of O(N), finding peaks has a time complexity of O(1), the while loop in the function has a worst-case time complexity of O(N), and the mathematical operation to determine the number of clusters has a time complexity of O(1). Therefore, the overall time complexity of the function is O(N), where N is the total number of pixels in the image.

## RESULTS

*A. Visual Validation*

The performance of image segmentation using the CCSE algorithm can be validated by visually comparing the segmented images produced using the optimal number of clusters (k) calculated by the CCSE method against those produced using existing methods, such as the Elbow method and Gap Statistic method. Since both the Elbow method and Gap Statistic method require a range of k values to test, we can set an initial maximum value Kmax as the range of k to test. for all test cases. In this instance, we choose a reasonable value of $K_{max}$=20 for all testing scenarios. This enables us to explore the effectiveness of the CCSE algorithm and compare its performance with well-established cluster estimation methods.
![image](https://user-images.githubusercontent.com/77912454/230666805-13bfcb29-824d-4660-a5fe-e0a51b7ef895.png)

*B. Quantitative Validation*

A further validation can be done by Color Reconstruction Error-based Segmentation Evaluation (CRESE) which  evaluates the performance of a color-based image segmentation method, such as the CCSE algorithm, by measuring the color reconstruction error in the segmented image. It calculates the color differences between the true color of each pixel and the average color of the corresponding cluster (label) in the segmented image and sums up the color differences for all pixels. A lower color reconstruction error indicates better performance of the segmentation method in estimating the number of color-based clusters.
Let O be the original image of size $H x W x 3$ (height, width, and RGB color channels), and let S be the segmented image of the same size with averaged colors per cluster.
Let $C = \lbrace c_1, c_2, ..., c_n \rbrace$ be the set of unique colors in S, where n is the number of unique colors. For each color ci in C, define Mi as the binary mask of size H x W that has 1s where S has color ci and 0s elsewhere.
The color reconstruction error E is given by:

$$E = \displaystyle\sum_{i=1}^n \displaystyle\sum_{h=1}^H \displaystyle\sum_{w=1}^W M_i (h,w) * \lVert O(h,w) - c_i \rVert$$

Where ${\lVert . \rVert}_2$ denotes the Euclidean distance. And the normalized error E to a percentage scale can be calculated as following:

$$E_{normalized} = \frac{E*100}{H*W*\sqrt{255^2*3}}$$

During the experiment, we found that enhancing image contrast also leads to an increase in the number of k. Since the contrast enhancement process has an O(1) time complexity, the calculation time of CRESE using the original image will not be affected. Consequently, based on the CRESE, we can obtain the validation plot in terms of normalized error vs. the number of clusters (k) for both enhanced images and original images.

![image](https://user-images.githubusercontent.com/77912454/230678918-6a3e5ecc-a198-4d0f-9115-9c4ade8d254a.png)

![image](https://user-images.githubusercontent.com/77912454/230678935-049162e7-5a13-4580-9b36-4b746c15dafa.png)

*C. Computational Efficiency*

In the previous section, we evaluated the accuracy of the three algorithms (CCSE, Elbow Method, and Gap Statistic Method) when processing images of different resolutions. In this section, we focus on comparing their computation times in the context of image segmentation. To do this, we use a high-resolution beach image without enhancement, having a size of 3840*2160 pixels as the test case. This unmodified image allows us to investigate the impact of image size parameters on the performance of each algorithm in their default configurations. To do this, we create resized versions of the original image, maintaining its aspect ratio while reducing its dimensions to 10%, 25%, 33%, 50%, and 75% of the original size. This results in a diverse set of test images with different pixel counts, allowing us to thoroughly examine the efficiency of each algorithm in handling a range of image sizes.

To maintain a fair comparison, we set the maximum number of possible clusters (k) to 20 for both the Elbow Method and Gap Statistic Method. For each resized image, we measure the time taken by each algorithm to estimate the optimal number of clusters and analyze how the computational complexity varies with image size. By comparing the computation times, we can evaluate the efficiency of each algorithm in processing images of different dimensions and identify the most suitable method for real-world applications, where image size may vary significantly. Furthermore, this comparison will highlight the scalability and adaptability of the CCSE in handling diverse image sizes while maintaining accurate and effective segmentation results.

![image](https://user-images.githubusercontent.com/77912454/230679314-df8e5666-798b-4875-a119-1f78c7904915.png)

*D. Sensitivity Analysis*

Sensitivity analysis is a crucial aspect of evaluating the performance of clustering algorithms, as it helps determine the robustness of each method against changes in their input parameters. In the context of our image segmentation study, we can assess the sensitivity of the Elbow Method and Gap Statistic Method by varying the maximum number of clusters (k) and observing how the clustering results and the calculated metrics are affected. Similarly, for the CCSE algorithm, we can examine how varying the distance parameter (D) influences the estimated number of clusters and the overall segmentation quality. By systematically altering these parameters and analyzing the resulting changes in the clustering quality, we can gain insights into the stability and adaptability of each method. A more robust algorithm will exhibit less sensitivity to parameter changes and produce more consistent results across a wider range of parameter values. In our study case, we always set parameter D for CCSE equal to Kmax for Elbow and Gap Statistic Methods, starting from 5 to 40 with an increase step of 5.

![image](https://user-images.githubusercontent.com/77912454/230679400-ca8f51c3-cb21-4285-ab68-5e6dce54ac6c.png)

Gap Statistic method shows a linear relationship between Kmax and the estimated k value, suggesting a high sensitivity to changes in Kmax. The Elbow Method demonstrates a non-linear relationship and somewhat less sensitivity to changes in Kmax  compared to the Gap Statistic method. However, the changes in k are still quite significant as Kmax increases.

In contrast, the CCSE method exhibits a more significant decrease in k as the distance parameter D increases. This suggests that the CCSE method is less sensitive to parameter changes compared to the other two methods, as it shows more consistent results across a wider range of parameter values.

Based on this analysis, the CCSE algorithm appears to have the best robustness among the three methods, as it demonstrates the least sensitivity to changes in its input parameters.

## CONCLUSION

After analyzing results demonstrate above, we can say that the proposed model and CCSE algorithm demonstrates superior performance in various aspects of image segmentation, including runtime efficiency, robustness, visual validation, and quantitative validation.

First, the segmented images produced by the CCSE algorithm appear visually like those produced by the other methods, demonstrating that the algorithm effectively captures the essential characteristics of the input images and provides satisfactory clustering results. We further use the CRESE metric on both original and contrast-enhanced images, the CCSE algorithm shows competitive performance compared to the Elbow method and Gap Statistic method. For original images, CCSE errors range from 2.31 to 3.28, while for enhanced images, the errors range from 2.08 to 3.01. These results indicate that the CCSE algorithm can provide accurate segmentation results, even when dealing with images with different levels of contrast.

On the other hand, the runtime comparison shows the CCSE algorithm significantly outperforms both the Elbow method and Gap Statistic method across various image sizes. With minimal impact on computation time as image size increases, the CCSE algorithm is highly efficient and well-suited for real-world applications, particularly when dealing with high-resolution images. Besides, the CCSE algorithm exhibits excellent robustness as it maintains consistent performance across different parameter settings. This characteristic ensures that the algorithm can adapt to various scenarios and produce reliable results.

Overall, the CCSE algorithm outshines its counterparts in various key aspects, making it a promising choice for image segmentation tasks in a wide range of applications. Its runtime efficiency, robustness, and ability to produce visually and quantitatively accurate results demonstrate its potential for addressing the challenges faced in computer vision and image processing fields. This research provides valuable insights into the advantages of the CCSE algorithm and lays the foundation for further exploration and optimization of its capabilities.
