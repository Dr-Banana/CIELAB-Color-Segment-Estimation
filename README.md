# CIELAB Color Segment Estimation (CCSE)

## Abstract

In the field of image analysis, computer algorithms must recognize and extract essential features or patterns within visual data to accurately represent objects, scenes, or activities. Color information is a critical aspect of object classification, as color differences enable humans to identify the presence of one or more objects. This research introduces a novel method for determining the optimal number of color clusters in CIELAB color space images, using mathematical expressions and K-Means as image segmentation method. Our approach combines perceptual lightness, red, green, blue, and yellow histogram standard deviations, along with the number of local maximum points, by employing a non-linear equation to calculate the ideal number of clusters. Experimental results reveal that the proposed method surpasses widely used techniques such as the Elbow Method and Gap Statistic Method in terms of time. Moreover, our algorithm demonstrates exceptional accuracy in processing visual data within real-world environments, maintaining computational efficiency even as the number of clusters increases.

## Files

- main.py

- NLT.py
  > The file that contains CCSE method, elbow method, and gap statistic method.

- Plot.py
  > The file used to plot runtime, robustness, and accuracy experiment data.
  
- PreProcess.py
  > The file used to do image preprocessing such as convert image to gray scale or enhance image contrast.

## Math

*A. Proposed Method*

The method proposed in this paper can be divided into three parts. The first part is to transform three-channel RGB color into three-channel LAB and extract the histogram of L, A and B respectively. The second part is to calculate the optimal standard deviation and number of local maxima from three channels respectively as parameters. Finally, the best number of clusters is determined using the CIELAB Color Segment Estimation (CCSE) function, as shown below:

$$
k = \lfloor ln(\frac{\sigma_{L}}{n_L})\rfloor max(n_a,n_b)+min(n_a,n_b)
$$

In this equation, $\sigma_{L}$ denotes the standard deviation of L histogram and $n_L$, $n_a$, $n_b$ represent the number of local maxima extracted from the L, a, and b histograms in the second part. $max⁡(n_a,n_b)$ refers to the larger value between n_a and n_b, while $min(n_a,n_b)$ indicates the smaller value. 
The detailed steps of the designed method can be found in the pipeline presented in Figure 3. It is important to note that this method specifically requires input images in the three-channel RGB color format.
