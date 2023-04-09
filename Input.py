import cv2
import numpy as np
import NLT

def video():
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 100)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)

    # Loop until the end of the video
    while True:
        ret, frame = vid.read()
        # Display the resulting frame
        cv2.imshow('frame', frame)
        cluster = NLT.ClusterMethod(video=frame)
        print("estimated K: ", cluster.k, "color difference:", 100-(NLT.crese(frame, cluster.imshow())))
        cv2.imshow('image segmentation', cluster.imshow())
        # plot every cluster separately
        # binaryImgs = cluster.imClusters()
        # combined_binary_images = np.hstack(binaryImgs)
        # cv2.imshow("Binary Images", combined_binary_images)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()