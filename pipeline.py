import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import pickle
from extract_features import  *
from scipy.ndimage.measurements import label
import time
from collections import deque

def add_global_heat(heatmap, bbox_list_of_lists):
    for bbox_list in bbox_list_of_lists:
        for box in bbox_list:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


#apply thresholds to heatmap images
def apply_threshold(heatmap_in, threshold):
    heatmap_out = np.copy(heatmap_in)
    # Zero out pixels below the threshold
    heatmap_out[heatmap_out <= threshold] = 0
    # Return thresholded map
    return heatmap_out


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


#load the trained model
filename = 'model.dat'
svc = pickle.load(open(filename, 'rb'))
print("model loaded")

#loads the scaler
filename_scaler = 'scaler.dat'
X_scaler = pickle.load(open(filename_scaler, 'rb'))
print("model loaded")

# Keeps only the last x appended box-lists
heatmap_boxes = deque(maxlen=10)


def single_image_pipeline(img):
    draw_img, _, _ = single_image_pipeline_raw(img)
    return draw_img

counter = 0
def single_image_pipeline_with_save(img):
    global counter
    counter += 1
    draw_img, heatmap_raw, heatmap_filtered = single_image_pipeline_raw(img)
    cv2.imwrite("output_images/stream_processed" + str(counter) + ".jpg", cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
    cv2.imwrite("output_images/stream_heatmap_raw" + str(counter) + ".jpg", cv2.cvtColor(heatmap_raw * 100, cv2.COLOR_GRAY2RGB))
    cv2.imwrite("output_images/stream_heatmap_filtered" + str(counter) + ".jpg", cv2.cvtColor(heatmap_filtered * 100, cv2.COLOR_GRAY2RGB))
    return draw_img


def single_image_pipeline_raw(img):
    global heatmap_boxes
    params = get_features_parameters()
    #scales = [0.75, 1.0, 1.5, 2]
    scales = [(350, 500, 1), (400, 600, 1.5), (500, 700, 2.5)]
    #scales = [1, 1.5, 2]
    #threshold = 1
    boxes = find_cars(img, scales, svc, X_scaler,
                        params['orient'], params['pix_per_cell'], params['cell_per_block'], params['spatial_size'], params['hist_bins'], params['color_space'])

    heatmap_boxes.append(boxes)

    heatmap = np.zeros_like(img[:,:,0])
    heatmap_raw = add_global_heat(heatmap, list(heatmap_boxes))
    threshold = len(heatmap_boxes)
    heatmap_filtered = apply_threshold(heatmap_raw, threshold)
    labels = label(heatmap_filtered)
    draw_img = draw_labeled_bboxes(img, labels)
    return draw_img, heatmap_raw, heatmap_filtered

def find_cars(img, scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space):
    #img = img.astype(np.float32)/255

    img_boxes = []

    for (ystart, ystop, scale) in scales:
        img_tosearch = img[ystart:ystop,:,:]

        ctrans_tosearch = convert_color(img_tosearch, color_space=color_space)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient*cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    img_boxes.append(((xbox_left, ytop_draw+ystart),\
                                     (xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    return img_boxes


def process_test_video():
    global counter
    counter = 0
    from moviepy.editor import VideoFileClip
    project_output = "out_test.mp4"
    clip_project = VideoFileClip("test_video.mp4")
    project_clip = clip_project.fl_image(single_image_pipeline_with_save)
    project_clip.write_videofile(project_output, audio=False)

def process_project_video():
    from moviepy.editor import VideoFileClip
    project_output = "out_project.mp4"
    clip_project = VideoFileClip("project_video.mp4")
    project_clip = clip_project.fl_image(single_image_pipeline)
    project_clip.write_videofile(project_output, audio=False)

def test_single_pipeline():
    img = cv2.imread("test_images/test4.jpg")
    start = time.process_time()
    img_dst, heatmap_raw, heatmap_filtered = single_image_pipeline_raw(img)
    #img_dst = single_image_pipeline(img)
    end = time.process_time()
    cv2.imwrite("output_images/heat_raw.jpg", cv2.cvtColor(heatmap_raw * 100, cv2.COLOR_GRAY2RGB))
    cv2.imwrite("output_images/heat_filtered.jpg", cv2.cvtColor(heatmap_filtered * 100, cv2.COLOR_GRAY2RGB))
    cv2.imwrite("output_images/heat_processed.jpg", img_dst)
    print(end - start)

def produce_test_images():
    img1 = cv2.imread("test_images/test1.jpg")
    cv2.imwrite("output_images/processed_test1.jpg", single_image_pipeline(img1))

    img3 = cv2.imread("test_images/test3.jpg")
    cv2.imwrite("output_images/processed_test3.jpg", single_image_pipeline(img3))

    img4 = cv2.imread("test_images/test4.jpg")
    cv2.imwrite("output_images/processed_test4.jpg", single_image_pipeline(img4))

    img6 = cv2.imread("test_images/test6.jpg")
    cv2.imwrite("output_images/processed_test6.jpg", single_image_pipeline(img6))


#test_single_pipeline()
#produce_test_images()
#process_test_video()
process_project_video()
