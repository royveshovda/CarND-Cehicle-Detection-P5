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

def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

#apply thresholds to heatmap images
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

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


def single_image_pipeline(img):
    params = get_features_parameters()
    #scales = [0.75, 1.0, 1.5, 2]
    scales = [1, 1.5, 2]
    threshold = 3
    boxes = find_cars(img, (400, 700), scales, svc, X_scaler,
                        params['orient'], params['pix_per_cell'], params['cell_per_block'], params['spatial_size'], params['hist_bins'], params['color_space'])
    heatmap = np.zeros_like(img[:,:,0])
    heatmap = add_heat(heatmap, boxes)
    heatmap = apply_threshold(heatmap, threshold)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(img, labels)
    return draw_img

def find_cars(img, y_limits, scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space):
    #img = img.astype(np.float32)/255

    img_tosearch = img[y_limits[0]:y_limits[1],:,:]
    img_boxes = []

    for scale in scales:
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
                    img_boxes.append(((xbox_left, ytop_draw+y_limits[0]),\
                                     (xbox_left+win_draw,ytop_draw+win_draw+y_limits[0])))
    return img_boxes


def process_test_video():
    from moviepy.editor import VideoFileClip
    project_output = "out_test.mp4"
    clip_project = VideoFileClip("test_video.mp4")
    project_clip = clip_project.fl_image(single_image_pipeline)
    project_clip.write_videofile(project_output, audio=False)

def process_project_video():
    from moviepy.editor import VideoFileClip
    project_output = "out_project.mp4"
    clip_project = VideoFileClip("project_video.mp4")
    project_clip = clip_project.fl_image(single_image_pipeline)
    project_clip.write_videofile(project_output, audio=False)

def test_single_pipeline():
    img = cv2.imread("test_images/test6.jpg")
    start = time.process_time()
    img_dst = single_image_pipeline(img)
    end = time.process_time()
    cv2.imwrite("output_images/processed.jpg", img_dst)
    #img_dst_heat = cv2.cvtColor(heatmap * 255, cv2.COLOR_GRAY2RGB)
    #cv2.imwrite("output_images/processed_heatmap.jpg", img_dst_heat)
    print(end - start)

#img = cv2.imread("test_images/test6.jpg")
#img_dst = single_image_pipeline(img)
#cv2.imwrite("output_images/processed.jpg", img_dst)
#process_test_video()
process_project_video()
#test_single_pipeline()
#play_video('test_video.mp4',"test_output1.avi")
#play_video('project_video.mp4',"output1.avi")
