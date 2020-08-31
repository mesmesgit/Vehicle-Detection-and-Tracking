"""@author: kyleguan
revised MES 5/22/20
@updated by Michael Drolet 8/30/20
"""
# installed module imports
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from moviepy.editor import VideoFileClip
from collections import deque
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
import time
# local imports
import helpers_FRCNN as helpers
import detector_FRCNN as detector
import tracker_FRCNN as tracker
import copy

# Global variables to be used by functions of VideoFileClip
frame_count = 0 # frame counter

max_age = 3  # no.of consecutive unmatched detection before
             # a track is deleted

min_hits = 3  # no. of consecutive matches needed to establish a track

tracker_list =[] # list for trackers
# list for track ID
track_id_list= deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])

# set the debug flag
debug = False       # True for test on sequence of images in test_images/;  False for video

# instantiate CarDetector object
det = detector.CarDetector()

track_debug = set()

tracking_obstructed = set()

current_image = None
cur_z_box = None


def pbox(*argv):
    box = []
    if len(argv) == 0:
        box = cur_z_box
    else:
        box = [x for x in argv]

    # if xx style is passed in, change it to simple vector
    for i, b in enumerate(box):
        if (type(b[0]) == np.ndarray or type(b[0]) == list):
            box[i] = [x[0] for x in b]

    pic = copy.copy(current_image)
    for i in range(len(box)):
       img1 = helpers.draw_box_label(pic, box[i], box_color=(255, 0, 0))
       plt.imshow(img1)
    plt.show()


def assign_detections_to_trackers(xbox, zbox, confidence_zbox, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatched trackers, unmatched detections.
    '''
    trackers = xbox
    detections = zbox

    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        for d, detection in enumerate(detections):
        #   det = convert_to_cv2bbox(det)
            IOU_mat[t,d] = helpers.box_iou2(trk,detection)

    # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)

    # matched_idx = linear_assignment(-IOU_mat)     # sklearn.utils
    matched_idx = linear_sum_assignment(-IOU_mat)         # scipy.optimize
    matched_idx = np.asarray(matched_idx)
    matched_idx = np.transpose(matched_idx)

    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            # see whether it is a false alarm (obscured)
            if(tracker_list[t].hits > min_hits):
                xx = tracker_list[t].predict_only_no_update()
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[3], xx[6], xx[9]]
                max_overlap = 0
                for box_found in detections:
                    if (helpers.box_iou2(xx, box_found) > 0):
                        max_overlap = max(max_overlap, helpers.obscured_overlap(box_found, xx))
                if(max_overlap < 0.85):
                    unmatched_trackers.append(t)
                else:
                    tracker_list[t].obstruction_count += 1
            else:
                unmatched_trackers.append(t)
    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signify the existence of
    # an untracked object

    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
#
#  end function assign_detections_to_trackers()
#


def pipeline(img):
    '''
    Pipeline function for detection and tracking
    '''
    global frame_count
    global tracker_list
    global max_age
    global min_hits
    global track_id_list
    global debug
    global det
    global current_image
    global cur_z_box

    frame_count+=1
    current_image = img
    img_dim = (img.shape[1], img.shape[0])
    z_box, confidence_zbox = det.get_localization(img) # measurement
    cur_z_box = z_box
    if debug:
       print('Frame:', frame_count)

    x_box =[]
    # if debug:
    #     for i in range(len(z_box)):
    #        img1= helpers.draw_box_label(img, z_box[i], box_color=(255, 0, 0))
    #        plt.imshow(img1)
    #     plt.show()

    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)

    # perform detection-tracker matching
    matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(x_box, z_box, confidence_zbox, iou_thrd = 0.2)

    if debug:
         print('Detection: ', z_box)
         print('x_box: ', x_box)
         print('matched:', matched)
         print('unmatched_det:', unmatched_dets)
         print('unmatched_trks:', unmatched_trks)


    # Deal with matched detections
    if matched.size >0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker_list[trk_idx]

            # Update measurement noise based on the confidence of the RCNN
            tmp_trk.R = np.eye(tmp_trk.z_dim) * (tmp_trk.R_std**2/(confidence_zbox[det_idx]**6))

            x_st, p_st = tmp_trk.kalman_filter(z)
            tmp_trk.x_hist.append(x_st)
            tmp_trk.p_hist.append(p_st)
            track_debug.add(tmp_trk)
            xx = tmp_trk.x_state.T[0].tolist()
            xx =[xx[0], xx[3], xx[6], xx[9]]
            x_box[trk_idx] = xx
            tmp_trk.box = xx
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0

    # Deal with unmatched detections
    if len(unmatched_dets)>0:
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker.Tracker() # Create a new tracker
            x = np.array([[z[0], 0, 0, z[1], 0, 0, z[2], 0, 0, z[3], 0, 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[3], xx[6], xx[9]]
            tmp_trk.box = xx
            tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
            tracker_list.append(tmp_trk)
            x_box.append(xx)

    # Deal with unmatched tracks
    if len(unmatched_trks)>0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[3], xx[6], xx[9]]
            tmp_trk.box = xx
            x_box[trk_idx] = xx
            tmp_trk.no_losses += 1

    # The list of tracks to be annotated
    good_tracker_list =[]
    for trk in tracker_list:
        if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
             good_tracker_list.append(trk)
             x_cv2 = trk.box
             if debug:
                 print('updated box: ', x_cv2)
                 print()
             img= helpers.draw_box_label(img, x_cv2) # Draw the bounding boxes on the images

    max_obstruction_count = 20
    kill_obstructed_track = [x for x in tracker_list if x.obstruction_count > max_obstruction_count]
    for x in kill_obstructed_track:
        x.no_losses = max_obstruction_count

    #
    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)

    for trk in deleted_tracks:
            track_id_list.append(trk.id)

    tracker_list = [x for x in tracker_list if x.no_losses<=max_age]

    if debug:
       print('Ending tracker_list: ',len(tracker_list))
       print('Ending good tracker_list: ',len(good_tracker_list))
    #
    return img
#
# end function pipeline()
#

def main():
    if debug: # test on a sequence of images
        images = [plt.imread(file) for file in sorted(glob.glob('./test_images/*.jpg'))]

        for i in range(len(images)):
             image = images[i]
             image_box = pipeline(image)
             out_filepath = './test_images/out/out{:06d}.jpg'.format(i+1)
             plt.imsave(out_filepath, image_box)

        plot_ps()
        plot_xs()

    else: # test on a video file.
        # start the clock
        start = time.time()
        input = './project_video.mp4'
        output = './project_video_overlay.mp4'
        clip1 = VideoFileClip(input).subclip(4,49) # The first 8 seconds doesn't have any cars...
        clip = clip1.fl_image(pipeline)
        clip.write_videofile(output, audio=False)

        # stop the clock
        end = time.time()
        # print the elapsed time
        print("Time elapsed: {} seconds.".format(round(end-start, 2)))
#
# end of main()
#

def plot_ps():
    all_tr_ps = []
    for tr in track_debug:
        tr_diag_hist_mat = []
        for p in tr.p_hist:
            ps = np.diagonal(p)
            tr_diag_hist_mat.append(ps)
        all_tr_ps.append(np.array(tr_diag_hist_mat))

    plots = dict()
    for tr in all_tr_ps:
        for idx in range(tr.shape[1]):
            if idx in plots:
                plots[idx].append(tr[:,idx])
            else:
                plots[idx] = []
                plots[idx].append(tr[:,idx])

    state_vars = ['cx', 'vx', 'ax', 'cy', 'vy', 'ay', 'w', 'vw', 'aw', 'h', 'vh', 'ah']

    fig, axs = plt.subplots(nrows = len(track_debug), ncols = len(plots.keys()))
    for idx, key in enumerate(plots.keys()):
        for i in range(len(plots[key])):
            data = np.around(plots[key][i], 8)
            axs[i, idx].plot(np.sqrt(data))
            axs[i, idx].plot(-1*np.sqrt(data))
        axs[0,idx].set_title(state_vars[idx])

    for chart in range(len(axs[:,0])):
        axs[chart,0].set_ylabel('Tracker ' + str(chart + 1) + ' - Variance')

    fig.suptitle('Tracker 1 stddev over time')
    plt.show()

def plot_xs():
    state_vars = ['cx', 'vx', 'ax', 'cy', 'vy', 'ay', 'w', 'vw', 'aw', 'h', 'vh', 'ah']
    all_tr_xs = []
    for tr in track_debug:
        tmp = np.array(tr.x_hist)
        all_tr_xs.append(tmp.reshape(tmp.shape[0], tmp.shape[1]))

    fig, axs = plt.subplots(nrows = len(all_tr_xs), ncols = len(state_vars))
    for row_idx, tr_x in enumerate(all_tr_xs):
        for col_idx in range(len(state_vars)):
            axs[row_idx, col_idx].plot(np.linspace(0, 1, num=len(tr_x[:, col_idx])), tr_x[:, col_idx])

    for chart in range(len(axs[:,0])):
        axs[chart,0].set_ylabel('Tracker ' + str(chart + 1) + ' - State Value')

    for col_idx in range(len(state_vars)):
        axs[0, col_idx].set_title(state_vars[col_idx])

    fig.suptitle('Tracker state values over time')
    plt.show()

if __name__ == "__main__":
    main()
