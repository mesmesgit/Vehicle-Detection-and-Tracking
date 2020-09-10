#
# save_box_sets_yolo.py
#
#   Script to run detector and tracker for a video,
#       and save array of organized bounding
#       boxes for vehicles in the video.
#
#   Rev MES 9/5/2020 to substitue YOLO for Mask-RCNN object detector
#
import os
import time
import numpy as np
import cv2 as cv
from scipy.optimize import linear_sum_assignment
from collections import deque
from moviepy.editor import VideoFileClip
import math
# local imports
import helpers_FRCNN as helpers
import detector_YOLOv3 as detector
import tracker_FRCNN as tracker

# Global variables
dataset = None
frame_count = 0  # frame counter

max_age = 5     # no.of consecutive unmatched detection before track is deleted

min_hits = 3  # no. of consecutive matches needed to establish a track

tracker_list = []  # list for trackers
# list for track ID
track_id_ref = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
#  assign random color to each track ID
colors = np.random.randint(0, 255, size=(len(track_id_ref), 3), dtype='uint8')

track_id_list = deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                       'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T'])

# set the debug flag
debug = False  # True for test on sequence of images in test_images/;  False for video

# instantiate CarDetector object
det = detector.CarDetector()

track_debug = set()

tracking_obstructed = set()

current_image = None
cur_z_box = None


def assign_detections_to_trackers(xbox, zbox, confidence_zbox, iou_thrd=0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatched trackers, unmatched detections.
    '''
    trackers = xbox
    detections = zbox

    IOU_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    for t, trk in enumerate(trackers):
        for d, detection in enumerate(detections):
            # det = convert_to_cv2bbox(det)
            iou_result = helpers.box_iou2(trk, detection)
            if math.isnan(iou_result) is True:
                iou_result = 0.99
            IOU_mat[t, d] = iou_result

    # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)

    # matched_idx = linear_assignment(-IOU_mat)     # sklearn.utils
    try:
        matched_idx = linear_sum_assignment(-IOU_mat)         # scipy.optimize
    except ValueError:
        print(" ")
        print("ValueError in linear_sum_assignment.")
        print("IOU_mat: ", IOU_mat)
        print("trackers: ", trackers)
        print("detections: ", detections)
        exit()
    matched_idx = np.asarray(matched_idx)
    matched_idx = np.transpose(matched_idx)

    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if(t not in matched_idx[:, 0]):
            # see whether it is a false alarm (obscured)
            if(tracker_list[t].hits > min_hits):
                xx = tracker_list[t].predict_only_no_update()
                xx = xx.T[0].tolist()
                xx = [xx[0], xx[3], xx[6], xx[9]]
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
        if(d not in matched_idx[:, 1]):
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signify the existence of
    # an untracked object

    for m in matched_idx:
        if(IOU_mat[m[0], m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if(len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
#
#  end function assign_detections_to_trackers()
#


def process_image(img):
    '''
    Pipeline function for detection and tracking
    '''
    global dataset
    global frame_count
    global max_age
    global min_hits
    global tracker_list
    global track_id_ref
    global colors
    global track_id_list
    global debug
    global det
    global track_debug
    global tracking_obstructed
    global current_image
    global cur_z_box

    frame_count += 1
    current_image = img
    img_dim = (img.shape[1], img.shape[0])
    z_box, confidence_zbox = det.get_localization(img, dataset)  # measurement
    #
    # debug
    # print(" ")
    # print("frame_count: ", frame_count)
    # print("z_box: ", z_box)
    # print("confidence_zbox: ", confidence_zbox)
    #
    # limit number of boxes to 20
    z_box = z_box[:20]
    confidence_zbox = confidence_zbox[:20]
    #
    cur_z_box = z_box
    if debug:
        print('Frame:', frame_count)

    x_box = []
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
    if matched.size > 0:
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
            xx = [xx[0], xx[3], xx[6], xx[9]]
            x_box[trk_idx] = xx
            tmp_trk.box = xx
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0

    # Deal with unmatched detections
    if len(unmatched_dets) > 0:
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker.Tracker()  # Create a new tracker
            x = np.array([[z[0], 0, 0, z[1], 0, 0, z[2], 0, 0, z[3], 0, 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[3], xx[6], xx[9]]
            tmp_trk.box = xx
            try:
                tmp_trk.id = track_id_list.popleft()  # assign an ID for the tracker
            except IndexError:
                continue
            tracker_list.append(tmp_trk)
            x_box.append(xx)

    # Deal with unmatched tracks
    if len(unmatched_trks) > 0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[3], xx[6], xx[9]]
            tmp_trk.box = xx
            x_box[trk_idx] = xx
            tmp_trk.no_losses += 1

    # The list of tracks to be annotated
    good_tracker_list = []
    final_boxes = []
    final_ids = []
    for trk in tracker_list:
        if ((trk.hits >= min_hits) and (trk.no_losses <= max_age)):
            good_tracker_list.append(trk)
            x_cv2 = trk.box
            final_boxes.append(helpers.convert_cxcy_to_cv2bbox(trk.box))
            final_ids.append(trk.id)
            if debug:
                print('updated box: ', x_cv2)
                print()
            img = helpers.draw_box_label(img, x_cv2)  # Draw the bounding boxes on the images

    max_obstruction_count = 20
    kill_obstructed_track = [x for x in tracker_list if x.obstruction_count > max_obstruction_count]
    for x in kill_obstructed_track:
        x.no_losses = max_obstruction_count

    #
    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses > max_age, tracker_list)

    for trk in deleted_tracks:
        track_id_list.append(trk.id)

    tracker_list = [x for x in tracker_list if x.no_losses <= max_age]

    if debug:
        print('Ending tracker_list: ', len(tracker_list))
        print('Ending good tracker_list: ', len(good_tracker_list))
    #
    return img, final_boxes, final_ids
#
# end function process_image()


def pipeline(img):
    global frame_count
    img, final_boxes, final_ids = process_image(img)
    print(" ")
    print(" ")
    print("Box output for frame {}".format(frame_count))
    print("final_boxes: ", final_boxes)
    print("final_ids: ", final_ids)
    return img


def process_video(rootPath,
                  video_filepath,
                  out_data_filepath,
                  runparams_filepath=None):
    # globals
    global dataset
    global frame_count
    global max_age
    global min_hits
    global tracker_list
    global track_id_ref
    global colors
    global track_id_list
    global debug
    global det
    global track_debug
    global tracking_obstructed
    global current_image
    global cur_z_box
    # start clock
    time_start = time.time()
    # re-init the deque and other stuff needed for a new video
    frame_count = 0  # frame counter
    tracker_list = []  # list for trackers
    track_id_list = deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T'])

    # instantiate CarDetector object
    det = detector.CarDetector()
    track_debug = set()
    tracking_obstructed = set()
    current_image = None
    cur_z_box = None

    # get video filename
    fparts = video_filepath.split('/')
    vid_filename = fparts[-1]
    #  run in a loop from the video
    cap = cv.VideoCapture(video_filepath)
    #  get the number of frames in the video
    amount_frames_cap = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    # get runparams info
    if runparams_filepath is not None:
        with np.load(runparams_filepath) as f:
            run_id = str(f['run_id'])
            collision_class = int(f['collision_class'])
            amount_frames = int(f['amount_frames'])
            start_frame = int(f['start_frame'])
            end_frame = int(f['end_frame'])
    else:
        gparts = vid_filename.split('.')
        run_id = gparts[0]
        collision_class = 0
        amount_frames = amount_frames_cap
        start_frame = 0
        end_frame = min(150, amount_frames_cap)

    #
    # debug
    # print(" ")
    # print(" ")
    # print("START OF run_id: ", run_id)
    # print("collision_class: ", collision_class)
    # print("amount_frames: ", amount_frames)
    # print("start_frame: ", start_frame)
    # print("end_frame: ", end_frame)
    # print(" ")
    # print("Frame Count from cap.get()")
    # print("Video / Frame_Count:")
    # print("{} / {}".format(vid_filename, amount_frames_cap))
    # check for agreement with runparams
    if amount_frames_cap != amount_frames:
        print("ERROR:  runparams differs from cap.get() for amount_frames.")
        print("amount_frames = ", amount_frames)
        print("amount_frames_cap = ", amount_frames_cap)
        exit()

    # create the data array
    num_boxes_max = 20
    data = np.zeros([150, num_boxes_max, 4])    # featv6:  150 frames @ 30fps
    #
    count = 0
    stored_frame_idx = 0
    while count < amount_frames:
        #  read a frame from the video
        ret, frame_orig = cap.read()
        if ret is None:
            print("Read attempt for frame {} returned None".format(count))
            count += 1
            continue
        # skip to the start_frame, stop processing after end_frame
        if count < start_frame:
            print("frame {} before start_frame, skipped.".format(count))
            count += 1
            continue
        elif count >= end_frame:
            print("frame {} after end_frame, skipped.".format(count))
            count += 1
            continue

        #
        # debug - get info about image
        # print(" ")
        # print("Read attempt for frame {} was successful".format(count))
        # print("type of frame: ", type(frame_orig))
        # print("shape of frame: ", np.shape(frame_orig))
        # print("value of first pixel of first frame: ", frame_orig[0,0,0])
        # print("type of value of first pixel: ", type(frame_orig[0,0,0]))
        # exit()

        # process the image to create bounding boxes
        img, final_boxes, final_ids = process_image(frame_orig)
        if len(final_boxes) > 0:
            # update data
            for idx, box in enumerate(final_boxes):
                if idx >= num_boxes_max:
                    print("***********")
                    print("WARNING: more than 20 bboxes in single frame!")
                    print("run_id {}, frame {}".format(run_id, count))
                    print("idx = {}".format(idx))
                    print("***********")
                    continue
                # get values in final_boxes (x1, y1, x2, y2)
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                # determine column or box slot based on track id
                box_slot = track_id_ref.index(final_ids[idx])
                # put final box values into data array
                data[stored_frame_idx, box_slot, 0] = x1
                data[stored_frame_idx, box_slot, 1] = y1
                data[stored_frame_idx, box_slot, 2] = x2
                data[stored_frame_idx, box_slot, 3] = y2
            #
        #  increment frame count
        # print(" ")
        print("frame {} complete.".format(count))
        # print("final_boxes: ", final_boxes)
        # print("final_ids: ", final_ids)
        # print("stored_frame_idx = {}.".format(stored_frame_idx))
        # print("data[stored_frame_idx]: ", data[stored_frame_idx])
        count += 1
        stored_frame_idx += 1
    # end loop while count
    # clean up
    cv.destroyAllWindows()
    cap.release()
    # timers
    time_elap = time.time() - time_start
    # debug
    print(" ")
    print("END OF run_id: ", run_id)
    print("collision_class: ", collision_class)
    print("amount_frames: ", amount_frames)
    print("start_frame: ", start_frame)
    print("end_frame: ", end_frame)
    # print("data: ", data)
    print("shape of data: ", np.shape(data))
    print("type of data: ", type(data))
    print("time for this run = {} seconds.".format(time_elap))
    # exit()
    #
    # save output data
    np.savez(out_data_filepath,
             run_id=run_id,
             collision_class=collision_class,
             amount_frames=amount_frames,
             start_frame=start_frame,
             end_frame=end_frame,
             data=data)
    #
# end of process_video()


def find_run_id(input):
    fparts = input.split('.')
    gparts = fparts[0].split('_')
    run_id_prefix = gparts[0]
    if run_id_prefix[0] == 'c' or run_id_prefix[0] == 'p':
        # sim run
        run_id = gparts[0] + '_' + gparts[1]
        run_series = run_id_prefix
        ds = "sim"
    elif run_id_prefix[0] == 's':
        # ss run
        run_id = gparts[0]
        run_series = run_id
        ds = "ss"
    elif run_id_prefix[0] == 'C' or run_id_prefix[0] == 'y':
        # vid run
        run_id = gparts[0]
        hparts = run_id_prefix.split('-')
        run_series = hparts[0]
        ds = "vid"
    else:
        print(" ")
        print("ERROR:  Unknown run_id for input={}.  Exiting.".format(input))
        exit()
    return run_id, run_series, ds


def process_dataset():
    # globals
    global dataset
    global max_age
    global min_hits
    # global-set specific values
    if dataset == "sim":
        max_age = 5     # no.of consecutive unmatched detection before track is deleted
        min_hits = 3  # no. of consecutive matches needed to establish a track
    elif dataset == "ss":
        max_age = 5     # no.of consecutive unmatched detection before track is deleted
        min_hits = 1  # no. of consecutive matches needed to establish a track
    elif dataset == "vid":
        max_age = 5     # no.of consecutive unmatched detection before track is deleted
        min_hits = 3  # no. of consecutive matches needed to establish a track
    else:
        max_age = 5     # no.of consecutive unmatched detection before track is deleted
        min_hits = 3  # no. of consecutive matches needed to establish a track
    #
    # set rootPath
    rootPath = "/home/mes/Documents/AVCES/"
    # set npzPath, which holds result of class- and frame-level review
    npzPath = rootPath + "imdata/dataset/" + dataset + "/runparams/"
    # need to get list of runids
    list_npz = sorted(os.listdir(npzPath))
    # list_npz = ["ss28.npz"]
    #
    for npz in list_npz:
        # get runparams info
        runparams_filepath = npzPath + npz
        # get run_series and set video filepath
        run_id, run_series, ds = find_run_id(npz)
        if ds != dataset:
            print(" ")
            print("ERROR:  ds does not match dataset, exiting.")
            exit()
        #
        if run_series[0] == 'c' or run_series[0] == 'p':
            vidPath = rootPath + "imdata/dataset/sim/videos/"
            video_filepath = vidPath + run_id + "_rgb.mp4"
        elif run_series[0] == 's':
            vidPath = rootPath + "imdata/dataset/ss/videos/"
            video_filepath = vidPath + run_id + "_rgb.mp4"
        elif run_series[0] == 'y':
            vidPath = rootPath + "imdata/dataset/vid/videos/"
            video_filepath = vidPath + run_id + ".mp4"
        elif run_series == 'CCT060':
            vidPath = rootPath + "imdata/downloads-YouTube/CCT060/"
            video_filepath = vidPath + run_id + ".mp4"
        elif run_series == 'CCT185':
            vidPath = rootPath + "imdata/downloads-YouTube/CCT185/"
            video_filepath = vidPath + run_id + ".mp4"
        else:
            with np.load(runparams_filepath) as f:
                collision_class = int(f['collision_class'])
            #
            if collision_class == 0:
                vidPath = rootPath + "imdata/downloads-YouTube/keep-nocoll/"
                video_filepath = vidPath + run_id + ".mp4"
            else:
                vidPath = rootPath + "imdata/downloads-YouTube/keep-yescoll/"
                video_filepath = vidPath + run_id + ".mp4"
            #
        #
        #  set output .npz filepath
        featv8_npz_filepath = rootPath + "imdata/dataset/" + dataset + \
                              "/features/" + run_id + "_featv8.npz"
        #
        #
        #
        process_video(rootPath,
                      video_filepath,
                      featv8_npz_filepath,
                      runparams_filepath)
        #
        #
        #
    # end of for loop iterating thru npzs
# end of process_dataset()


def main():
    # globals
    global dataset
    test = 0
    if test == 1:
        # start the clock
        start = time.time()
        input = './project_video.mp4'
        output = './project_video_overlay.mp4'
        clip1 = VideoFileClip(input).subclip(29, 37)  # The first 8 seconds doesn't have any cars...
        clip = clip1.fl_image(pipeline)
        clip.write_videofile(output, audio=False)

        # stop the clock
        end = time.time()
        # print the elapsed time
        print("Time elapsed: {} seconds.".format(round(end-start, 2)))
    elif test == 2:
        dataset = "sim"
        rootPath = "/home/mes/Documents/AVCES/"
        video_filepath = rootPath + "imdata/dataset/sim/videos/czb6_002_rgb.mp4"
        out_data_filepath = rootPath + "imdata/dataset/sim/features/czb6_002_featv8.npz"
        runparams_filepath = rootPath + "imdata/dataset/sim/runparams/czb6_002.npz"
        process_video(rootPath,
                      video_filepath,
                      out_data_filepath,
                      runparams_filepath)
    else:
        dataset = "vid"
        process_dataset()
# end of main()


if __name__ == "__main__":
    main()
