#  edited python script
"""@author: kyleguan
revised MES 5/22/20
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from moviepy.editor import VideoFileClip
from collections import deque
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
import time

import helpers
import detector
import tracker

# Global variables to be used by functions of VideoFileClip
frame_count = 0 # frame counter

max_age = 4  # no.of consecutive unmatched detection before
             # a track is deleted

min_hits = 1  # no. of consecutive matches needed to establish a track

tracker_list =[] # list for trackers
# list for track ID
track_id_list= deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])

# set the debug flag
debug = False       # True for test on sequence of images in test_images/;  False for video

# instantiate CarDetector object
det = detector.CarDetector()


def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatched trackers, unmatched detections.
    '''

    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk)
        for d, detection in enumerate(detections):
        #   det = convert_to_cv2bbox(det)
            IOU_mat[t,d] = helpers.box_iou2(trk,detection)

    # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)

    matched_idx = linear_assignment(-IOU_mat)

    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
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

    frame_count+=1

    img_dim = (img.shape[1], img.shape[0])
    z_box = det.get_localization(img) # measurement
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
    matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.3)

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
            tmp_trk= tracker_list[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx =[xx[0], xx[1], xx[2], xx[3]]
            x_box[trk_idx] = xx
            tmp_trk.box =xx
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0

    # Deal with unmatched detections
    if len(unmatched_dets)>0:
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker.Tracker() # Create a new tracker
            x = np.array([[z[0], z[1], z[2], z[3], 0, 0, 0, 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[1], xx[2], xx[3]]
            tmp_trk.box = xx
            tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
            tracker_list.append(tmp_trk)
            x_box.append(xx)

    # Deal with unmatched tracks
    if len(unmatched_trks)>0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[1], xx[2], xx[3]]
            tmp_trk.box = xx
            x_box[trk_idx] = xx


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
        #
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
    #
    # MES edits 6/1/20
    # Device configuration
    import torch
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')
    device2 = torch.device('cpu')
    #
    # determine which computer/platform we are running on
    if (os.name == "posix"):
        os_list = os.uname()
        if (os_list[0] == "Darwin"):
            pf_detected = 'MAC'
        elif (os_list[0] == "Linux"):
            if (os_list[1] == 'en4119351l'):
                pf_detected = 'Quadro'
            elif (os_list[1] == '19fef43c2174'):
                pf_detected = 'Exxact'
            elif (os_list[1] == 'EN4113948L'):
                pf_detected = 'Kevin'
    else:
        pf_detected = 'PC'
    #
    # debug
    print("os.name: ", os.name)
    print("os_list[0]: ", os_list[0])
    print("os_list[1]: ", os_list[1])
    exit()
    #
    # set the root path based on the computer/platform
    #   rootPath is path to directory in which webots/ and imdata/ directories reside
    if (pf_detected == 'MAC'):
        rootPath = '/Users/mes/Documents/ASU-Classes/Research/Ben-Amor/code/'
        device = device2

    elif (pf_detected == 'Quadro'):
        rootPath = '/home/local/ASUAD/mestric1/Documents/AVCES/'

    elif (pf_detected == 'Exxact'):
        rootPath = '/home/dockeruser/Documents/AVCES/'
        device = device1
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # gpu_ids = [0, 1]
        gpu_ids = [1]

    elif (pf_detected == 'Kevin'):
        rootPath = '/home/local/ASUAD/mestric1/Documents/AVCES/'
        device = device0
        gpu_ids = None

    elif (pf_detected == 'PC'):
        # rootPath = 'C:\Users\cesar\Desktop\Furi\'
        print("PC platform detected.  Exiting.")
        exit()
    else:
        print("Computer/Platform not detected.  Exiting.")
        exit()
    #
    # instantiate CarDetector object
    # det = detector.CarDetector()

    if debug: # test on a sequence of images
        images = [plt.imread(file) for file in glob.glob('./test_images/*.jpg')]

        for i in range(len(images))[0:7]:
             image = images[i]
             image_box = pipeline(image)
             out_filepath = './test_images/out{:06d}.jpg'.format(i+1)
             plt.imsave(out_filepath, image_box)

    else: # test on a video file.
        # start the clock
        start=time.time()
        input = 'project_video.mp4'
        output = 'test_FRCNN.mp4'
        clip1 = VideoFileClip(input)#.subclip(4,49) # The first 8 seconds doesn't have any cars...
        clip = clip1.fl_image(pipeline)
        clip.write_videofile(output, audio=False)

        # stop the clock
        end  = time.time()
        # print the elapsed time
        print(round(end-start, 2), 'Seconds to finish')
#
# end of main()
#

if __name__ == "__main__":
    main()
