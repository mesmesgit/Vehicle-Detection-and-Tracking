#  Revised python script
#  Rev MES 5/22/20
# @updated by Michael Drolet 8/30/20
'''
Implement and test tracker
'''
import numpy as np
from numpy import dot
from scipy.linalg import inv, block_diag


class Tracker(): # class for Kalman Filter-based tracker
    def __init__(self):
        # Initialize parametes for tracker (history)
        self.id = 0  # tracker's id
        self.box = [] # list to store the coordinates for a bounding box:  [cx, cy, w, h]
        self.hits = 0 # number of detection matches
        self.no_losses = 0 # number of unmatched tracks (track loss)

        # Initialize parameters for Kalman Filtering
        #  State Definition
        # original state: [up, up_dot, left, left_dot, down, down_dot, right, right_dot]
        # or[up, up_dot, left, left_dot, height, height_dot, width, width_dot]
        #
        #  New State Defn:
        # [center x, velocity x, acceleration x, "" y, "" y, "" y, width, ..., ..., height, ... , ...]
        # ['cx', 'vx', 'ax', 'cy', 'vy', 'ay', 'w', 'vw', 'aw', 'h', 'vh', 'ah']
        #
        self.x_state=[]
        self.dt = 1.   # time interval
        self.sigma = 2
        self.state_dim = 12
        self.z_dim = 4
        self.P_std = 2
        self.R_std = 2
        self.obstruction_count = 0
        # Process matrix, assuming constant velocity model
        self.F_comp_mat = np.array([[1, self.dt, self.dt**2/2.],
                                    [0, 1, self.dt],
                                    [0, 0, 1]])

        # self.F_p = np.array([[1, self.dt, 0,  0,  0,  0,  0, 0],
        #                    [0, 1,  0,  0,  0,  0,  0, 0],
        #                    [0, 0,  1,  self.dt, 0,  0,  0, 0],
        #                    [0, 0,  0,  1,  0,  0,  0, 0],
        #                    [0, 0,  0,  0,  1,  self.dt, 0, 0],
        #                    [0, 0,  0,  0,  0,  1,  0, 0],
        #                    [0, 0,  0,  0,  0,  0,  1, self.dt],
        #                    [0, 0,  0,  0,  0,  0,  0,  1]])

        self.F = block_diag(self.F_comp_mat, self.F_comp_mat,
                            self.F_comp_mat, self.F_comp_mat)

        # Measurement matrix, assuming we can only measure the coordinates

        # self.H_p = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
        #                    [0, 0, 1, 0, 0, 0, 0, 0],
        #                    [0, 0, 0, 0, 1, 0, 0, 0],
        #                    [0, 0, 0, 0, 0, 0, 1, 0]])

        self.H = np.zeros((self.z_dim, self.state_dim))
        for row in range(self.H.shape[0]):
            self.H[row, int(row*(self.state_dim/self.z_dim))] = 1


        # Initialize the state covariance
        self.P = np.eye(self.state_dim) * self.P_std**2


        # Initialize the process covariance
        # self.Q_comp_mat_p = np.array([[self.dt**4/4., self.dt**3/2.],
        #                             [self.dt**3/2., self.dt**2]])

        self.Q_comp_mat = np.array([[self.dt**4/4., self.dt**3/2., self.dt**2/2.],
                                    [self.dt**3/2., self.dt**2, self.dt],
                                    [self.dt**2/2., self.dt, 1]])

        self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat,
                            self.Q_comp_mat, self.Q_comp_mat)

        self.Q = self.Q * self.sigma**2

        # Initialize the measurement covariance
        self.R = np.eye(self.z_dim) * self.R_std**2

        self.x_hist = []
        self.p_hist = []

    def update_R(self):
        R_diag_array = self.R_scaler * np.ones(self.state_dim)
        self.R = np.diag(R_diag_array)


    def kalman_filter(self, z):
        '''
        Implement the Kalman Filter, including the predict and the update stages,
        with the measurement z
        '''
        #x = self.x_state
        # Predict
        self.x_state = dot(self.F, self.x_state)

        self.P = 1. * dot(dot(self.F, self.P), self.F.T) + self.Q

        #Update
        y = z - dot(self.H, self.x_state) # residual
        PHT = dot(self.P, self.H.T)
        #S = dot(self.H, self.P).dot(self.H.T) + self.R
        S = dot(self.H, PHT) + self.R
        SI = np.linalg.inv(S)
        #K = dot(self.P, self.H.T).dot(inv(S)) # Kalman gain
        K = dot(PHT, SI)
        self.x_state = self.x_state + dot(K, y)
        #self.P = self.P - dot(K, self.H).dot(self.P)
        I_KH = np.eye(len(self.x_state)) - dot(K, self.H)

        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(K, self.R), K.T)

        self.x_state = self.x_state # .astype(int) # convert to integer coordinates
                                     #(pixel values)
        return self.x_state, self.P

    def predict_only(self):
        '''
        Implment only the predict stage. This is used for unmatched detections and
        unmatched tracks
        '''
        x = self.x_state
        # Predict
        self.x_state = dot(self.F, self.x_state)
        self.P = 1. * dot(dot(self.F, self.P), self.F.T) + self.Q
        # self.x_state = x.astype(int)

    def predict_only_no_update(self):
        return dot(self.F, self.x_state)

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import glob
    import helpers_FRCNN as helpers

    # Create an instance
    trk = Tracker()
    # Test R_ratio
    trk.R_scaler = 1.0/16
    # Update measurement noise covariance matrix
    trk.update_R()
    # Initial state
    x_init = np.array([390, 0, 1050, 0, 513, 0, 1278, 0])
    x_init_box = [x_init[0], x_init[2], x_init[4], x_init[6]]
    # Measurement
    z=np.array([399, 1022, 504, 1256])
    trk.x_state= x_init.T
    trk.kalman_filter(z.T)
    # Updated state
    x_update =trk.x_state
    x_updated_box = [x_update[0], x_update[2], x_update[4], x_update[6]]

    print('The initial state is: ', x_init)
    print('The measurement is: ', z)
    print('The update state is: ', x_update)

    # Visualize the Kalman filter process and the
    # impact of measurement nosie convariance matrix

    images = [plt.imread(file) for file in glob.glob('./test_images/*.jpg')]
    img=images[3]

    plt.figure(figsize=(10, 14))
    helpers.draw_box_label(img, x_init_box, box_color=(0, 255, 0))
    ax = plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.title('Initial: '+str(x_init_box))

    helpers.draw_box_label(img, z, box_color=(255, 0, 0))
    ax = plt.subplot(3, 1, 2)
    plt.imshow(img)
    plt.title('Measurement: '+str(z))

    helpers.draw_box_label(img, x_updated_box)
    ax = plt.subplot(3, 1, 3)
    plt.imshow(img)
    plt.title('Updated: '+str(x_updated_box))
    plt.show()
