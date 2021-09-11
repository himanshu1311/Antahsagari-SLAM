
import numpy as np
import pandas as pd
from numpy.linalg import inv
import math as m
import matplotlib.pyplot as plt


# kalman filter to compute current state of ROV based on IMU data from pixhawk
class kalman:

    def __init__(self, initial_X):
        self.U_t = None

        self.F_t = None

        self.B_t = None

        self.Q_t = None

        self.X_t = initial_X.T

        self.H = np.matrix([[1, 0, 0, 0, 0, 0],  # Set zero for dead-reckoning
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])

        self.P = np.matrix([[1, 0, 0, 0, 0, 0],
                            [0, 1000, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1000, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1000]])

        self.R = 2.04*np.identity(6)

        self.K_gain = 0

    def compute_matrix(self, dt):  # updates the transition matrix F, control matrix B and process covariance matrix

        sigma_v2 = 9
        self.F_t = np.matrix([[1, dt, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 1, dt, 0, 0],
                              [0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, dt],
                              [0, 0, 0, 0, 0, 1]])

        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4

        self.B_t = np.matrix([[dt2 / 2, 0, 0],
                              [dt, 0, 0],
                              [0, dt2 / 2, 0],
                              [0, dt, 0],
                              [0, 0, dt2 / 2],
                              [0, 0, dt]])

        self.Q_t = np.matrix([[dt4 / 4, dt3 / 2, 0, 0, 0, 0],
                              [dt3 / 2, dt2, 0, 0, 0, 0],
                              [0, 0, dt4 / 4, dt3 / 2, 0, 0],
                              [0, 0, dt3 / 2, dt2, 0, 0],
                              [0, 0, 0, 0, dt4 / 4, dt3 / 2],
                              [0, 0, 0, 0, dt3 / 2, dt2]])

        self.Q_t = sigma_v2 * self.Q_t

    def predicted_state(self, dt, a_x, a_y, a_z):  # Predicts the state vetor and error covariance matrix

        self.U_t = np.matrix([[a_x], [a_y], [a_z]])

        self.X_t = (self.F_t * self.X_t) + (self.B_t * self.U_t)

        self.P = (self.F_t * self.P * self.F_t.T) + self.Q_t

    def update(self, depth_y, x, y):  # Updates the state vector based on measurement
        self.ip = np.matrix([x, 0, y, 0, depth_y, 0]).T
        V = self.ip - (self.H * self.X_t)

        S = (self.H * self.P * self.H.T) + self.R
        self.K_gain = (self.P * self.H.T) * np.linalg.inv(S)

        self.X_t = self.X_t + self.K_gain * V

        I = np.identity(6)

        self.P = (1 - self.K_gain * self.H) * self.P

    def update1(self, depth_y, x, y):  # Updates the state vector based on measurement
        self.ip = np.matrix([x, 0, y, 0, depth_y, 0]).T
        V = self.ip - (self.H * self.X_t)
        S = (self.H * self.P * self.H.T) + self.R
        I = np.identity(6)

    def current_state(self):  # Returns the current state vector

        return (self.X_t)

    def current_covariance(self):
        return (self.P)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# input data from data file
def read_data(file_name):
    values = pd.read_excel(io=file_name)
    time = values['Timesample'].to_numpy()
    roll = values['Roll'].to_numpy()
    pitch = values['Pitch'].to_numpy()
    yaw = values['Yaw'].to_numpy()
    accel_x = values['Acceleration_x'].to_numpy()
    accel_y = values['Acceleration_y'].to_numpy()
    accel_z = values['Acceleration_z'].to_numpy()
    depth = values['Altitude'].to_numpy()
    latitude = values['Latitude'].to_numpy()
    longitude = values['Longitude'].to_numpy()
    return time, accel_x, accel_y, accel_z, roll, pitch, yaw, depth, latitude, longitude
    print(accel_x)


[time, accel_x, accel_y, accel_z, roll, pitch, yaw, depth, latitude, longitude] = read_data(<INPUT FILE>)
    
arr_len = len(accel_x)
ground_len = len(latitude)
# acceleration in north,east and downward directions
Accel_N = -accel_x
Accel_E = accel_y
Accel_D = accel_z

# Timeline
timeline = 0



# Compute ground truth values in cartesian coordinate--------------------------------------------------------------------------------------------------------------------------------------------------------------------
earth_rad = 6371 * 1e3
x_g = np.zeros(arr_len, dtype=float)
y_g = np.zeros(arr_len, dtype=float)
prev_x = 0
prev_y = 0
distance = np.zeros(arr_len - 1, dtype=float)
bearing = np.zeros(arr_len - 1, dtype=float)
latitude = m.pi / 180 * latitude
longitude = m.pi / 180 * longitude

# compute distance and bearing between adjacent points
for i in range(1, arr_len):
    dlat = latitude[i] - latitude[i - 1]
    dlong = longitude[i] - longitude[i - 1]
    a = (m.sin(dlat / 2)) ** 2 + m.cos(latitude[i]) * m.cos(latitude[i - 1]) * (m.sin(dlong / 2)) ** 2
    c = 2 * m.atan2(m.sqrt(a), m.sqrt(1 - a))
    distance[i - 1] = earth_rad * c
    bearing[i - 1] = m.atan2(m.sin(dlong) * m.cos(latitude[i]),
                             m.cos(latitude[i - 1]) * m.sin(latitude[i]) - m.sin(latitude[i - 1]) * m.cos(
                                 latitude[i]) * m.cos(dlong))

# compute cartesian coordinates
for j in range(1, arr_len):
    x_g[j] = distance[j - 1] * m.cos(bearing[j - 1]) + prev_x
    y_g[j] = distance[j - 1] * m.sin(bearing[j - 1]) + prev_y
    prev_x = x_g[j]
    prev_y = y_g[j]

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Estimated state of ROV in terms of x, y and z coordinates
current_mat = np.zeros((arr_len, 6), dtype=float)
countx= 0
county= 0
countz= 0
for j in range(1, arr_len):
    kalob = kalman(current_mat[[j - 1], :])  # Initalising next object of class kalman with previous state values
    kalob.compute_matrix(time[j])
    kalob.predicted_state(time[j], Accel_N[j], Accel_E[j], Accel_D[j])

    timeline = timeline + 0.02
    if timeline*100 % 20 == 0:
        kalob.update(depth[countz], x_g[countx], y_g[county])
        countx +=1
        county += 1
        countz += 1

    current_mat[[j], :] = (kalob.current_state()).T

X_estimated = current_mat[:, 0]
Y_estimated = current_mat[:, 2]
Z_estimated = current_mat[:, 4]

mat = np.matrix([X_estimated, Y_estimated]).T
string_ind = ['X_estimate', 'Y_estimate']
ext_val = pd.DataFrame(mat, columns=string_ind)
writer = pd.ExcelWriter(r'C:\Users\Dell\Desktop\processed_dataf2.xlsx')
ext_val.to_excel(writer, sheet_name='output.xlsx')
writer.save()

plt.plot(X_estimated, Y_estimated)
plt.show()


# In[ ]:


# In[ ]:




