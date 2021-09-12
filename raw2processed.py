#---------------------------IMPORT LIBRARIES-----------------------------------
import pandas as pd
import numpy as np


#-------------------------READ RAW DATA----------------------------------------
df = pd.read_excel("Raw_data.xlsx")

#------------------------DATA PROCESSING ---------------------------------------

# Extract IMU2 data from the Raw data
dx = df[df.timestamp.str.contains('IMU2')]
dx.to_excel("processed_data.xlsx")
dx1 = pd.read_excel("processed_data.xlsx", parse_dates=["timestamp"])

# Rename the columns
dx1.rename(columns={1:'GyrX'}, inplace=True)
dx1.rename(columns={2:'GyrY'}, inplace=True)
dx1.rename(columns={3:'GyrZ'}, inplace=True)
dx1.rename(columns={4:'AccX'}, inplace=True)
dx1.rename(columns={5:'AccY'}, inplace=True)
dx1.rename(columns={6:'AccZ'}, inplace=True)

# Drop unnecessary columns
dx1.drop(columns=[ 'Unnamed: 0',
                   7,      'value8',      'value9',     'value10',
       'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14',
       'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18',
       'Unnamed: 19'], inplace=True)

# Extract the magnitude of the Accelerometer and Gyro data from respective columns
dx1['GyrX']=dx1['GyrX'].str[7:]
dx1['GyrY']=dx1['GyrY'].str[7:]
dx1['GyrZ']=dx1['GyrZ'].str[7:]
dx1['AccX']=dx1['AccX'].str[7:]
dx1['AccY']=dx1['AccY'].str[7:]
dx1['AccZ']=dx1['AccZ'].str[7:]


# Extract AHRS2 Data from Raw data
dy=df[df.timestamp.str.contains('AHR2')]
dy.to_excel("processed_data1.xlsx")
dy1 = pd.read_excel("processed_data1.xlsx", parse_dates=["timestamp"])

# Rename the columns
dy1.rename(columns={1:'Roll'}, inplace=True)
dy1.rename(columns={2:'Pitch'}, inplace=True)
dy1.rename(columns={3:'Yaw'}, inplace=True)
dy1.rename(columns={4:'Altitude'}, inplace=True)
dy1.rename(columns={5:'Latitude'}, inplace=True)
dy1.rename(columns={6:'Longitude'}, inplace=True)
dy1.rename(columns={7:'Q1'}, inplace=True)
dy1.rename(columns={'value8':'Q2'}, inplace=True)
dy1.rename(columns={'value9':'Q3'}, inplace=True)
dy1.rename(columns={'value10':'Q4'}, inplace=True)

# Drop the unnecssary columns
dy1.drop(columns=['Unnamed: 0',  'Unnamed: 11',
       'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15',
       'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19' ], inplace=True)

# Extract the magnitude of the coordinates and angles(roll, pitch & yaw) from respective columns
dy1['Roll']=dy1['Roll'].str[7:]
dy1['Pitch']=dy1['Pitch'].str[8:]
dy1['Yaw']=dy1['Yaw'].str[6:]
dy1['Altitude']=dy1['Altitude'].str[7:]
dy1['Latitude']=dy1['Latitude'].str[7:]
dy1['Longitude']=dy1['Longitude'].str[7:]

# Quaternions
dy1['Q1']=dy1['Q1'].str[5:]
dy1['Q2']=dy1['Q2'].str[5:]
dy1['Q3']=dy1['Q3'].str[5:]
dy1['Q4']=dy1['Q4'].str[5:]


# Process the timestamp to extract time arrival of each data point
dx1['timestamp']=dx1['timestamp'].str[:22]
dy1['timestamp']=dy1['timestamp'].str[:22]


# Indexing for IMU2 data
index = pd.DatetimeIndex(dx1['timestamp'])
d=index.astype(np.int64)
qe = pd.DataFrame(d)
qe.rename(columns={'timestamp':'timediff'}, inplace=True)
resultx = pd.concat([qe, dx1], axis=1, join='inner')
resultx.drop(columns=['timestamp'], inplace=True)
resultx['timediff'] = resultx['timediff'] - resultx['timediff'].shift(+1)

# Indexing for AHRS data
index1 = pd.DatetimeIndex(dy1['timestamp'])
e=index1.astype(np.int64)/10**9
pe = pd.DataFrame(e)
pe.rename(columns={'timestamp':'timediff'}, inplace=True)
resulty = pd.concat([pe, dy1], axis=1, join='inner')
resulty.drop(columns=['timestamp'], inplace=True)
resulty['timediff'] = resulty['timediff'] - resulty['timediff'].shift(+1)



#----------------------------SAVE THE IMU DATA--------------------------------
resultx.to_excel(r'C:\Users\ASUS\OneDrive\Desktop\processed_data(IMU).xlsx')




#--------------------------SAVE THE AHRS DATA----------------------------------
resulty.to_excel(r'C:\Users\ASUS\OneDrive\Desktop\processed_data(AHRS).xlsx')
