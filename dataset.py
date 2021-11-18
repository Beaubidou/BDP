import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split


class Dataset:

    def __init__(self, reload_all=False, interpolate=True):
        """
        :param reload_all: [bool] if True, compute the average temp. and add it to the dataframe. If False, load data
                            from data.csv which already containes average temp. computed (to avoid computing time)
        :param interpolate: [bool] if True, add and interpolate missing samples
        """

        if reload_all:
            
            self.data = None

            # Load data
            self.__raw_bathroom = pd.read_csv('Mesures/temperature_bathroom.csv', ',', header=0,
                                              names=['time', 'current_value_bathroom', 'setpoint_bathroom'])
            self.__raw_kitchen = pd.read_csv('Mesures/temperature_kitchen.csv', ',', header=0,
                                             names=['time', 'current_value_kitchen', 'setpoint_kitchen'])
            self.__raw_bedroom1 = pd.read_csv('Mesures/temperature_bedroom_1.csv', ',', header=0,
                                              names=['time', 'current_value_bedroom1', 'setpoint_bedroom1'])
            self.__raw_bedroom2 = pd.read_csv('Mesures/temperature_bedroom_2.csv', ',', header=0,
                                              names=['time', 'current_value_bedroom2', 'setpoint_bedroom2'])
            self.__raw_bedroom3 = pd.read_csv('Mesures/temperature_bedroom_3.csv', ',', header=0,
                                              names=['time', 'current_value_bedroom3', 'setpoint_bedroom3'])
            self.__raw_diningroom = pd.read_csv('Mesures/temperature_diningroom.csv', ',', header=0,
                                                names=['time', 'current_value_diningroom', 'setpoint_diningroom'])
            self.__raw_livingroom = pd.read_csv('Mesures/temperature_livingroom.csv', ',', header=0,
                                                names=['time', 'current_value_livingroom', 'setpoint_livingroom'])
            self.__raw_outside = pd.read_csv('Mesures/temperature_outside.csv', ',', header=0,
                                             names=['time', 'current_value_outside'])
            self.__raw_heating_syst = pd.read_csv('Mesures/temperature_heating_system.csv', ',', header=0,
                                             names=['time', 'water_pressure', 'water_temperature'])
            # interpolate missing values
            if interpolate:
                data_frames_rooms = [self.__raw_bathroom, self.__raw_kitchen, self.__raw_bedroom1, self.__raw_bedroom2,
                               self.__raw_bedroom3, self.__raw_diningroom, self.__raw_livingroom]
                new_data_frames_rooms = []

                for curr_data in data_frames_rooms:
                    curr_data['time'] = pd.to_datetime(curr_data.time, infer_datetime_format=True) # Convert 'time' column to datetime type
                    curr_data['time'] = curr_data['time'].dt.floor('T') # Round minute of 'time' column
                    curr_data.set_index('time', inplace=True)
                    newIndex = pd.date_range(start=curr_data.index[0],
                                            end=curr_data.index[len(curr_data.index)-1], freq='5min') # Create a 5 min step vector form the first to the last date
                    curr_data = curr_data.reindex(newIndex)
                    curr_data.iloc[:, 0] = curr_data.iloc[:, 0].interpolate(method='time')
                    curr_data.iloc[:, 1] = curr_data.iloc[:, 1].fillna(method='ffill')
                    curr_data.reset_index(inplace=True)
                    curr_data = curr_data.rename(columns={'index': 'time'})
                    new_data_frames_rooms.append(curr_data)

                data_frames_others = [self.__raw_outside, self.__raw_heating_syst]
                new_data_frames_others = []

                for curr_data in data_frames_others:
                    curr_data['time'] = pd.to_datetime(curr_data.time, infer_datetime_format=True)
                    curr_data['time'] = curr_data['time'].dt.floor('T')
                    curr_data.set_index('time', inplace=True)
                    newIndex = pd.date_range(start=curr_data.index[0],
                                            end=curr_data.index[len(curr_data.index)-1], freq='5min')
                    curr_data = curr_data.reindex(newIndex)
                    curr_data = curr_data.interpolate(method='time')
                    curr_data.reset_index(inplace=True)
                    curr_data = curr_data.rename(columns={'index': 'time'})
                    new_data_frames_others.append(curr_data)

                tmp_data_frames_rooms = reduce(lambda left, right: pd.merge(left, right, on='time'), new_data_frames_rooms)
                tmp_data_frames_others = reduce(lambda left, right: pd.merge(left, right, on='time'), new_data_frames_others)

                data_frames = [tmp_data_frames_rooms, tmp_data_frames_others]

            else:
                # Putting them in a list in order to merge them on date
                loop_data_frames = [self.__raw_bathroom, self.__raw_kitchen, self.__raw_bedroom1, self.__raw_bedroom2,
                               self.__raw_bedroom3, self.__raw_diningroom, self.__raw_livingroom, self.__raw_outside, self.__raw_heating_syst]
                data_frames = []

                for curr_data in loop_data_frames:
                    curr_data['time'] = pd.to_datetime(curr_data.time, infer_datetime_format=True)
                    curr_data['time'] = curr_data['time'].dt.floor('T')
                    data_frames.append(curr_data)

            self.data = reduce(lambda left, right: pd.merge(left, right, on='time'), data_frames)

            print("nb of samples = " + str(self.data.shape[0]))

            # Compute two column containing the average temperature and average setpoint of the house 

            # room_area = [bathroom_area, kitchen_area, bedroom1_area, bedroom2_area, bedroom3_area, diningroom_area, livingroom_area]
            room_area = [4 * 3.87, 4 * 3.87, 4 * 3.94, 4 * 4.60, 4 * 3.13, 4 * 3.94, 4 * 7.81]
            current_mean_value = []
            current_mean_setpoint = []

            for i in range(self.data.shape[0]):
                current_mean_value.append(round( (room_area[0]*self.data.loc[i].at["current_value_bathroom"] + \
                    room_area[1]*self.data.loc[i].at["current_value_kitchen"] + room_area[2]*self.data.loc[i].at["current_value_bedroom1"] + \
                    room_area[3]*self.data.loc[i].at["current_value_bedroom2"] + room_area[4]*self.data.loc[i].at["current_value_bedroom3"] + \
                    room_area[5]*self.data.loc[i].at["current_value_diningroom"] + \
                    room_area[6]*self.data.loc[i].at["current_value_livingroom"]) / sum(room_area) ,2) )

                current_mean_setpoint.append(round( (room_area[0]*self.data.loc[i].at["setpoint_bathroom"] + \
                    room_area[1]*self.data.loc[i].at["setpoint_kitchen"] + room_area[2]*self.data.loc[i].at["setpoint_bedroom1"] + \
                    room_area[3]*self.data.loc[i].at["setpoint_bedroom2"] + room_area[4]*self.data.loc[i].at["setpoint_bedroom3"] + \
                    room_area[5]*self.data.loc[i].at["setpoint_diningroom"] + \
                    room_area[6]*self.data.loc[i].at["setpoint_livingroom"]) / sum(room_area) ,2) )


            self.data.insert(1, 'setpoint_house', current_mean_setpoint)
            self.data.insert(1, 'current_value_house', current_mean_value)

            # Write data in csv file
            
            if interpolate:
                self.data.to_csv('Mesures/data_interpolated.csv', index=False, header=True)
            else: 
                self.data.to_csv('Mesures/data.csv', index=False, header=True)

        else : 

            # Read data from csv file in the 'Mesures' folder

            if interpolate:
                try:
                    self.data = pd.read_csv('Mesures/data_interpolated.csv', sep=',')
                except FileNotFoundError:
                    print("File not found. The file is probably not loaded, try first to load your file in your 'Mesures' folder with the following parametres : data = Dataset(reload_all=True, interpolate=True)")
            else:
                try:
                    self.data = pd.read_csv('Mesures/data.csv', sep=',')
                except FileNotFoundError:
                    print("File not found. The file is probably not loaded, try first to load your file in your 'Mesures' folder with the following parametres : data = Dataset(reload_all=True, interpolate=False)")


            self.data['time'] = pd.to_datetime(self.data.time, infer_datetime_format=True).astype('datetime64[ns]')

        #Have an array of time starting at 0 for instant 0 and going to the end of the time set
        self.data['timeIntervals'] = np.linspace(0, 5 * (len(self.data)-1), num=len(self.data))
        self.data.set_index('timeIntervals', inplace=True)

        # Create functions for inputs in the model simulation
        self.function_T_out = interp1d(self.data.index, self.data['current_value_outside'])
        self.function_T_set_house = interp1d(self.data.index, self.data['setpoint_house'])

        #Add more for the multizone
        self.function_T_set_kitchen = interp1d(self.data.index, self.data['setpoint_kitchen'])
        self.function_T_set_diningroom = interp1d(self.data.index, self.data['setpoint_diningroom'])
        self.function_T_set_livingroom = interp1d(self.data.index, self.data['setpoint_livingroom'])
        self.function_T_set_bathroom = interp1d(self.data.index, self.data['setpoint_bathroom'])
        self.function_T_set_bedroom1 = interp1d(self.data.index, self.data['setpoint_bedroom1'])
        self.function_T_set_bedroom2 = interp1d(self.data.index, self.data['setpoint_bedroom2'])
        self.function_T_set_bedroom3 = interp1d(self.data.index, self.data['setpoint_bedroom3'])



    def getInputs(self, multizone=False):
        if multizone:
            return self.function_T_out, self.function_T_set_kitchen, self.function_T_set_diningroom, \
                self.function_T_set_livingroom, self.function_T_set_bathroom, self.function_T_set_bedroom1, \
                self.function_T_set_bedroom2, self.function_T_set_bedroom3

        else:
            return self.function_T_out, self.function_T_set_house
    def getTemperaturesForInput(self):
        #Get the tempratures in the rooms as function to train separately the rooms
        Tkitchen = interp1d(self.data.index, self.data["current_value_kitchen"])
        Tdinigroom = interp1d(self.data.index, self.data["current_value_diningroom"])
        Tlivingroom = interp1d(self.data.index, self.data["current_value_livingroom"])
        Tbathroom = interp1d(self.data.index, self.data["current_value_bathroom"])
        Tbed1 = interp1d(self.data.index, self.data["current_value_bedroom1"])
        Tbed2 = interp1d(self.data.index, self.data["current_value_bedroom2"])
        Tbed3 = interp1d(self.data.index, self.data["current_value_bedroom2"])

        return [Tkitchen, Tdinigroom, Tlivingroom, Tbathroom, Tbed1, Tbed2, Tbed3]

    def train_test_sample_split(self, start_date="2020-05-24 19:40:00", length=20, test_ratio=0.5, multi_z=False, shuffle=False):
        """
        Separate the provied dataset into a train set and a test set
        :param start_date: [str ("YYYY-MM-DD")] 
        :param end_date: [str ("YYYY-MM-DD")] 
        :param test_ratio: [float < 1] the test/data set ratio 
        :param multi_z: [bool] True if multi-zone, False if single-zone
        :return: X_train, X_test, y_train, y_test
        """

        i_s = self.data.index[self.data['time'] == start_date]
        i_e = i_s + 5 * length

        ds = self.data[i_s[0] : i_e[0]]

        if multi_z: #Put all the column we want for the multizone and push it on gitHub
            y = ds[['current_value_kitchen', 'current_value_diningroom', 'current_value_livingroom', 'current_value_bathroom', 'current_value_bedroom1', 'current_value_bedroom2', 'current_value_bedroom3']]
            X = ds['time']

        else:
            y = ds[['current_value_house']] # True value of T_in
            X = ds['time'] # Inputs are hadeled by functions -> just return time where happend the sets for plot

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=shuffle)
        

        return X_train, X_test, y_train, y_test
    
    def plotDataset(self): 
        plt.figure()
        try:
            plt.title("")
            plt.xlabel('Time')
            plt.ylabel('Temperature (°c)')

            plt.plot(self.data['time'], self.data['current_value_outside'], label='T_out', linewidth=0.5)
            plt.plot(self.data['time'], self.data['current_value_house'], label='T_house', linewidth=0.5)
            plt.plot(self.data['time'], self.data['setpoint_house'], label='Set_house', linewidth=0.5)
            #plt.plot(self.data['time'], self.data['water_temperature'], label='T_Heater', linewidth=0.5)

            plt.legend()
            #plt.xlim(0, 2)

            fname = "Info_data_SZ"
            plt.savefig("plots/{}.png".format(fname))

        finally:
            plt.close()
    
    def drawBoxplot(self):
        plt.figure()
        try:
            # draw boxplot of each column of the data -> to compare it with our predictions later
            self.data.boxplot(column=['current_value_livingroom', 'current_value_outside'])
            # return axessubplot object
            fname = "Boxplots_T_living&T_outside"
            plt.savefig("plots/{}.png".format(fname))
        finally:
            plt.close()

# test
if __name__ == '__main__':

    # First time need to load all the data -> it takes some time  but after that just take the second line and it's quick
    #dataset = Dataset(reload_all=True, interpolate=True)
    dataset = Dataset(reload_all=False, interpolate=True)


    #X_train, X_test, y_train, y_test = dataset.train_test_sample_split(start_date="2020-05-24 19:40:00", length=50)

    #print(np.array(y_train.head(1)['current_value_house']))

    #print(X_train)
    #print(X_test)

    #dataset.plotDataset()
    #dataset.drawBoxplot()

    pd.set_option("display.max.columns", None)
    pd.set_option("display.max.rows", None)
    rooms = dataset.data[['current_value_kitchen', 'current_value_diningroom', 'current_value_livingroom', 'current_value_bathroom', 'current_value_bedroom1', 'current_value_bedroom2', 'current_value_bedroom3']]
    corr = rooms.corr()

    print(corr)
    
    i_s = dataset.data.index[dataset.data['time'] == "2020-11-01 00:00:00"]
    i_e = dataset.data.index[dataset.data['time'] == "2021-04-01 00:00:00"]

    ds = dataset.data[i_s[0] : i_e[0]]

    ds_living = ds[['current_value_outside','current_value_livingroom','setpoint_livingroom']]
    print(ds_living.corr())

    ds_living = ds[['current_value_outside','current_value_bedroom3','setpoint_bedroom3']]
    print(ds_living.corr())
