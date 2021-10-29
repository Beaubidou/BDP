import pandas as pd
from functools import reduce


class Dataset:

    def __init__(self, reload_all=False, interpolate=False):
        """
        :param reload_all: [bool] if True, compute the average temp. and add it to the dataframe. If False, load data
                            from data.csv which already containes average temp. computed (to avoid computing time)
        :param interpolate: [bool] if True (and reload_all=True), add and interpolate missing samples
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
                    curr_data['time'] = pd.to_datetime(curr_data.time, infer_datetime_format=True)
                    curr_data['time'] = curr_data['time'].dt.floor('T')
                    curr_data.set_index('time', inplace=True)
                    newIndex = pd.date_range(start=curr_data.index[0],
                                            end=curr_data.index[len(curr_data.index)-1], freq='5min')
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
                data_frames = [self.__raw_bathroom, self.__raw_kitchen, self.__raw_bedroom1, self.__raw_bedroom2,
                               self.__raw_bedroom3, self.__raw_diningroom, self.__raw_livingroom, self.__raw_outside, self.__raw_heating_syst]

            self.data = reduce(lambda left, right: pd.merge(left, right, on='time'), data_frames)

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
            self.data['time'] = pd.to_datetime(self.data.time, infer_datetime_format=True)
            self.data['time'] = self.data['time'].dt.floor('T')
            
            self.data.to_csv('Mesures/data.csv', index=False, header=True)

        else : 

            self.data = pd.read_csv('Mesures/data.csv', sep=',')
            self.data['time'] = pd.to_datetime(self.data.time, infer_datetime_format=True)


# test
if __name__ == '__main__':

    data = Dataset(reload_all=True, interpolate=True).data.head(100)
    print(data)
