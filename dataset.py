import pandas as pd
from functools import reduce


class Dataset:

    def __init__(self, reload_all=False):
        """
        :param reload_all: [bool] if True, compute the average temp. and add it to the dataframe. If False, load data
                            from data.csv which already containes average temp. computed (to avoid computing time)
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

            # Putting them in a list in order to merge them on date
            data_frames = [self.__raw_bathroom, self.__raw_kitchen, self.__raw_bedroom1, self.__raw_bedroom2,
                           self.__raw_bedroom3, self.__raw_diningroom, self.__raw_livingroom, 
                            self.__raw_outside, self.__raw_heating_syst]

            self.data = reduce(lambda left, right: pd.merge(left, right, on='time'), data_frames)

            # room_area = [bathroom_area, kitchen_area, bedroom1_area, bedroom2_area, bedroom3_area, diningroom_area, livingroom_area]
            room_area = [4 * 3.87, 4 * 3.87, 4 * 3.94, 4 * 4.60, 4 * 3.13, 4 * 3.94, 4 * 7.81]
            current_mean_value = []

            for i in range(self.data.shape[0]):
                current_mean_value.append(round( (room_area[0]*self.data.loc[i].at["current_value_bathroom"] + \
                    room_area[1]*self.data.loc[i].at["current_value_kitchen"] + room_area[2]*self.data.loc[i].at["current_value_bedroom1"] + \
                    room_area[3]*self.data.loc[i].at["current_value_bedroom2"] + room_area[4]*self.data.loc[i].at["current_value_bedroom3"] + \
                    room_area[5]*self.data.loc[i].at["current_value_diningroom"] + \
                    room_area[6]*self.data.loc[i].at["current_value_livingroom"]) / sum(room_area) ,2) )

            self.data.insert(1, 'current_value_house', current_mean_value)
            

            self.data.to_csv('Mesures/data.csv', index=False, header=True)

        else : 

            self.data = pd.read_csv('Mesures/data.csv', sep=',')
