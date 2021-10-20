import pandas as pd
from functools import reduce


class Dataset:

    def __init__(self):

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

        # Putting them in a list in order to merge them on date
        data_frames = [self.__raw_bathroom, self.__raw_kitchen, self.__raw_bedroom1, self.__raw_bedroom2,
                       self.__raw_bedroom3, self.__raw_diningroom, self.__raw_livingroom, self.__raw_outside]

        self.data = reduce(lambda left, right: pd.merge(left, right, on='time'), data_frames)
