
import pandas as pd
from dataset import Dataset
from model import Model
import numpy as np


if __name__ == '__main__':

    pd.set_option("display.max.columns", None)

    # Get the data
    d = Dataset().data

    # Take a part of the dataset
    start = 2000
    number_of_datapoint = 20
    TestF = d.loc[start:start + number_of_datapoint]

    # Create the model with complexity 1 (1R1C)
    model = Model(2)
    
    # Create time to simulate
    times = TestF.index[0]
    timef = TestF.index[-1]

    time = np.linspace(times, timef, (timef - times) + 1)

    res = model.fit(time, TestF['current_value_livingroom'])
    
    print("model fitted")
    q_para = res.x[0]
    r = res.x[1]
    c = res.x[2]
    params = q_para, r, c
    state_init = TestF['current_value_livingroom'][start]
    #fitted = g(time, inistate, params)

    print(params)
    #print(fitted)
    #print
    print(TestF['current_value_livingroom'])
