from scipy.optimize import minimize
from scipy.integrate import odeint
import numpy as np
from dataset import Dataset


def getOutsideTemperature(t, d):
    """
    Get the outside temperature
    :param t: [int] time where to pick the temperature
    :param d:
    :return:
    """
    data = d['current_value_outside']
    return data[int(t)]


class Model:
    # Variables defining the model
    resistors = []
    capacitors = []
    parameters = []

    def __init__(self, comp):
        # Values to initialize the parameters -------------> choose better
        r_init = 1
        c_init = 1
        self.q = 1

        self.complexity = comp
        self.data = Dataset().data

        # Loop defining the parameters
        for i in range(self.complexity):
            self.resistors.append(r_init)
            self.capacitors.append(c_init)

        tmp = self.resistors + self.capacitors
        tmp.append(self.q)

        self.parameters = tuple(tmp)

        # For the representation of our results 
        self.objectives = []

    def equaDiffSingleZone(self, y, t, r, c, q_para):
        temp_zone = y[0]

        d_temp_zone = (q_para - (temp_zone - getOutsideTemperature(t, self.data)) / r) / c

        return d_temp_zone

    def equations(self, temp_current, time, *argv):
        """
        :param temp_current: [T1 T2 ... Tn]
        :param time: current time
        :return: [dT1 dT2 ... dTn]
        """
        # Takes a number of parameters dependent to the complexity
        params = []
        for arg in argv:
            params.append(arg)
        print(params)
        
        d_temp = []

        # Regroup the different states in the model
        temp_current = [getOutsideTemperature(time, self.data)] + list(temp_current)
        c = self.complexity
        #print("enter equation")
        #print(temp_current)

        # Compute the differential equations
        for i in range(c - 1):
            j = i + 1
            d_temp.append((temp_current[j - 1] - temp_current[j]) / (params[j - 1] * params[c + j - 1])
                          + (temp_current[j + 1] - temp_current[j])
                          / (params[j]*params[c+j-1]))

        d_temp.append((params[-1] + (temp_current[-2] - temp_current[-1]) / params[c - 1]) / params[-2])
        return tuple(d_temp)

    def simulate(self, time, init, params):

        predict = odeint(func=self.equations, y0=init, t=time, args=tuple(params))
        return predict

    def obj(self, params, t, data):  # add more if other obj function
        true_values = np.array(data)

        # Find initial intermediate temperatures
        temp_out = getOutsideTemperature(t[0], self.data)
        temp_init = list(np.linspace(temp_out, true_values[0], self.complexity + 1))
        temp_init.pop(0)

        # Simulate the internal temperature with the model
        simulated_values = self.simulate(t, temp_init, params)
        if self.complexity != 1:
            simulated_values = simulated_values[:, 1]

        # Compute RMSE
        s = np.sum((np.array(simulated_values) - true_values) ** 2)
        self.objectives.append(s)
        return s

    def fit(self, time, data):

        result = minimize(self.obj, self.parameters, args=(time, data), method='Powell')
        return result
