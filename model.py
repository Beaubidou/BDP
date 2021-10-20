from scipy.optimize import minimize
from scipy.integrate import odeint
import numpy as np
from dataset import Dataset


def getTemperature(t, d):
    """
    :param t: [int] time where to pick the temperature
    :param d:
    :return:
    """
    data = d['current_value_outside']
    return data[int(t)]


class Model:
    #Variables defining the model
    resistors = []
    capacitors = []
    parameters = []

    def __init__(self, comp):
        # Values to initialize the parameters -------------> choose better
        initR = 1
        initC = 1
        self.Q = 1

        self.complexity = comp
        self.data = Dataset().data

        # Loop defining the parameters
        for i in range(self.complexity):
            self.resistors.append(initR)
            self.capacitors.append(initC)

        tmp = self.resistors + self.capacitors
        tmp.append(self.Q)

        self.parameters = tuple(tmp)

        # For the representation of our results 
        self.objectives = []

    def equ_diff_1zone(self, y, t, R, C, q_para):
        Tz = y[0]

        dTz = (q_para - (Tz - getTemperature(t, self.data)) / R) / C

        return dTz

    def equations(self, currentT, time, *argv):
        """
        :param currentT: [T1 T2 ... Tn]
        :param time: current time
        :param data: outside temperatures
        :return: [dT1 dT2 ... dTn]
        """
        # Takes a number of parameters dependent to the complexity
        params = []
        for arg in argv:
            params.append(arg)
        print(params)
        
        dT = []

        # Regroup the different states in the model
        currentT = [getTemperature(time, self.data)] + list(currentT)
        c = self.complexity
        print("enter equation")
        print (currentT)

        # Compute the differential equations
        for i in range(c - 1):
            j = i + 1
            dT.append((currentT[j-1]-currentT[j])/(params[j-1]*params[c+j-1])+(currentT[j+1]-currentT[j])
                      / (params[j]*params[c+j-1]))

        dT.append((params[-1] + (currentT[-2] - currentT[-1]) / params[c-1]) / params[-2])
        return tuple(dT)

    def simulate(self, time, init, params):

        predict = odeint(func=self.equations, y0=init, t=time, args=tuple(params))
        return predict

    def obj(self, params, t, data):  # add more if other obj function
        trueValues = np.array(data)

        # Find initial intermidiate temperatures
        Tout = getTemperature(t[0], self.data)
        initT = list(np.linspace(Tout, trueValues[0], self.complexity+1))
        initT.pop(0)

        # Simulate the internal temperature with the model
        simulatedValues = self.simulate(t, initT, params)
        if self.complexity != 1:
            simulatedValues = simulatedValues[:,1]

        # Compute RMSE
        s = np.sum((np.array(simulatedValues) - trueValues) ** 2)
        self.objectives.append(s)
        return s

    def fit(self, time, data):

        result = minimize(self.obj, self.parameters, args=(time,data), method='Powell')
        return result
