from dataset import Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns


from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize



class Model:
    objectives = []
    parametersTried = []

    timeObj = 0
    timeOdeint = 0

    def __init__(self):

        ###################### Elements to test ####################
        # Set the compexity of the model (1R1C, 2R2C, ...)
        self.complexity = 1

        # Set the initial parameters [R1 C1 Q]
        self.parameters = (0.1, 10000, 5) #1R1C

        # Set an optimization method
        self.solver = 'L-BFGS-B' #'powell'

        #Length of the set taken to train and test (33% for testing)
        length = 10000

        ######################## Create dataset #######################
        # Take the data relative to the house to model
        dataset = Dataset(reload_all=False, interpolate=True)

        #Return padas.Dataframes for true temperature inside the house
        self.dates_train, self.dates_test, self.y_train, self.y_test = dataset.train_test_sample_split(start_date="2020-05-24 19:40:00", length=length)
        self.time_train = np.array(self.y_train.index)
        self.init_train = np.array(self.y_train.head(1)['current_value_house'])
        self.time_test = np.array(self.y_test.index)
        self.init_test = np.array(self.y_test.head(1)['current_value_house'])

        self.function_T_out, self.function_T_set_house = dataset.getInputs()
        
    def equations(self, currentT, t_i, *argv):
        """
        currentT = [T_in] 1R1C - [T_wall T_in] 2R2C
        """
        # Takes a number of parameters dependent to the complexity
        params = [] #[R1 ... Rn C1 ... Cn Qp]
        for arg in argv:
            params.append(arg)

        # Get inputs at time t
        T_out = self.function_T_out(t_i)
        T_heater = self.function_T_set_house(t_i)

        dT = []

        if self.complexity == 1:
            # dT_in = (T_out-T_in)/RC + Q/C
            dT.append((T_out - currentT[0])/(params[0] *params[1]))
            

        elif self.complexity == 2:
            dT.append(((T_out - currentT[0])/params[0] + (currentT[1]-currentT[0])/params[1])/params[3])
            dT.append(((currentT[0] - currentT[1])/params[1] + (T_heater-currentT[1])/params[5])/params[3])
        else:
            print("too complex for me")


        return dT

    def simulate(self, params=None, training=True):
        # Take the current parameters when training and the parameters of the model when testing
        if params is None:
            params = self.parameters
        
        # Take the initial condition and the time array depending to the set we use
        if training:
            init = self.init_train
            timeArray = self.time_train
        else:
            init = self.init_test
            timeArray = self.time_test
        first = time.time()
        # Compute the evolution of T_in in a set of time
        predict = odeint(func=self.equations, y0=init, t=timeArray, args=tuple(params))
        second = time.time()
        self.timeOdeint += second - first
        print("Computing time of odeint : " + str(self.timeOdeint))
            
        return predict
    
    def obj(self, params, training=True, plot=False):
        if params is None:
            params = self.parameters
        # Get the true values of the temperature inside during the time set of the LS
        if training:
            trueValues = np.array(self.y_train)
            simulatedValues = self.simulate(params=params)
            datesArray = self.dates_train

        else:
            trueValues = np.array(self.y_test)
            simulatedValues = self.simulate(training=False)
            datesArray = self.dates_test
        
        first = time.time()

        # Compute RMSE
        s = mean_squared_error(trueValues, np.array(simulatedValues), squared=False)
        
        # Keep values to show how it evolves
        self.objectives.append(s)
        self.parametersTried.append(params)
        print(params)
        print(s)
        print()

        #Print the distribution of the residuals to see if normal
        residuals = np.array(trueValues - simulatedValues)
        res = sns.displot(residuals)
        print(type(res))
        #plt.show()

        """ plt.figure()
        try:
            plt.title("")
            plt.xlabel('')
            plt.ylabel('')

            res = sns.displot(residuals)
            print(type(res))
            plt.show()
            

            fname = "residualsDist"
            plt.savefig("{}.png".format(fname))

        finally:
            plt.close() """



        if plot:
            if training:
                name = "train"
            else:
                name= "test"

            plt.figure()
            try:
                plt.title("")
                plt.xlabel('Time')
                plt.ylabel('Temperature (°c)')

                plt.plot(datesArray, simulatedValues, label='Simulated values')
                plt.plot(datesArray, trueValues, label='True values')
                
                plt.legend()

                fname = "plots/Model_simulation_"+name
                plt.savefig("{}.png".format(fname))

            finally:
                plt.close()
        second = time.time()
        self.timeObj += (second - first)
        print("computing time of objective : " + str(self.timeObj))
        return s   

    def fit(self):
        # Optimization of the parameters
        result = minimize(self.obj, self.parameters, method=self.solver)#, bounds=self.bounds) # L-BFGS-B powell
        
        # Put the trained parameters in the model
        self.parameters = result.x

        return result  

if __name__ == '__main__':
    pd.set_option("display.max.columns", None)

    #print(np.dtype('datetime64[ns]') == np.dtype('<M8[ns]')) #True
    #print(np.dtype('datetime64[ns]') == np.dtype('<M8[ns]'))
    
    model = Model()

    #Train the model
    #res = model.fit()
    #print(res)

    #Plot the final result on the trainig set
    objTrain = model.obj(None, training=True, plot=True)

    #Test the model with parameters in the model & plot it
    objTest = model.obj(None, training=False, plot=True)
    print(objTest)