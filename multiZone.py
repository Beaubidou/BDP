import matplotlib.pyplot as plt
import time

from scipy.optimize import minimize
from scipy.integrate import odeint
import numpy as np
from sklearn.metrics import mean_squared_error

from dataset import Dataset

class Model:

    # Sets to plot the evolution of the objective function and the parameters tried by the model
    objectives = []
    parametersTried = []

    # Track the compilation time of odeint function and the rest of the opperations
    timeObj = 0
    timeOdeint = 0

    # Set of rooms of the house in the order considered here
    rooms = ['kitchen', 'diningroom', 'livingroom', 'bathroom', 'bedroom1', 'bedroom2', 'bedroom3']

    def __init__(self):

        # Set the initial parameters -> function init to have a beter initial guess
        #self.parameters = self.initParams()
        #[q1 q2 q3 q4 q5 q6 q7 Rw1 Rw2 Rw3 Rw4 Rw5 Rw6 Rw7 C1 C2 C3 C4 C5 C6 C7 R12 R123 R47 R45 R56 R67]
        tmp = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        self.parameters = tmp
        
        # Set an optimization method
        self.solver = 'L-BFGS-B' #'powell' #'L-BFGS-B' #'SLSQP' 

        # Length of the set taken to train and test (33% for testing)
        length = 100

        ######################## Create dataset #######################
        # Take the data relative to the house to model
        dataset = Dataset(reload_all=False, interpolate=True)

        # y_train and y_test return padas.Dataframes for true temperature inside the house
        # The index is ['kitchen', 'diningroom', 'livingroom', 'bathroom', 'bedroom1', 'bedroom2', 'bedroom3']
        # dates_train and dates_test is juste a time vector of the observations for x axis in plots
        self.dates_train, self.dates_test, self.y_train, self.y_test = dataset.train_test_sample_split(start_date="2020-07-03 10:40:00", length=length, multi_z=True)
        
        # Get the vector time for odeint and the initial conditions for the differential equations (in the order of vector rooms)
        self.time_train = np.array(self.y_train.index)
        self.init_train = np.array(self.y_train.head(1)).flatten()
        
        self.time_test = np.array(self.y_test.index)
        self.init_test = np.array(self.y_test.head(1)).flatten()

        # Get all the inputs needed in the differential equations (functions that take as argument 
        # a time t and give the cooresponding temprature)
        self.function_T_out, self.function_T_set_kitchen, self.function_T_set_diningroom, \
                self.function_T_set_livingroom, self.function_T_set_bathroom, self.function_T_set_bedroom1, \
                self.function_T_set_bedroom2, self.function_T_set_bedroom3 = dataset.getInputs(multizone=True)

    def initParams(self):
        #Fonction de Bissot à remettre
        print("initializing parameters")

    def equations_multizones(self, y, t, *argv):
        params = []
        for arg in argv:
            params.append(arg)

        #Get input values at time t
        T_out = self.function_T_out(t)

        Tset1 = self.function_T_set_kitchen(t)
        Tset2 = self.function_T_set_diningroom(t)
        Tset3 = self.function_T_set_livingroom(t)
        Tset4 = self.function_T_set_bathroom(t)
        Tset5 = self.function_T_set_bedroom1(t)
        Tset6 = self.function_T_set_bedroom2(t)
        Tset7 = self.function_T_set_bedroom3(t)
        
        # Get current temperature guessed in each room
        T1 = y[0]
        T2 = y[1]
        T3 = y[2]
        T4 = y[3]
        T5 = y[4]
        T6 = y[5]
        T7 = y[6]

        # Rename all the parameters for the differential equations
        q1 = params[0]
        q2 = params[1]
        q3 = params[2]
        q4 = params[3]
        q5 = params[4]
        q6 = params[5]
        q7 = params[6]
        

        Rw1 = params[7]
        Rw2 = params[8]
        Rw3 = params[9]
        Rw4 = params[10]
        Rw5 = params[11]
        Rw6 = params[12]
        Rw7 = params[13]

        C1 = params[14]
        C2 = params[15]
        C3 = params[16]
        C4 = params[17]
        C5 = params[18]
        C6 = params[19]
        C7 = params[20]

        R12 = params[21]
        R123 = params[22]
        R47 = params[23]
        R45 = params[24]
        R56 = params[25]
        R67 = params[26]

        # Set of differential equation (first order complexity)
        dT1 = (self.q_func(T1, q1, Tset1) - ((T1 - T_out)/Rw1) - ((T1 - T2)/R12) - ((T1 - T3)/R123)) / C1
        dT2 = (self.q_func(T2, q2, Tset2) - ((T2 - T_out)/Rw2) - ((T2 - T1)/R12) - ((T2 - T3)/R123)) / C2
        dT3 = (self.q_func(T3, q3, Tset3) - ((T3 - T_out)/Rw3) - ((T3 - T1)/R123) - ((T3 - T2)/R123)) / C3
        dT4 = (self.q_func(T4, q4, Tset4) - ((T4 - T_out)/Rw4) - ((T4 - T7)/R47) - ((T4 - T7)/R47)) / C4
        dT5 = (self.q_func(T5, q5, Tset5) - ((T5 - T_out)/Rw5) - ((T5 - T4)/R45) - ((T5 - T4)/R45)) / C5
        dT6 = (self.q_func(T6, q6, Tset6) - ((T6 - T_out)/Rw6) - ((T6 - T5)/R56) - ((T6 - T5)/R56)) / C6
        dT7 = (self.q_func(T7, q7, Tset7) - ((T7 - T_out)/Rw7) - ((T7 - T6)/R67) - ((T7 - T6)/R67)) / C7

        
        return dT1, dT2, dT3, dT4, dT5, dT6, dT7 

    def q_func(self, T_in, q_para, T_set):
        if (T_set - T_in) + 1 > 0:
            Q = (T_set - T_in)/q_para
        else:
            Q = 0
    
        return Q

    def obj(self, params, training=True, plot=False):

        if params is None:
            params = self.parameters
        #To see what parameters are used by the optimizer
        print(params)

        # Get the true values of the temperature inside each room during the time set 
        # Simulate the model on the right time set
        if training:
            trueValues = self.y_train
            simulatedValues = self.simulate(params=params)

        else:
            trueValues = self.y_test
            simulatedValues = self.simulate(training=False)
        
        first = time.time()

        # Compute RMSE
        s1 = mean_squared_error(np.array(simulatedValues[:,0]), np.array(trueValues['current_value_kitchen']), squared=False)
        s2 = mean_squared_error(np.array(simulatedValues[:,1]), np.array(trueValues['current_value_diningroom']), squared=False)
        s3 = mean_squared_error(np.array(simulatedValues[:,2]), np.array(trueValues['current_value_livingroom']), squared=False)
        s4 = mean_squared_error(np.array(simulatedValues[:,3]), np.array(trueValues['current_value_bathroom']), squared=False)
        s5 = mean_squared_error(np.array(simulatedValues[:,4]), np.array(trueValues['current_value_bedroom1']), squared=False)
        s6 = mean_squared_error(np.array(simulatedValues[:,5]), np.array(trueValues['current_value_bedroom2']), squared=False)
        s7 = mean_squared_error(np.array(simulatedValues[:,6]), np.array(trueValues['current_value_bedroom3']), squared=False)
        s = (s1 + s2 + s3 + s4 + s5 + s6 + s7)/7
        
        # Keep values to show how it evolves & print current parameters + current objective
        # Maybe plot this if useful?
        self.objectives.append(s)
        self.parametersTried.append(params)
        
        print(s)
        print()

        #Print the distribution of the residuals to see if normal
        #Here must put an indice to plots the residuals for one room

        #residuals = np.array(trueValues - simulatedValues)
        #res = sns.displot(residuals)
        #print(type(res))
        #plt.show()

        # Plot the true values & the simulated ones for each room
        if plot:
            for i in range(7):
                self.plotSimulation(simulatedValues[:,i], trueValues["current_value_"+self.rooms[i]], self.rooms[i], training=training)
            
        second = time.time()
        self.timeObj += (second - first)
        print("computing time of objective : " + str(self.timeObj))
        
        return s
    
    def simulate(self, params=None, training=True):
        # Take the current parameters when training and the parameters of the model when testing
        if params is None:
            params = self.parameters
        
        # Take the initial conditions and the time array depending to the set we use
        if training:
            init = self.init_train
            timeArray = self.time_train
        else:
            init = self.init_test
            timeArray = self.time_test
        first = time.time()
        # Compute the evolution of T_in in a set of time
        predict = odeint(func=self.equations_multizones, y0=init, t=timeArray, args=tuple(params))
        second = time.time()
        self.timeOdeint += second - first
        print("Computing time of odeint : " + str(self.timeOdeint))
            
        return predict
    
    def fit(self):
        # Optimization of the parameters
        result = minimize(self.obj, self.parameters, method=self.solver)#, bounds=self.bounds) # L-BFGS-B powell
        
        # Put the trained parameters in the model
        self.parameters = result.x

        return result  

    def plotSimulation(self, simulatedValues, trueValues, roomName, training=True):
        if training:
            name = "train"
            datesArray = self.dates_train
        else:
            name= "test"
            datesArray = self.dates_test

        plt.figure()
        try:
            plt.title(roomName)
            plt.xlabel('Time')
            plt.ylabel('Temperature (°c)')

            plt.plot(datesArray, simulatedValues, label='Simulated values')
            plt.plot(datesArray, trueValues, label='True values')
            
            plt.legend()

            fname = "plots/Model_simulation_"+name+"_"+roomName
            plt.savefig("{}.png".format(fname))

        finally:
            plt.close()


if __name__ == '__main__':
    
    model = Model()

    #Fit the model
    res = model.fit()
    print(res)

    #Plot the results given by the trainig on the trainig set
    res = model.obj(None, training=True, plot=True)

    #Test the model on the test set and plot results
    res = model.obj(None, training=False, plot=True)
