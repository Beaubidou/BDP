import matplotlib.pyplot as plt
import time

from scipy.optimize import minimize
from scipy.integrate import odeint
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import norm

from dataset import Dataset


class model:
    # Sets to plot the evolution of the objective function and the parameters tried by the model
    objectives = []
    parametersTried = []
    rooms = ['kitchen', 'diningroom', 'livingroom', 'bathroom', 'bedroom1', 'bedroom2', 'bedroom3']


    def __init__(self):
        #self.R = [15, 15, 30, 10, 15, 15, 12]
        self.R = [6, 1.5, 2, 2, 2, 2, 2]
        self.C = [4*10**4, 4*10**4, 8*10**4, 2*10**4, 4*10**4, 5*10**4, 3*10**4]
        #self.q = [125, 125, 250, 62, 125, 160, 100]
        self.q = [100, 100, 200, 50, 100, 130, 80]
        # R12 R123 R47 R45 R56 R67
        self.R_neighbour = [5, 5, 5, 5, 5, 5]

        self.variance = 1

        tmp = [self.variance] + self.q + self.R + self.C + self.R_neighbour
        self.parameters = tmp

        
        
        
        # Length of the set taken to train and test (33% for testing)
        length = 2000

        ######################## Create dataset #######################
        # Take the data relative to the house to model
        dataset = Dataset(reload_all=False, interpolate=True)

        # y_train and y_test return padas.Dataframes for true temperature inside the house
        # The index is ['kitchen', 'diningroom', 'livingroom', 'bathroom', 'bedroom1', 'bedroom2', 'bedroom3']
        # dates_train and dates_test is juste a time vector of the observations for x axis in plots
        self.dates_train, self.dates_test, self.y_train, self.y_test = dataset.train_test_sample_split(start_date="2020-11-03 10:40:00", length=length, multi_z=True)
        
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

    def q_func(self, T_in, q_para, T_set):
        if (T_set - T_in) + 1 > 0:
            #Q = (T_set - T_in)/q_para
            Q = q_para #will give us a line -> I think there is no correlation between the diff and the power developed by the heater
            #devrait donner une ligne de pente proportionelle à q_para/5min + forme condensateur via C
        else:
            Q = 0
    
        return Q
    
    def equations(self, y, t, params):
        

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
        dT1 = 300*(self.q_func(T1, q1, Tset1) - ((T1 - T_out)/Rw1) - ((T1 - T2)/R12) - ((T1 - T3)/R123)) / C1
        dT2 = 300*(self.q_func(T2, q2, Tset2) - ((T2 - T_out)/Rw2) - ((T2 - T1)/R12) - ((T2 - T3)/R123)) / C2
        dT3 = 300*(self.q_func(T3, q3, Tset3) - ((T3 - T_out)/Rw3) - ((T3 - T1)/R123) - ((T3 - T2)/R123)) / C3
        dT4 = 300*(self.q_func(T4, q4, Tset4) - ((T4 - T_out)/Rw4) - ((T4 - T7)/R47) - ((T4 - T7)/R47)) / C4
        dT5 = 300*(self.q_func(T5, q5, Tset5) - ((T5 - T_out)/Rw5) - ((T5 - T4)/R45) - ((T5 - T4)/R45)) / C5
        dT6 = 300*(self.q_func(T6, q6, Tset6) - ((T6 - T_out)/Rw6) - ((T6 - T5)/R56) - ((T6 - T5)/R56)) / C6
        dT7 = 300*(self.q_func(T7, q7, Tset7) - ((T7 - T_out)/Rw7) - ((T7 - T6)/R67) - ((T7 - T6)/R67)) / C7

        
        return np.array([T1 + dT1, T2 + dT2, T3 + dT3, T4 + dT4, T5 + dT5, T6 + dT6, T7 + dT7]) 
    
    def log_likelihood(self, params, training=True, plot=False):
        #R, C, log_f = theta

        if params is None:
            params = self.parameters
        #To see what parameters are used by the optimizer
        print(params)

        # Get the true values of the temperature inside each room during the time set 
        # Simulate the model on the right time set
        if training:
            ObsValues = self.y_train
            simulatedValues = self.simulate(params=params[1:], plot=plot)

        else:
            ObsValues = self.y_test
            simulatedValues = self.simulate(training=False, plot=plot)

        #noise_var = np.exp(log_f) * 0.1
        res = np.sum(norm.logpdf(ObsValues, loc=simulatedValues, scale=params[0]))/len(ObsValues)
        
        self.objectives.append(res)
        self.parametersTried.append(params)

        return res

    def simulate(self, params=None, training=True, plot=False):
        # Take the current parameters when training and the parameters of the model when testing
        if params is None:
            params = self.parameters[1:]
        
        # Take the initial conditions and the time array depending to the set we use
        if training:
            #!!!!!On doit trouver qqch d'autre pour init (je crois qu'on peut pas partir d'une valeur connue)
            init = self.init_train
            timeArray = self.time_train
            trueValues = self.y_train
        else:
            init = self.init_test
            timeArray = self.time_test
            trueValues = self.y_test

        T_house = np.zeros((len(timeArray)+1,7))
        T_house[0,:] = init
        # Compute the evolution of T_in in a set of time
        #predict = odeint(func=self.equations_multizones, y0=init, t=timeArray, args=tuple(params))
        for i, t in enumerate(timeArray):
            T_house[i+1,:] = self.equations(T_house[i,:], t, params)
        
        # Plot the true values & the simulated ones for each room
        if plot:
            for i in range(7):
                self.plotSimulation(T_house[1:,i], trueValues["current_value_"+self.rooms[i]], self.rooms[i], training=training)
            
        
        return T_house[1:,:]

    def optimize(self):

        nll = lambda *args: -self.log_likelihood(*args)
        soln = minimize(nll, self.parameters)
        self.parameters = soln.x
        return soln

    def plotSimulation(self, simulatedValues, trueValues, roomName, training=True):
        if training:
            name = "train"
            datesArray = self.dates_train
            T_out_vec = self.function_T_out(self.time_train)
            T_set = self.getT_set(roomName, self.time_train)
        else:
            name= "test"
            datesArray = self.dates_test
            T_out_vec = self.function_T_out(self.time_test)
            T_set = self.getT_set(roomName, self.time_test)
        

        plt.figure()
        try:
            plt.title(roomName)
            plt.xlabel('Time')
            plt.ylabel('Temperature (°c)')

            plt.plot(datesArray, trueValues, "b.", label='Measured temperatures', alpha=0.1)
            plt.plot(datesArray, simulatedValues, label='Simulated temperatures')
            plt.plot(datesArray, T_out_vec, label='Temperature outside', alpha=0.3)
            plt.plot(datesArray, T_set, label='Set point', alpha=0.3)

            plt.legend()

            fname = "plots/Model_simulation_"+name+"_"+roomName
            plt.savefig("{}.png".format(fname))

        finally:
            plt.close()

    def getT_set(self, name, set):
        if name == 'kitchen':
            T = self.function_T_set_kitchen(set)
        elif name == 'livingroom':
            T = self.function_T_set_livingroom(set)
        elif name == 'diningroom':
            T = self.function_T_set_diningroom(set)
        elif name == 'bathroom':
            T = self.function_T_set_bathroom(set)
        elif name == 'bedroom1':
            T = self.function_T_set_bedroom1(set)
        elif name == 'bedroom2':
            T = self.function_T_set_bedroom2(set)
        elif name == 'bedroom3':
            T = self.function_T_set_bedroom3(set)
        return T

    

if __name__ == '__main__':

    m = model()

    #First see result with initial parameters
    initialObj = m.log_likelihood(None, training=True, plot=True)
    print("The initial objective value is : " + str(-initialObj))

    #Then optimize the parameters
    res = m.optimize()
    print(res)

    #See results on the trainig set (how well the optimization worked)
    finalObj = m.log_likelihood(None, training=True, plot=True)
    print("The objective value after training is : " + str(-finalObj))

    #See results on the test set (real efficiency of the model)
    testObj = m.log_likelihood(None, training=False, plot=True)
    print("The objective value on the test set is : " + str(-testObj))
    