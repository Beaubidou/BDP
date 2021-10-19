import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import math
from scipy.integrate import odeint
from dataset import Dataset


pd.set_option("display.max.columns", None)
d = Dataset().data
start = 2000
number_of_datapoint = 200
TestF = d.loc[start:start+number_of_datapoint]
objectives = []


def T_amb(t):
	data = d['current_value_outside']
	tdivided = t #convert to index
	tfinal = int(tdivided)
	# if tfinal >= number_of_datapoint:
	# 	return data[number_of_datapoint-1]
	return data[tfinal]
def q(t, q_para):
	t = int(t)
	q = q_para*(d['setpoint_livingroom'][t]**2)
	return q
def equ_diff_1zone(y, t, q_para , R, C):
	Tz = y[0]

	dTz = (q(t, q_para) -(Tz - T_amb(t))/R)/C  
	return dTz


def g(t, initial, paras):
	predict = odeint(func = equ_diff_1zone, y0 = initial , t = time, args = tuple(paras))
	return predict
def obj(paras, t, data):#add more if other obj function
	

	T_data = data
	inistate = T_data[start]
	T_sim = g(t, inistate, paras)

	s = np.sum((np.array(T_sim) - np.array(T_data))**2)
	objectives.append(s)

	return s



times = TestF.index[0]
timef = TestF.index[-1]

time = np.linspace(times, timef, (timef-times)+1)

params = 1, 1, 1

def fit(time, params, data):

	result = minimize(obj, params, args = (time, data), method = 'Powell')
	return result

res = fit(time, params, TestF['current_value_livingroom'])

q_para = res.x[0]
R = res.x[1]
C = res.x[2]
params = q_para, R , C
inistate = TestF['current_value_livingroom'][start]
fitted = g(time, inistate, params)

print(params)
print(fitted)
print
print(TestF['current_value_livingroom'])

# plt.plot(objectives[50:])
# plt.show()

# res = fit(parameters)
# print(res)
# parameters = res
# fit(parameters)
# print(res)
