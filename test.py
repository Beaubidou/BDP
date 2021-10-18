import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import math
from scipy.integrate import odeint
import dataset


pd.set_option("display.max.columns", None)
d = dataset.data.data
TestF = d.head(4)

def T_amb(t):
	data = TestF['current_value_outside']
	t = int(t)
	return data[t]

def equ_diff_1zone(y, t, q, R, C):
	Tz = y[0]

	dTz = (q -(Tz - T_amb(t))/R)/C  
	return dTz


def g(t, initial, paras):
	predict = odeint(func = equ_diff_1zone, y0 = initial , t = time, args = tuple(paras))
	return predict
def obj(paras, t, data):#add more if other obj function
	

	T_data = data
	inistate = T_data[0]
	T_sim = g(t, inistate, paras)

	s = np.sum((np.array(T_sim) - np.array(T_data))**2)

	return s



time = np.linspace(0., 3., 3)

params = 1 , 1, 1

def fit(time, params, data):

	result = minimize(obj, params, args = (time, data), method = 'L-BFGS-B')
	return result

res = fit(time, params, TestF['current_value_livingroom'])
q = res.x[0]
R = res.x[1]
C = res.x[2]
params = q, R , C
inistate = TestF['current_value_livingroom'][0]
fitted = g(time, inistate, params)

print(fitted)
print
print(TestF['current_value_livingroom'])

# res = fit(parameters)
# print(res)
# parameters = res
# fit(parameters)
# print(res)
