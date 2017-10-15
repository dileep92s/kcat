import pandas as pd
import numpy as np
from numpy import dot, sum, tile, linalg
from numpy.linalg import inv 
import matplotlib.pyplot as plt

import time
from visual import *
import math
import pickle


def kalman(z):
	# intial parameters
	n_iter = z.shape[0]
	sz = (n_iter,) # size of array

	Q = 1e-5 # process variance

	# allocate space for arrays
	xhat=np.zeros(sz)      # a posteri estimate of x
	P=np.zeros(sz)         # a posteri error estimate
	xhatminus=np.zeros(sz) # a priori estimate of x
	Pminus=np.zeros(sz)    # a priori error estimate
	K=np.zeros(sz)         # gain or blending factor

	R = 0.1**2 # estimate of measurement variance, change to see effect

	# intial guesses
	xhat[0] = z.iloc[0]
	P[0] = 1.0

	for k in range(1,n_iter):
		# time update
		xhatminus[k] = xhat[k-1]
		Pminus[k] = P[k-1]+Q

		# measurement update
		K[k] = Pminus[k]/( Pminus[k]+R )
		xhat[k] = xhatminus[k]+K[k]*(z.iloc[k]-xhatminus[k])
		P[k] = (1-K[k])*Pminus[k]
		
	return xhat

xl = pd.ExcelFile("kitcat.xlsx")
df = xl.parse("sh1")

w1 = df[["Wearable-1", "Ax","Ay","Az","Gx","Gy","Gz","Sensor time"]].dropna()
w2 = df[["Wearable-2", "Ax.1","Ay.1","Az.1","Gx.1","Gy.1","Gz.1","Sensor time.1"]].dropna()

gx = kalman(w2[["Gx.1"]])
gy = kalman(w2[["Gy.1"]])
gz = kalman(w2[["Gz.1"]])

'''
with open("dataset","w") as f:
	pickle.dump((x,y,z),f)


with open("dataset") as f:
	gx,gy,gz = pickle.load(f)	
'''

f1 = frame()
bo = box(frame=f1,pos=vector(0,-5,0), size=vector(20,3,3), axis=vector(1,0,0), color=color.red)
bk = box(frame=f1,pos=vector(-8,-8,0), size=vector(3,3,3), axis=vector(1,0,0), color=color.blue)
fw = box(frame=f1,pos=vector(8,-8,0), size=vector(3,3,3), axis=vector(1,0,0), color=color.green)
hd = box(frame=f1,pos=vector(8,-2,0), size=vector(3,3,3), axis=vector(1,0,0), color=color.white)

# f2 = frame()
# bo = box(frame=f2,pos=vector(0,-5,0), size=vector(20,3,3), axis=vector(1,0,0), color=color.red)
# bk = box(frame=f2,pos=vector(-8,-8,0), size=vector(3,3,3), axis=vector(1,0,0), color=color.blue)
# fw = box(frame=f2,pos=vector(8,-8,0), size=vector(3,3,3), axis=vector(1,0,0), color=color.green)
# hd = box(frame=f2,pos=vector(8,-2,0), size=vector(3,3,3), axis=vector(1,0,0), color=color.white)

xo,yo,zo = 0,0,0
for x,y,z in zip(gz,gy,gx):
	xn,yn,zn = xo-x,yo-y,zo-z
	print xn,yn,zn
	f1.rotate(angle=(math.radians(xn)), axis=vector(1, 0, 0))
	f1.rotate(angle=(math.radians(yn)), axis=vector(0, 1, 0))
	f1.rotate(angle=(math.radians(zn)), axis=vector(0, 0, 1))
	xo,yo,zo = x,y,z
	rate(30)
