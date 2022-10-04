#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 16:48:45 2022

@author: jaket
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def addOrSub(a, b):
    return [(a[0] ** 2 + b[0] ** 2) ** (1/2), a[1] + b[1]] 
def mult(a, b):
    return [(((a[0]/a[1]) ** 2 + (b[0]/b[1]) ** 2) ** (1/2)) * (a[1] * b[1]), a[1]*b[1]]
def divide(a, b):
   return [(((a[0]/a[1]) ** 2 + (b[0]/b[1]) ** 2) ** (1/2)) * (a[1] / b[1]), a[1]/b[1]]
def power(a, n):
    return [((a[0]/a[1]) * n) * (a[1]**n), a[1]**n]

def vinitial(deltaH, deltaPrime):
    h = addOrSub(deltaH, deltaPrime)
    squared =  [h[0], h[1] * 9.8 * (10/7)]
    return power(squared, (0.5))
    
def vy(vinitial, h2, h3, L):
    subbed = addOrSub(h2, h3)
    divided = divide(subbed, L)
    return mult(divided, vinitial)

def final(vy, vinitial, h2, D, L):
    divideDL = divide(D, L)
    multDLV = mult(vinitial, divideDL)
    
    updateH2 = [h2[0], 2*9.8*h2[1]]
    vysquare = power(vy, 2)
    addvyh2 = addOrSub(vysquare, updateH2)
    sqrted = power(addvyh2, 0.5)
    addedvy = addOrSub(vy, sqrted)
    updatedAdded = [addedvy[0], addedvy[1]/9.8]
    
    return mult(multDLV, updatedAdded)

def propogate(deltaH, deltaPrime, h2, h3, L, D):
    vinit = vinitial(deltaH, [deltaPrime[0], -deltaPrime[1]])
    vwhy = vy(vinit, h2, [h3[0], -h3[1]], L)
    return final(vwhy, vinit, h2, D, L)

def calcMeanDistance(vals, estimate):
    summer = 0
    for x in vals:
        y = x / 100
        estimates = estimate + y
        summer += estimates
    return round(summer/20, 3)

def calcStdDev(vals, mean, estimate):
    summer = 0
    for x in vals:
        y = x / 100
        estimates = estimate + y
        summer += (estimates - mean) ** 2
    return round((summer/19) ** (1/2), 3)
        

def calcMean(fileName, estimate):
    df = pd.read_csv(fileName, header = 2)  
    df = df.iloc[: , :-1]
    z1 = df["x"].to_numpy()
    x1 = df["z"].to_numpy()
    z2 = df["x.1"].to_numpy()
    x2 = df["z.1"].to_numpy()
    z3 = df["x.2"].to_numpy()
    x3 = df["z.2"].to_numpy()
    z4 = df["x.3"].to_numpy()
    x4 = df["z.3"].to_numpy()
    
    x1mean = calcMeanDistance(x1, estimate)
    x2mean = calcMeanDistance(x2, estimate)
    x3mean = calcMeanDistance(x3, estimate)
    x4mean = calcMeanDistance(x4, estimate)
    
    x1std = calcStdDev(x1, x1mean, estimate)
    x2std = calcStdDev(x2, x2mean, estimate)
    x3std = calcStdDev(x3, x3mean, estimate)
    x4std = calcStdDev(x4, x4mean, estimate)
    
    z1mean = calcMeanDistance(z1, 0)
    z2mean = calcMeanDistance(z2, 0)
    z3mean = calcMeanDistance(z3, 0)
    z4mean = calcMeanDistance(z4, 0)
    
    z1std = calcStdDev(z1, z1mean, 0)
    z2std = calcStdDev(z2, z2mean, 0)
    z3std = calcStdDev(z3, z3mean, 0)
    z4std = calcStdDev(z4, z4mean, 0)
    
    
    
    print(f"Average x1: {x1mean} Std Dev: {x1std} Std Dev Mean: {round(x1std/(20 ** 0.5), 3)}")
    print(f"Average z1: {z1mean} Std Dev: {z1std} Std Dev Mean: {round(z1std/(20 ** 0.5), 3)}")
    print(f"Average x2: {x2mean} Std Dev: {x2std} Std Dev Mean: {round(x2std/(20 ** 0.5), 3)}")
    print(f"Average z2: {z2mean} Std Dev: {z2std} Std Dev Mean: {round(z2std/(20 ** 0.5), 4)}")
    print(f"Average x3: {x3mean} Std Dev: {x3std} Std Dev Mean: {round(x3std/(20 ** 0.5), 3)}")
    print(f"Average z3: {z3mean} Std Dev: {z3std} Std Dev Mean: {round(z3std/(20 ** 0.5), 3)}")
    print(f"Average x4: {x3mean} Std Dev: {x4std} Std Dev Mean: {round(x4std/(20 ** 0.5), 3)}")
    print(f"Average z4: {z4mean} Std Dev: {z4std} Std Dev Mean: {round(z4std/(20 ** 0.5), 3)}")
    
    plt.hist(x1)
    plt.title("Metal Ball Trial 1")
    plt.xlabel("Distance from estimate in x (cm)")
    plt.ylabel("Number of occurances")
    plt.show()
    
    plt.hist(x2)
    plt.title("Metal Ball Trial 2")
    plt.xlabel("Distance from estimate in x (cm)")
    plt.ylabel("Number of occurances")
    plt.show()
    
    plt.hist(z1)
    plt.title("Metal Ball Trial 1")
    plt.xlabel("Distance from estimate in z (cm)")
    plt.ylabel("Number of occurances")
    plt.show()
    
    plt.hist(z2)
    plt.title("Metal Ball Trial 2")
    plt.xlabel("Distance from estimate in z (cm)")
    plt.ylabel("Number of occurances")
    plt.show()
    
    plt.hist(x3)
    plt.title("Plastic Ball Trial 1")
    plt.xlabel("Distance from estimate in x (cm)")
    plt.ylabel("Number of occurances")
    plt.show()
    
    plt.hist(x4)
    plt.title("Plastic Ball Trial 2")
    plt.xlabel("Distance from estimate in x (cm)")
    plt.ylabel("Number of occurances")
    plt.show()
    
    plt.hist(z3)
    plt.title("Plastic Ball Trial 1")
    plt.xlabel("Distance from estimate in z (cm)")
    plt.ylabel("Number of occurances")
    plt.show()
    
    plt.hist(z4)
    plt.title("Plastic Ball Trial 2")
    plt.xlabel("Distance from estimate in z (cm)")
    plt.ylabel("Number of occurances")
    plt.show()


    