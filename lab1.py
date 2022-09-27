#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:46:39 2022

@author: jaket
"""
import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt

def readCV(name, sheetname):
    df = pd.read_excel(name, header=[1], sheet_name=sheetname)
    return df

def velAtT(a, t, v0):
    return a * t + v0

def posAtT(a, t, v0, x0):
    return (1/2 * a * (t ** 2)) + v0 * t + x0

def calcTheta(df):
    for index in df.index:
        height = (df["h (mm)"][index])/1000
        length = df["L (m)"][index]
        theta = np.rad2deg(np.arcsin(height/ length).item())
        df.loc[index,"θ"] = theta
    return df

def updateL2(df):
    for index in df.index:
        L2 = df["y2"][index] - df["y1"][index]
        df.loc[index,"L2"] = L2
    return df

def updateFriction(df):
    for index in df.index:
        fric = ((df["v1"][index] ** 2) / (2 * df["ax"][index] * df["L2"][index])) - 1
        df.loc[index,"fric"] = fric
    return df

def updateRest(df):
    for index in df.index:
        rest = restitution(df["vf"][index], df["vi"][index])
        df.loc[index,"e"] = rest
    return df

def updatedUncertainty(df):
    for index in df.index:
        uncertaine = uncertainTE(df["σvi"][index], df["σvf"][index], partialVISqaure(df["vi"][index], df["vf"][index]), partialVFSquare(df["vi"][index], df["vf"][index]))
        df.loc[index,"σe"] = uncertaine
    return df;

def restitution(vf, vi):
    return abs(vf) / abs(vi)

def avgRest(rests):
    return sum(rests)/len(rests)

def avgRest2(rests, ebar):
    x = 0
    for i in range(len(rests)-1):
        x = x + ((rests[i] - ebar) ** 2)
    return x

def partialVISqaure(vi, vf):
    return (-1 * ((vf) / (vi ** 2))) ** 2

def partialVFSquare(vi, vf):
    return (1 / (vi)) ** 2

def uncertainTE(ov1, ov2, pv1, pv2):
    return ((ov1**2) * pv1 + (ov2**2) * pv2) ** (1/2)

def weightedAvg(rests, df):
    x = 0
    for i in range(len(rests)):
        x += rests[i] / (df["σe"][i] ** (2))
    y = 0
    for i in range(len(rests)):
        y += 1/(df["σe"][i] ** (2))
    
    return x/y

def weightedAvgSigma(df):
    x = 0
    for index in df.index:
        x += 1/(df["σe"][index] ** (2))
    return (x ** (-1/2))

def avgax(df, h):
    sum = 0
    for i in range((h - 1) * 10 , (h - 1) * 10 + 10):
        if(math.isnan(df["ax"][i])):
            return sum/8
        sum += df["ax"][i]
    return sum/10

def sigmaax(df, h, a_x):
    sum = 0
    for i in range((h - 1) * 10 , (h - 1) * 10 + 10):
        if(math.isnan(df["ax"][i])):
            return ((sum / (8 - 1)) ** 1/2) / (8 ** (1/2)) 
        sum += (a_x -  df["ax"][i]) ** 2
    return ((sum / (10 - 1)) ** 1/2) / (10 ** (1/2)) 
    
    
def main():
    #Part 1
    df1 = readCV("lab1spread.xlsx", "sheet1")
    df1 = updateRest(df1)
    df1 = updatedUncertainty(df1);
    print(df1)
    restLst = df1["e"].to_numpy()
    avg = avgRest(restLst)
    print(f"Unweighted Avg restitution: {avg}")
    sigma = (avgRest2(restLst, avg) / (len(restLst) - 1)) ** (1/2)
    sigmaebar = sigma / (len(restLst) ** (1/2))
    print(f"Sigma bar = {sigmaebar}")
    df1 = df1.astype(float)
    print(f"Weighted average rest: {weightedAvg(restLst, df1)}")
    print(f"Weighted average sigma: {weightedAvgSigma(df1)}")
    #sns.regplot(x = "vi", y = "e", data = df1)
        

    
    
    
    #Part 2
    df2 = readCV("lab1spread.xlsx", "sheet2")
    df2 = calcTheta(df2)
    df2 = updateL2(df2)
    df2 = updateFriction(df2)
    print(df2.to_string())
    avgAXList = []
    for i in range(1, 6):
        avgAXList.append(avgax(df2, i))
    avgASigList = []
    for i in range(1, 6):
        avgASigList.append(sigmaax(df2, i, avgAXList[i-1]))
    print(f"Average a_x's: {avgAXList}")
    print(f"Average a_x's: {avgASigList}")
    df3 = pd.DataFrame(list(zip(avgAXList, [0.001, 0.002, 0.003, 0.004, 0.005])), columns=["a_x", "h"])
    x = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
    #sns.regplot(x = "h", y = "a_x", data = df3, ax=ax)
    a, b = np.polyfit(x, avgAXList, 1)
    plt.scatter(x, avgAXList)
    plt.plot(x, a*x+b) 
    plt.errorbar(x, avgAXList, yerr = avgASigList, fmt="o")
    plt.xlabel("height in m")
    plt.ylabel("x component of acceleration")
    print(f"Estimation of gravity: {a}")

       
    

    
    
    
if __name__ == "__main__":
    main()
    

    


        

