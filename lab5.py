#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 16:49:15 2022

@author: jaket
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

def lab5(fileName):
    df = pd.read_csv(fileName)  
    df = df.iloc[: , :9]
    
    x = df.iloc[:, :]["cos^2(theta)"].to_numpy()
    y = df.iloc[:, :]["is"].to_numpy()
    err = df.iloc[:, :]["error"].to_numpy()
    
    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)  # to create a linear function with coefficients

    plt.plot(x, y, 'ro', x, poly1d_fn(x), '-b')
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(df["cos^2(theta)"],df["is"])
    print(std_err)
    #sns.regplot(x="cos^2(theta)", y="is", data=df, line_kws={'label':"y={0:.3f}x±{1:.3f} when I = 2".format(slope,std_err)})
    print(slope)
    print(intercept)
    plt.errorbar(x, y, yerr=err, fmt="o")
    plt.xlabel("cos^2(theta)")
    plt.ylabel("I/I0")
    plt.show()
    
    df = pd.read_csv("lab5data2.csv")  
    df = df.iloc[: , :6]
    
    x = df.iloc[:, :]["m"].to_numpy()
    y = df.iloc[:, :]["xm"].to_numpy()
    err = df.iloc[:, :]["error"].to_numpy()
    
    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)  # to create a linear function with coefficients

    plt.plot(x, y, 'ro', x, poly1d_fn(x), '-b', color="red")
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(df["m"],df["xm"])
    print(std_err)
    #sns.regplot(x="cos^2(theta)", y="is", data=df, line_kws={'label':"y={0:.3f}x±{1:.3f} when I = 2".format(slope,std_err)})
    print(slope)
    print(intercept)
    plt.errorbar(x, y, yerr=err, fmt="o")
    plt.xlabel("m")
    plt.ylabel("xm")
    plt.show()
    
    df = pd.read_csv("lab5data3.csv")  
    
    x = df.iloc[:, :]["n"].to_numpy()
    y = df.iloc[:, :]["xn"].to_numpy()
    err = df.iloc[:, :]["error"].to_numpy()
    
    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)  # to create a linear function with coefficients

    plt.plot(x, y, 'ro', x, poly1d_fn(x), '-b', color="red")
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(df["n"],df["xn"])
    print(std_err)
    #sns.regplot(x="cos^2(theta)", y="is", data=df, line_kws={'label':"y={0:.3f}x±{1:.3f} when I = 2".format(slope,std_err)})
    print(slope)
    print(intercept)
    plt.errorbar(x, y, yerr=err, fmt="o")
    plt.xlabel("n")
    plt.ylabel("xn")
    plt.show()
    
    