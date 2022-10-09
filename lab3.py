#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 19:36:29 2022

@author: jaket
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def updateLI(df):
    for index in df.index:
        LI = (df["L (cm)"][index]/100) *  df["i (A)"][index]
        df.loc[index,"iL"] = LI
    return df

def mult(a, b):
    return [(((a[0]/a[1]) ** 2 + (b[0]/b[1]) ** 2) ** (1/2)) * (a[1] * b[1]), a[1]*b[1]]
def divide(a, b):
   return [(((a[0]/a[1]) ** 2 + (b[0]/b[1]) ** 2) ** (1/2)) * (a[1] / b[1]), a[1]/b[1]]

def updateMG(df):
    for index in df.index:
        mg = (df["m (kg)"][index]/1000) *  9.8
        df.loc[index,"mg"] = mg
    return df

def grapher(fileName):
    df = pd.read_csv(fileName)  
    df = df.iloc[: , :4]
    updateLI(df)
    updateMG(df)
    print(df)
    sns.scatterplot(data=df, x="mg", y="iL", hue="I (A)")
    
    df1 = df.iloc[20:25 , :]
    slope, intercept, r_value, p_value, std_err = stats.linregress(df1['mg'],df1['iL'])
    print(std_err)
    ax = sns.regplot(x="mg", y="iL", data=df1, line_kws={'label':"y={0:.2f}x±{1:.2f} when I = 2".format(slope,std_err)})
    
    ax.legend()
    plt.show()
    
    magneticFields = [12.53, 15.16, 20.77, 17.89, 19.49]
    Ls = [4, 3.5, 3, 2.5, 2]
    plt.plot(Ls, magneticFields, linestyle = "", marker="o")
    plt.plot(np.unique(Ls), np.poly1d(np.polyfit(Ls, magneticFields, 1))(np.unique(Ls)))
    plt.xlabel("Current of Electromagnet I (Amps)")
    plt.ylabel("Magnetic field produced between coils B (Tesla)")
    
    