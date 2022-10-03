#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 16:48:45 2022

@author: jaket
"""

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
    