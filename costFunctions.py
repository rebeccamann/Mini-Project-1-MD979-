import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import json
from types import SimpleNamespace
import math
import mpltern
import matplotlib.colors as mcolors
import datetime

import outbreakFunctions


def QALY_GDPloss(allAgeGrpParams, efficaciesArray, vaccineStart, lockdown1Start, lockdown1End, lockdown2Start, lockdown2End, outbreak2Start, days, schoolClosure):
    
    totalCost = 0
    totalQALY = []
    
    for i in range(len(efficaciesArray)):
        
        simulation = outbreakFunctions.simAllAges(allAgeGrpParams, efficaciesArray[i], vaccineStart, lockdown1Start, lockdown1End, lockdown2Start, lockdown2End, outbreak2Start, days, schoolClosure)
        simulation.ODEsolver( )
        hospY, deadY, vaccY = simulation.outbreakEndResults(0)
        hospA, deadA, vaccA = simulation.outbreakEndResults(1)
        hospE, deadE, vaccE = simulation.outbreakEndResults(2)
        
        
        fatalQALY = (23.5*deadY + 17.6*deadA + 5.3*deadE)
        hospQALY = 10/365 * (0.948*hospY + 0.881*hospA + 0.738*hospE)

        totalQALY.append(fatalQALY + hospQALY)
     
    # GDP does not change with efficacy so we can do this outside the for loop:
    monthlyGDPnoLD = 103.6397
    monthlyGDPinLD = 103.6397 - 50.2415*0.5 + 116.9759*(0.5**2) - 90.7837*(0.5**4)
    dailyGDPloss = (monthlyGDPnoLD*(10**9) - monthlyGDPinLD*(10**9))/30
    
    if schoolClosure==True:
        totalGDPloss = (dailyGDPloss+ 70000000)*((lockdown1End - lockdown1Start) + (lockdown2End - lockdown2Start)) # if schools closed add 0.2bn GDP loss daily
    else:
        totalGDPloss = dailyGDPloss*((lockdown1End - lockdown1Start) + (lockdown2End - lockdown2Start))
    
    return(totalQALY, totalGDPloss)


def totalCost(W, totalQALY, totalGDPloss, weights):
    totalCost=0
    for i in range(len(weights)):
        netMonetaryLoss = W*(totalQALY[i]) + totalGDPloss
    
        totalCost += weights[i] * netMonetaryLoss
    return(totalCost)


def QALY_GDParrays(allAgeGrpParams, efficaciesArray, vaccineStart, lockdown1Start, lockdown2Start, whichLockdownVaried , wks, outbreak2Start, days, schoolClosure):
    
    weekIncrement = wks[1]-wks[0]
    
    QALYarray = []
    GDParray = []
    
    if whichLockdownVaried == 1: #if we are changing the length of LD1:
        if lockdown2Start == days + 1: # if there is no lockdown 2
            for i in wks:
                QALYs, GDP = QALY_GDPloss(allAgeGrpParams, efficaciesArray, vaccineStart, lockdown1Start, lockdown1Start+i*7, days+1 , days+1, outbreak2Start, days, schoolClosure)

                QALYarray.append(QALYs)
                GDParray.append(GDP)
        else: # if there is a 4 week lockdown 2
            for i in wks:
                QALYs, GDP = QALY_GDPloss(allAgeGrpParams, efficaciesArray, vaccineStart, lockdown1Start, lockdown1Start+i*7, lockdown1Start+i*7 + 40 , lockdown1Start+i*7 + 68, outbreak2Start, days, schoolClosure)

                QALYarray.append(QALYs)
                GDParray.append(GDP)
            
    if whichLockdownVaried == 2: #if we are changing the length of LD2:
        for j in wks:
            QALYs, GDP = QALY_GDPloss(allAgeGrpParams, efficaciesArray, vaccineStart, lockdown1Start, lockdown1Start + 4*7, lockdown2Start, lockdown2Start + j*7, outbreak2Start, days, schoolClosure)

            QALYarray.append(QALYs)
            GDParray.append(GDP)
    
    return(QALYarray, GDParray)

def get_category(weeks):
    if weeks == 0:
        return 0
    if 1 <= weeks <= 5:
        return 1
    if 6 <= weeks <= 10:
        return 2
    if 11 <= weeks <= 15:
        return 3
    if 16 <= weeks <= 20:
        return 4
    else:
        return 5
    
def optimalLockdownCategories(allAgeGrpParams, efficaciesArray, vaccineStart, lockdown1Start, lockdown2Start, whichLockdownVaried, W, wks, n, outbreak2Start, days, schoolClosure):
    QALYarray, GDParray = QALY_GDParrays(allAgeGrpParams, efficaciesArray, vaccineStart, lockdown1Start, lockdown2Start, whichLockdownVaried, wks, outbreak2Start, days, schoolClosure)
    
    weekIncrement = wks[1] - wks[0]
    w1,w2,w3 = [],[],[]
    resultsCat = []
    resultsLoss = []
    
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j

            v1, v2, v3 = i/n, j/n, k/n
            w1.append(v1)
            w2.append(v2)
            w3.append(v3)
            
            costArray = []
            for l in range(len(wks)):
                costArray.append(totalCost(W, QALYarray[l], GDParray[l], [v1,v2,v3]))

            optimalLockdown = costArray.index(min(costArray))*weekIncrement
            resultsCat.append(get_category(optimalLockdown))
            resultsLoss.append(min(costArray))

    return(w1,w2,w3, resultsCat, resultsLoss)


def trianglePlotterSchoolsOpenVsClosed(W, schoolsOpenArray, schoolsClosedArray):    
    #unpacking the schools open/closed arrays:
    w1_open, w2_open, w3_open, results_open = schoolsOpenArray
    w1_closed, w2_closed, w3_closed, results_closed = schoolsClosedArray
    
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'pink']
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4, 5, 6] 
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(15, 7))
    plt.suptitle(f"Willingness to pay £{W} ", fontsize = 25)
    
    # SCHOOLS OPEN
    ax = fig.add_subplot(1,2,1, projection='ternary')

    plot = ax.tripcolor(w1_open, w2_open, w3_open, results_open, cmap=cmap, norm=norm, shading='flat')
    ax.set_tlabel('$w_2 = 0$', fontsize=20)
    ax.set_llabel('$w_3 = 0$', fontsize=20)
    ax.set_rlabel('$w_1 = 0$', fontsize=20)
    ax.taxis.set_label_position('tick1')
    ax.laxis.set_label_position('tick1')
    ax.raxis.set_label_position('tick1')
    ax.set_title("Schools Open", fontsize = 15)

    ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.taxis.set_ticks(ticks)
    ax.laxis.set_ticks(ticks)
    ax.raxis.set_ticks(ticks)
    ax.taxis.set_ticklabels([])
    ax.laxis.set_ticklabels([])
    ax.raxis.set_ticklabels([])

    ax.grid(True, linestyle=':', color='black', alpha=0.3)
    
    # SCHOOLS CLOSED
    ax = fig.add_subplot(1,2,2, projection='ternary')

    plot = ax.tripcolor(w1_closed, w2_closed, w3_closed, results_closed, cmap=cmap, norm=norm, shading='flat')
    ax.set_tlabel('$w_2 = 0$', fontsize=20)
    ax.set_llabel('$w_3 = 0$', fontsize=20)
    ax.set_rlabel('$w_1 = 0$', fontsize=20)
    ax.taxis.set_label_position('tick1')
    ax.laxis.set_label_position('tick1')
    ax.raxis.set_label_position('tick1')
    ax.set_title("Schools Closed", fontsize = 15)

    ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.taxis.set_ticks(ticks)
    ax.laxis.set_ticks(ticks)
    ax.raxis.set_ticks(ticks)
    ax.taxis.set_ticklabels([])
    ax.laxis.set_ticklabels([])
    ax.raxis.set_ticklabels([])

    ax.grid(True, linestyle=':', color='black', alpha=0.3)


    cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
    colorbar = fig.colorbar(plot, cax=cax, ticks=[0.5,1.5,2.5,3.5,4.5, 5.5])
    colorbar.ax.set_yticklabels(['None', '1-5 weeks', '6-10 weeks', '11-15 weeks', '16-20 weeks', '21+ weeks'])
    colorbar.set_label("Weeks of Lockdown 1", rotation=270, fontsize = 15, va="baseline")



    
    # filename = input('filename >> ')
    # if filename != 'x':
    #     plt.savefig(f'{filename}{str(datetime.datetime.now())[11:19]}.png', bbox_inches='tight') # Add this to save the figures
        
    plt.show()