# -*- coding: utf-8 -*-
"""
Spyder Editor
Serge ADOMAYAKPO
This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Import du dataset
bsk= pd.read_csv("Market_Basket_Optimisation.csv", header= None)

transac= []
for i in range(0,7501):
    transac.append([str(bsk.values[i,j]) for j in range(0,20)])
    
    
#Train du apriori
from apyori import apriori

regles= apriori(transac, min_support= 0.003, min_confidence= 0.2, min_lift=3, min_length=2)

#results= list(regles)


results_list = []

for i in range(0, len(regles)):

    results_list.append('RULE:\t' + str(regles[i][0]) + '\nSUPPORT:\t' + str(regles[i][1]))