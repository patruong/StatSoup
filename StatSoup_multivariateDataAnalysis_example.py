# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:31:05 2018

@author: Patrick

Code for tutorials in the useful links:

    http://www.statsmakemecry.com/smmctheblog/stats-soup-anova-ancova-manova-mancova
    http://python-for-multivariate-analysis.readthedocs.io/a_little_book_of_python_for_multivariate_analysis.html#means-and-variances-per-group


"""

import numpy as np
from matplotlib.pyplot import scatter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


import statsmodels.api as sm

from os import listdir
from os.path import isfile, join
from os import chdir
from os import getcwd
from IPython.display import display, HTML

path = "C:\\cygwin64\\home\\Patrick\\StatSoup"

chdir(path)
print(getcwd())


data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
data.columns = ["V"+str(i) for i in range(1, len(data.columns)+1)]  # rename column names to be similar to R naming convention
data.V1 = data.V1.astype(str)
X = data.loc[:, "V2":]  # independent variables data
y = data.V1  # dependent variable data
data


## MATRIX SCATTERPLOT
data.loc[:, "V2":"V6"]

pd.tools.plotting.scatter_matrix(data.loc[:, "V2":"V6"], diagonal="kde")
plt.tight_layout()
plt.show()

pd.tools.plotting.scatter_matrix(data.loc[:, "V2":"V6"], diagonal="hist")
plt.tight_layout()
plt.show()

# Positive relationship between V4 and V5, we plot with lmplot to look closer
sns.lmplot("V4", "V5", data, hue="V1", fit_reg=True);


## PROFILE PLOT
ax = data[["V2","V3","V4","V5","V6"]].plot()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));

# Mean and standard deviation
X.mean()
# https://stackoverflow.com/questions/24984178/different-std-in-pandas-vs-numpy
X.std() # Pandas uses unbiased estimator (N-1) in denominator
X.apply(np.std) # Numpy uses N

"""Very large range of standard deviations, so we need to standardize so that 
it has sample variance of 1 and sample mean of 0"""

## STANDARDISATION

# MEAN and VARIANCE PER GROUP

class1data = data[y=="1"]
class1data.loc[:, "V2":].apply(np.mean)
class1data.loc[:, "V2":].apply(np.std)


class2data = data[y=="2"]
class2data.loc[:, "V2":].apply(np.mean)
class2data.loc[:, "V2":].apply(np.std)

class3data = data[y=="3"]
class3data.loc[:, "V2":].apply(np.mean)
class3data.loc[:, "V2":].apply(np.std)


def printMeanAndSdByGroup(variables, groupvariable):
    data_groupby = variables.groupby(groupvariable)
    print("## Means:")
    display(data_groupby.apply(np.mean))
    print("\n## Standard deviations:")
    display(data_groupby.apply(np.std))
    print("\n## Sample sizes:")
    display(pd.DataFrame(data_groupby.apply(len)))

printMeanAndSdByGroup(X, y)
    
variable = X.V2
groupvariable = y

## Between-Groups Variance and Within-Groups Variance for a variable
## Within-Groups Variance for a variable
def calcWithinGroupsVariance(variable, groupvariable):
    # find out how many values the group variable can take
    levels = sorted(set(groupvariable))
    numlevels = len(levels)
    # get the mean and standard deviation for each group:
    numtotal = 0
    denomtotal = 0
    for leveli in levels:
        levelidata = variable[groupvariable==leveli]
        levelilength = len(levelidata)
        # get the standard deviation for group i:
        sdi = np.std(levelidata)
        numi = (levelilength)*sdi**2
        denomi = levelilength
        numtotal = numtotal + numi
        denomtotal = denomtotal + denomi
    # calculate the within-groups variance
    Vw = numtotal / (denomtotal - numlevels)
    return Vw

def calcWithinGroupsVariance(variable, groupvariable): #own formula for understanding
    # find out how many values the group variable can take
    unique_groups = groupvariable.unique()
    num_unique_groups = len(unique_groups)
    # get the mean and standard deviation for each group
    num_total = 0
    denom_total = 0
    for i in unique_groups:
        group_i = variable[groupvariable==i]
        len_group_i = len(group_i)
        # get the standard deviation for group i
        sd_i = np.std(group_i)
        # Within-group variance formula
        num_i = (len_group_i)*sd_i**2
        denom_i = (len_group_i)
        # Summation procedure in within-group variance formula
        num_total = num_total + num_i
        denom_total = denom_total + denom_i
    S_w = num_total / (denom_total - num_unique_groups)
    return S_w

## Between-Groups Variance for a variable
def calcBetweenGroupsVariance(variable, groupvariable):
    # find out how many values the group variable can take
    levels = sorted(set((groupvariable)))
    numlevels = len(levels)
    # calculate the overall grand mean:
    grandmean = np.mean(variable)
    # get the mean and standard deviation for each group:
    numtotal = 0
    denomtotal = 0
    for leveli in levels:
        levelidata = variable[groupvariable==leveli]
        levelilength = len(levelidata)
        # get the mean and standard deviation for group i:
        meani = np.mean(levelidata)
        sdi = np.std(levelidata)
        numi = levelilength * ((meani - grandmean)**2)
        denomi = levelilength
        numtotal = numtotal + numi
        denomtotal = denomtotal + denomi
    # calculate the between-groups variance
    Vb = numtotal / (numlevels - 1)
    return(Vb)
    
    