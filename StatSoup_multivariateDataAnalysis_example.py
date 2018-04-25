# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:31:05 2018

@author: Patrick

Code for tutorials in the useful links:

    http://www.statsmakemecry.com/smmctheblog/stats-soup-anova-ancova-manova-mancova
    http://python-for-multivariate-analysis.readthedocs.io/a_little_book_of_python_for_multivariate_analysis.html#means-and-variances-per-group


"""
from __future__ import print_function


import numpy as np
from matplotlib.pyplot import scatter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
from IPython.display import display, HTML


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
    V_w = num_total / (denom_total - num_unique_groups)
    return V_w

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
    
#variable = X.V2
#groupvariable = Y
def calcBetweenGroupsVariance(variable, groupvariable): #own formula for understanding
    
    # find out how many values the group variable can take
    unique_groups = groupvariable.unique()
    num_unique_groups = len(unique_groups)
    
    #calculate the overall grand mean
    grand_mean = np.mean(variable)
    
    # the the mean and standard deviation for each group
    num_total = 0
    denom_total = 0
    for i in unique_groups:
        group_i = variable[groupvariable == i]
        len_group_i = len(group_i)
        # get the mean and standard deviation for group i
        mean_i = np.mean(group_i)
        std_i = np.std(group_i)
        # Between-group variance formula
        num_i = (len_group_i) * ((mean_i - grand_mean)**2)
        denom_i = (len_group_i)
        # Summation procedure in between-group variance formula
        num_total = num_total + num_i
        denom_total = denom_total + denom_i
    # valvulate the between-groups variance
    V_b = num_total / (num_unique_groups - 1) 
    return(V_b)

# Calculate seperation achieved by a variable - Seperation achieved by V2
calcBetweenGroupsVariance(X.V2, y) / calcWithinGroupsVariance(X.V2, y)

# Calculate Seperation acieved by all variables in a multivariate dataset
def calcSeparations(variables, groupvariable):
    # calculate the separation for each variable
    for variablename in variables:
        variablei = variables[variablename]
        Vw = calcWithinGroupsVariance(variablei, groupvariable)
        Vb = calcBetweenGroupsVariance(variablei, groupvariable)
        sep = Vb/Vw
        print("variable", variablename, "Vw=", Vw, "Vb=", Vb, "separation=", sep)

#variables = X
#groupvariable = y

def calcSeparations(variables, groupvariable):
    # Calculate the separation for each variable
    for i in variables:
        variable_i = variables[i]
        V_w = calcWithinGroupsVariance(variable_i, groupvariable) 
        V_b = calcBetweenGroupsVariance(variable_i, groupvariable)
        sep = V_b/V_w
        print( "variable", i, "V_w=", V_w, "V_b=", V_b, "separation=", sep)
    

## Between-groups Covariance and Within-groups Covariance for Two Variables

def calcWithinGroupsCovariance(variable1, variable2, groupvariable):
    levels = sorted(set(groupvariable))
    numlevels = len(levels)
    Covw = 0.0
    # get the covariance of variable 1 and variable 2 for each group:
    for leveli in levels:
        levelidata1 = variable1[groupvariable==leveli]
        levelidata2 = variable2[groupvariable==leveli]
        mean1 = np.mean(levelidata1)
        mean2 = np.mean(levelidata2)
        levelilength = len(levelidata1)
        # get the covariance for this group:
        term1 = 0.0
        for levelidata1j, levelidata2j in zip(levelidata1, levelidata2):
            term1 += (levelidata1j - mean1)*(levelidata2j - mean2)
        Cov_groupi = term1 # covariance for this group
        Covw += Cov_groupi
    totallength = len(variable1)
    Covw /= totallength - numlevels
    return Covw



def calcWithinGroupsCovariance(variable1, variable2, groupvariable):
    # find out how many values the group variables can take
    unique_groups = groupvariable.unique()
    num_unique_groups = len(unique_groups)
    cov_w = 0.0
    
    # get the covarance of variable 1 and variable 2 for each groups
    for i in unique_groups:
        group_i_var1 = variable1[groupvariable == i]
        group_i_var2 = variable2[groupvariable == i]
        mean_var1 = np.mean(group_i_var1) #for each group in var1
        mean_var2 = np.mean(group_i_var2) #for each group in var2
        len_group_i = len(group_i_var1)
        # get the covariance for this group
        cov_j = 0.0
        for q,k in zip(group_i_var1, group_i_var2):
            cov_j += (q - mean_var1)*(k - mean_var2)
        cov_group_i = cov_j 
        cov_w += cov_group_i
    totallength = len(variable1)
    cov_w = cov_w / (totallength - num_unique_groups)
    return cov_w


def calcBetweenGroupsCovariance(variable1, variable2, groupvariable):
    # find out how many values the group variable can take
    levels = sorted(set(groupvariable))
    numlevels = len(levels)
    # calculate the grand means
    variable1mean = np.mean(variable1)
    variable2mean = np.mean(variable2)
    # calculate the between-groups covariance
    Covb = 0.0
    for leveli in levels:
        levelidata1 = variable1[groupvariable==leveli]
        levelidata2 = variable2[groupvariable==leveli]
        mean1 = np.mean(levelidata1)
        mean2 = np.mean(levelidata2)
        levelilength = len(levelidata1)
        term1 = (mean1 - variable1mean) * (mean2 - variable2mean) * levelilength
        Covb += term1
    Covb /= numlevels - 1
    return Covb


variable1 = X.V8
variable2 = X.V11
groupvariable = y
def calcBetweenGroupsCovariance(variable1, variable2, groupvariable):
    # find out how many values the goup variable can take
    unique_groups = groupvariable.unique()
    num_unique_groups = len(unique_groups)
    # calculate the grand means
    var1_Gmean = np.mean(variable1)
    var2_Gmean = np.mean(variable2)
    # calculate the between-groups covariance
    cov_b = 0.0
    for i in unique_groups:
        group_i_var1 = variable1[groupvariable == i]
        group_i_var2 = variable2[groupvariable == i]
        mean_var1 = np.mean(group_i_var1)
        mean_var2 = np.mean(group_i_var2)
        len_group_i = len(group_i_var1)
        cov_i = (mean_var1 - var1_Gmean) * (mean_var2 - var2_Gmean) * len_group_i
        cov_b += cov_i
    cov_b = cov_b / (num_unique_groups - 1)
    return Covb


#######################
## OTHER THAN MANOVA ##
#######################
    
## Calculating Correlations for Multivariate Data
corr = stats.pearsonr(X.V2, X.V3)
print("p-value:\t", corr[1])
print("cor:\t\t", corr[0])

corrmat = X.corr()

sns.heatmap(corrmat, vmax = 1., square = False).xaxis.tick_top()

# adapted from http://matplotlib.org/examples/specialty_plots/hinton_demo.html
def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

    ax.patch.set_facecolor('lightgray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'red' if w > 0 else 'blue'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    nticks = matrix.shape[0]
    ax.xaxis.tick_top()
    ax.set_xticks(range(nticks))
    ax.set_xticklabels(list(matrix.columns), rotation=90)
    ax.set_yticks(range(nticks))
    ax.set_yticklabels(matrix.columns)
    ax.grid(False)

    ax.autoscale_view()
    ax.invert_yaxis()

hinton(corrmat)

def mosthighlycorrelated(mydataframe, numtoreport):
    # find the correlations
    cormatrix = mydataframe.corr()
    # set the correlations on the diagonal or lower triangle to zero,
    # so they will not be reported as the highest ones:
    cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T
    # find the top n correlations
    cormatrix = cormatrix.stack()
    cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index()
    # assign human-friendly names
    cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"]
    return cormatrix.head(numtoreport)

mosthighlycorrelated(X, 10)

# Standardising Variables

standardisedX = scale(X)
standardisedX = pd.DataFrame(standardisedX, index=X.index, columns=X.columns)

standardisedX.apply(np.mean)
standardisedX.apply(np.std)

# PCA
pca = PCA().fit(standardisedX)

def pca_summary(pca, standardised_data, out=True):
    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    a = list(np.std(pca.transform(standardised_data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
    summary = pd.DataFrame(zip(a, b, c), index=names, columns=columns)
    if out:
        print("Importance of components:")
        display(summary)
    return summary

summary = pca_summary(pca, standardisedX)

summary.sdev

#Total variance explained by the components is the sum of the variances of the components
np.sum(summary.sdev**2)

## Deciding how many principal components to retain
def screeplot(pca, standardised_values):
    y = np.std(pca.transform(standardised_values), axis=0)**2
    x = np.arange(len(y)) + 1
    plt.plot(x, y, "o-")
    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
    plt.ylabel("Variance")
    plt.show()

screeplot(pca, standardisedX)

summary.sdev**2

"""
Kaiser’s criterion: that we should only retain principal components for which 
the variance is above 1 (when principal component analysis was applied to 
standardised data)
"""

summary.cumprop

## Loadings for principal components


pca.components_[0] # note that loadings are for standardized versions Z1..Zk of V1..Vk

np.sum(pca.components_[0]**2) # loadings for a PC so 1.


def calcpc(variables, loadings):
    # find the number of samples in the data set and the number of variables
    numsamples, numvariables = variables.shape
    # make a vector to store the component
    pc = np.zeros(numsamples)
    # calculate the value of the component for each sample
    for i in range(numsamples):
        valuei = 0
        for j in range(numvariables):
            valueij = variables.iloc[i, j]
            loadingj = loadings[j]
            valuei = valuei + (valueij * loadingj)
        pc[i] = valuei
    return pc

calcpc(standardisedX, pca.components_[0])

plt.plot(calcpc(standardisedX, pca.components_[0]))

pca.transform(standardisedX)[:, 0]

plt.plot(pca.transform(standardisedX)[:, 0])

# Scatterplots of the Principal Components

def pca_scatter(pca, standardised_values, classifs):
    foo = pca.transform(standardised_values)
    bar = pd.DataFrame(zip(foo[:, 0], foo[:, 1], classifs), columns=["PC1", "PC2", "Class"])
    sns.lmplot("PC1", "PC2", bar, hue="Class", fit_reg=False)

pca_scatter(pca, standardisedX, y)

printMeanAndSdByGroup(standardisedX, y);

## Linear Discriminant Analysis

lda = LinearDiscriminantAnalysis().fit(X, y)

def pretty_scalings(lda, X, out=False):
    ret = pd.DataFrame(lda.scalings_, index=X.columns, columns=["LD"+str(i+1) for i in range(lda.scalings_.shape[1])])
    if out:
        print("Coefficients of linear discriminants:")
        display(ret)
    return ret

pretty_scalings_ = pretty_scalings(lda, X, out=True)

lda.scalings_[:, 0]

pretty_scalings_.LD1

def calclda(variables, loadings):
    # find the number of samples in the data set and the number of variables
    numsamples, numvariables = variables.shape
    # make a vector to store the discriminant function
    ld = np.zeros(numsamples)
    # calculate the value of the discriminant function for each sample
    for i in range(numsamples):
        valuei = 0
        for j in range(numvariables):
            valueij = variables.iloc[i, j]
            loadingj = loadings[j]
            valuei = valuei + (valueij * loadingj)
        ld[i] = valuei
    # standardise the discriminant function so that its mean value is 0:
    ld = scale(ld, with_std=False)
    return ld

# calculate LDA for each sample with our LDA1 loading
calclda(X, lda.scalings_[:, 0])

plt.plot(calclda(X, lda.scalings_[:, 0]))

# Try either, they produce the same result, use help() for more info
# lda.transform(X)[:, 0]
lda.fit_transform(X, y)[:, 0]

def groupStandardise(variables, groupvariable):
    # find the number of samples in the data set and the number of variables
    numsamples, numvariables = variables.shape
    # find the variable names
    variablenames = variables.columns
    # calculate the group-standardised version of each variable
    variables_new = pd.DataFrame()
    for i in range(numvariables):
        variable_name = variablenames[i]
        variablei = variables[variable_name]
        variablei_Vw = calcWithinGroupsVariance(variablei, groupvariable)
        variablei_mean = np.mean(variablei)
        variablei_new = (variablei - variablei_mean)/(np.sqrt(variablei_Vw))
        variables_new[variable_name] = variablei_new
    return variables_new

groupstandardisedX = groupStandardise(X, y)

lda2 = LinearDiscriminantAnalysis().fit(groupstandardisedX, y)
pretty_scalings(lda2, groupstandardisedX)

lda.fit_transform(X, y)[:, 0] 
lda2.fit_transform(groupstandardisedX, y)[:, 0]
#Actual values for standardized and non-standardized are the same
plt.plot(lda.fit_transform(X, y)[:, 0] )
plt.plot(lda2.fit_transform(groupstandardisedX, y)[:, 0])

## Separation Achieved by the Discriminant Functions
def rpredict(lda, X, y, out=False):
    ret = {"class": lda.predict(X),
           "posterior": pd.DataFrame(lda.predict_proba(X), columns=lda.classes_)}
    ret["x"] = pd.DataFrame(lda.fit_transform(X, y))
    ret["x"].columns = ["LD"+str(i+1) for i in range(ret["x"].shape[1])]
    if out:
        print("class")
        print(ret["class"])
        print()
        print("posterior")
        print(ret["posterior"])
        print()
        print("x")
        print(ret["x"])
    return ret

lda_values = rpredict(lda, standardisedX, y, True)

calcSeparations(lda_values["x"], y)


def proportion_of_trace(lda):
    ret = pd.DataFrame([round(i, 4) for i in lda.explained_variance_ratio_ if round(i, 4) > 0], columns=["ExplainedVariance"])
    ret.index = ["LD"+str(i+1) for i in range(ret.shape[0])]
    ret = ret.transpose()
    print("Proportion of trace:")
    print(ret.to_string(index=False))
    return ret

proportion_of_trace(LinearDiscriminantAnalysis(solver="eigen").fit(X, y));


## Stacked Histogram of the LDA Values

def ldahist(data, g, sep=False):
    xmin = np.trunc(np.min(data)) - 1
    xmax = np.trunc(np.max(data)) + 1
    ncol = len(set(g))
    binwidth = 0.5
    bins=np.arange(xmin, xmax + binwidth, binwidth)
    if sep:
        fig, axl = plt.subplots(ncol, 1, sharey=True, sharex=True)
    else:
        fig, axl = plt.subplots(1, 1, sharey=True, sharex=True)
        axl = [axl]*ncol
    for ax, (group, gdata) in zip(axl, data.groupby(g)):
        sns.distplot(gdata.values, bins, ax=ax, label="group "+str(group))
        ax.set_xlim([xmin, xmax])
        if sep:
            ax.set_xlabel("group"+str(group))
        else:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
ldahist(lda_values["x"].LD1, y)
ldahist(lda_values["x"].LD2, y)

# LDA scatter
sns.lmplot("LD1", "LD2", lda_values["x"].join(y), hue="V1", fit_reg=False);


printMeanAndSdByGroup(lda_values["x"], y);




import sklearn.metrics as metrics

def lda_classify(v, levels, cutoffpoints):
    for level, cutoff in zip(reversed(levels), reversed(cutoffpoints)):
        if v > cutoff: return level
    return levels[0]
    
y_pred = lda_values["x"].iloc[:, 0].apply(lda_classify, args=(lda.classes_, [-1.751107, 2.122505],)).values
y_true = y


# from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

print(metrics.classification_report(y_true, y_pred))
cm = metrics.confusion_matrix(y_true, y_pred)
#webprint_confusion_matrix(cm, lda.classes_)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plot_confusion_matrix(cm_normalized, lda.classes_, title='Normalized confusion matrix')

