from asyncio.windows_events import NULL
import csv
from distutils.command.build import build
import os
from matplotlib.style import use
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import col
import seaborn as sns
import time
import math
import PySimpleGUI as sg
import os.path
import copy

import itertools

LARGE_NUM = 10000000000000000

BUILDING_CLASSES = ['A0','A1','A2','A3','A4','A5','A6','A7','A8','A9',
                    'B1','B2','B3','B9',
                    'C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','CM',
                    'D0','D1','D2','D3','D4','D5','D6','D7','D8','D9',
                    'E1','E2','E3','E4','E7','E9',
                    'F1','F2','F4','F5','F8','F9',
                    'G0','G1','G2','G3','G4','G5','G6','G7','G8','G9','GU','GW',
                    'HB','HH','HR','HS','H1','H2','H3','H4','H5','H6','H7','H8','H9',
                    'I1','I2','I3','I4','I5','I6','I7','I9',
                    'J1','J2','J3','J4','J5','J6','J7','J8','J9',
                    'K1','K2','K3','K4','K5','K6','K7','K8','K9',
                    'L1','L2','L3','L8','L9',
                    'M1','M2','M3','M4','M9',
                    'N1','N2','N3','N4','N9',
                    'O1','O2','O3','O4','O5','O6','O7','O8','O9',
                    'P1','P2','P3','P4','P5','P6','P7','P8','P9',
                    'Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','RA','RB','RG','RH','RK','RP',
                    'RR','RS','RT','RW','R0','R1','R2','R3','R4','R5','R6','R7','R8','R9','RR',
                    'S0','S1','S2','S3','S4','S5','S9',
                    'T1','T2','T9',
                    'U0','U1','U2','U3','U4','U5','U6','U7','U8','U9',
                    'V0','V1','V2','V3','V4','V5','V6','V7','V8','V9',
                    'W1','W2','W3','W4','W5','W6','W7','W8','W9',
                    'Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9',
                    'Z0','Z1','Z2','Z3','Z4','Z5','Z7','Z8','Z9']

BOROUGH = ['manhattan', 'bronx', 'brooklyn', 'queens', 'staten island']

class Graph:
    def boxplot(data, column, add_to_title = NULL):
        '''
        Boxplot creation function
        '''

        # Setting the size of the plot
        plt.figure(figsize=(15,6))

        column = str(column)

        # Plot the data and configure the settings
        sns.boxplot(x=column, data=data)
        plt.ticklabel_format(style='plain', axis='x')
        if(add_to_title != NULL):
            plt.title('Boxplot of ' + column + ' ' + add_to_title)
        else:
            plt.title('Boxplot of ' + column)
        plt.show()
    
    def cumulative_distribution(data, column, add_to_title = NULL):
        '''
        Cumulative Distribution creation function
        '''

        # Setting the size of the plot
        plt.figure(figsize=(15,6))

        column = str(column)

        # Get the data and format it
        x = data[[column]].sort_values(by=column).reset_index()
        x['PROPERTY PROPORTION'] = 1
        x['PROPERTY PROPORTION'] = x['PROPERTY PROPORTION'].cumsum()
        x['PROPERTY PROPORTION'] = 100* x['PROPERTY PROPORTION'] / len(x['PROPERTY PROPORTION'])

        # Plot the data and configure the settings
        plt.plot(x['PROPERTY PROPORTION'],x[column], linestyle='None', marker='o')
        
        if(add_to_title != NULL):
            plt.title('Cumulative Distribution of Properties according to ' + column + ' ' + add_to_title)
        else:
            plt.title('Cumulative Distribution of Properties according to ' + column)
        plt.xlabel('Percentage of Properties in ascending order of ' + column)
        plt.ylabel(column)
        plt.ticklabel_format(style='plain', axis='y')
        plt.show()

    def regplot(data, y_title, x_title, add_to_title = NULL):
        '''
        Regplot creation function
        '''

        # Setting the size of the plot
        plt.figure(figsize=(15,6))

        # Plot the data and configure the settings
        plt.ticklabel_format(style='plain', axis='y',useOffset=False)
        plt.ticklabel_format(style='plain', axis='x',useOffset=False)
        sns.regplot(x=x_title, y=y_title, data=data, fit_reg=False, scatter_kws={'alpha':0.3})
        if(add_to_title != NULL):
            plt.title(x_title + ' vs ' + y_title + ' ' + add_to_title)
        else:
            plt.title(x_title + ' vs ' + y_title)
        plt.show()

class Model:
    def __init__(self, data, filename):
        # This is a constructor for the Model object
        
        self.file_dir = filename

        # setting all the Object members
        self.feature_prepare(data)

        # initializing the theta vector
        self.theta = np.ones(np.size(self.features, 1))
        #debug print
        self.debug_print()
    
    def debug_print(self):
        # This is a debug function of the Model class
        print("all: \n", self.all_data)
        print("features: \n", self.features)
        print("head: \n", self.features[:50, :])
        print("y: \n", self.y)

    def feature_prepare(self, data):
        '''
        Feature Engineering Prepare function
        '''

        saved_data = data
        #removing unnecessary columns
        saved_data.drop(columns = ['BUILDING CLASS CATEGORY', 'TAX CLASS AT PRESENT', 'EASE-MENT', 'BUILDING CLASS AT PRESENT', 'ADDRESS', 'APARTMENT NUMBER'], inplace=True)
        #cleaning
        if not os.path.isfile(self.file_dir[0:-5] + '_new.csv'):
            saved_data = self.clean(saved_data)
        else:
            saved_data = pd.read_csv (self.file_dir[0:-5] + '_new.csv')
            saved_data = saved_data.iloc[: , 1:]
        
        print(sum(data.duplicated(data.columns)))

        self.data = saved_data
        saved_data = saved_data.astype('float64')
        self.all_data = saved_data.to_numpy()
        self.features = np.c_[np.ones(np.size(self.all_data, 0)), self.all_data[:, 0:2], self.all_data[:, 3:-1]] #adding X0
        self.y = self.all_data[:, 2]

    def feature_prepare_old(self, data):
        '''
        Old Feature Prepare function
        '''

        saved_data = data
        #removing unnecessary columns
        saved_data.drop(columns = ['BUILDING CLASS CATEGORY', 'TAX CLASS AT PRESENT', 'EASE-MENT', 'BUILDING CLASS AT PRESENT', 'ADDRESS', 'APARTMENT NUMBER'], inplace=True)
        #cleaning
        if not os.path.isfile(r'C:\Users\Dan\Desktop\ML-NYC\new-data.csv'):
            saved_data = self.clean(saved_data)
        else:
            saved_data = pd.read_csv (r'C:\Users\Dan\Desktop\ML-NYC\new-data.csv')
            saved_data = saved_data.iloc[: , 1:]
        
        print(sum(data.duplicated(data.columns)))

        #change SALE DATE to the months
        for row in range(np.size(saved_data, 0)):
            print(saved_data.at[row, 'SALE DATE'])
            value = int(str(saved_data.at[row, 'SALE DATE'])[5:7])
            saved_data.at[row, 'SALE DATE'] = value
            print(row) #for debug

        #neighborhood dictionary and building classes initialization
        self.neighborhood = {}
        self.building_classes = BUILDING_CLASSES

        #create dictionary for NEIGHBORHOOD
        for row in range(np.size(saved_data, 0)):
            key_n = saved_data.at[row, 'NEIGHBORHOOD']
            self.neighborhood[key_n] = 0
            print(row) #for debug

        #create dictionary for BUILDING CLASS AT TIME OF SALE
        self.building_classes = dict.fromkeys(self.building_classes)

        #create values for NEIGHBORHOOD dictionary
        counter = 1
        for key in self.neighborhood:
            self.neighborhood[key] = counter
            counter+=1

        #create values for BUILDING CLASS AT TIME OF SALE dictionary
        counter = 1
        for key in self.building_classes:
            self.building_classes[key] = counter
            counter+=1

        #change NEIGHBORHOOD and BUILDING CLASS AT TIME OF SALE to the values
        for row in range(np.size(saved_data, 0)):
            saved_data.at[row, 'NEIGHBORHOOD'] = self.neighborhood[saved_data.at[row, 'NEIGHBORHOOD']]
            saved_data.at[row, 'BUILDING CLASS AT TIME OF SALE'] = self.building_classes[saved_data.at[row, 'BUILDING CLASS AT TIME OF SALE']]
            print(row) #for debug

        print(len(self.building_classes))
        saved_data = saved_data.astype('float64')
        self.all_data = saved_data.to_numpy()
        self.features = np.c_[np.ones(np.size(self.all_data, 0)), self.all_data[:, 0:2], self.all_data[:, 3:-1]] #adding X0
        self.y = self.all_data[:, 2]

    def clean(self, data):
        '''
        Data Cleaning function
        '''

        # set the right types for the features
        data['SALE PRICE'] = pd.to_numeric(data['SALE PRICE'], errors='coerce')
        data['LAND SQUARE FEET'] = pd.to_numeric(data['LAND SQUARE FEET'], errors='coerce')
        data['GROSS SQUARE FEET']= pd.to_numeric(data['GROSS SQUARE FEET'], errors='coerce')
        data['SALE DATE'] = pd.to_datetime(data['SALE DATE'], errors='coerce')
        data['TAX CLASS AT TIME OF SALE'] = data['TAX CLASS AT TIME OF SALE'].astype('category')
        
        # seeing the data before cleaning
        data_info = data.describe()
        data_info.to_csv(self.file_dir[0:-5] + '_data-info.csv')
        print(data_info)

        #setting the columns that need to be checked
        check_col = ['TOTAL UNITS', 'LAND SQUARE FEET','GROSS SQUARE FEET', 'SALE PRICE']

        for col in data.columns:
            data = data[data[col].notnull()]
        
        data = data[data['SALE PRICE'] > 1000]

        for col in check_col:
            data = data[data[col] > 0]

        #removing numbering columns
        del data['Unnamed: 0']

        #checking for duplicates and removing them
        data = self.check_duplicates(data)

        #For BoxPlot
        Graph.boxplot(data, 'SALE PRICE')

        #For Cumulative Distribution
        Graph.cumulative_distribution(data, 'SALE PRICE')

        data = data[(data['SALE PRICE'] > 400000) & (data['SALE PRICE'] < 6000000)]

        #For BoxPlot
        Graph.boxplot(data, 'SALE PRICE', '(After)')

        #For Cumulative Distribution
        Graph.cumulative_distribution(data, 'SALE PRICE', '(After)')

        #For BoxPlot
        Graph.boxplot(data, 'GROSS SQUARE FEET')

        #For BoxPlot
        Graph.boxplot(data, 'LAND SQUARE FEET')
        
        data = data[(data['GROSS SQUARE FEET'] < 20000) & (data['LAND SQUARE FEET'] < 20000)]
        print(len(data))

        #For BoxPlot
        Graph.boxplot(data, 'GROSS SQUARE FEET', '(After)')

        #For BoxPlot
        Graph.boxplot(data, 'LAND SQUARE FEET', '(After)')

        # limiting the TOTAL UNITS and check the sum of all the units accordingly
        data = data[(data['TOTAL UNITS'] > 0) & (data['TOTAL UNITS'] < 50)] 
        data = data[(data['TOTAL UNITS'] == data['COMMERCIAL UNITS'] + data['RESIDENTIAL UNITS'])] 
        print(len(data))

        # deleting zip codes that cannot be
        data = data[(data['ZIP CODE'] > 10000) & (data['ZIP CODE'] <= 14975)]
        print(len(data))

        # making the zipcode start from 0
        data['ZIP CODE'] = data['ZIP CODE'] - 10001

        data = data[data['YEAR BUILT'] > 0]
        print(len(data))

        # creating BUILDING AGE feature instead of YEAR BUILT
        data['BUILDING AGE'] = 2017 - data['YEAR BUILT']

        del data['YEAR BUILT']

        #For Regplot
        Graph.regplot(data, 'SALE PRICE', 'BUILDING AGE')

        #For BUILDING CLASS AT TIME OF SALE Boxplot
        plt.figure(figsize=(20,6))
        order = sorted(data['BUILDING CLASS AT TIME OF SALE'].unique())
        sns.boxplot(x='BUILDING CLASS AT TIME OF SALE', y='SALE PRICE', data=data, order=order)
        plt.xticks(rotation=90)
        plt.title('Sale Price Distribution by BUILDING CLASS')
        plt.show()

        #For TAX CLASS AT TIME OF SALE Boxplot
        plt.figure(figsize=(20,6))
        order = sorted(data['TAX CLASS AT TIME OF SALE'].unique())
        sns.boxplot(x='TAX CLASS AT TIME OF SALE', y='SALE PRICE', data=data, order=order)
        plt.xticks(rotation=90)
        plt.title('Sale Price Distribution by TAX CLASS')
        plt.show()

        #For NEIGHBORHOOD class Boxplot
        plt.figure(figsize=(30,6))
        order = sorted(data['NEIGHBORHOOD'].unique())
        sns.boxplot(x='NEIGHBORHOOD', y='SALE PRICE', data=data, order=order)
        plt.xticks(rotation=90)
        plt.title('Sale Price Distribution by NEIGHBORHOOD')
        plt.show()

        # updating the values in the BOROUGH feature to the actual boroughs
        data['BOROUGH'][data['BOROUGH'] == 1] = 'Manhattan'
        data['BOROUGH'][data['BOROUGH'] == 2] = 'Bronx'
        data['BOROUGH'][data['BOROUGH'] == 3] = 'Brooklyn'
        data['BOROUGH'][data['BOROUGH'] == 4] = 'Queens'
        data['BOROUGH'][data['BOROUGH'] == 5] = 'Staten Island'

        # Boxplot of the different boroughs
        plt.figure(figsize=(10,6))
        order = sorted(data['BOROUGH'].unique())
        sns.boxplot(x='BOROUGH', y='SALE PRICE', data=data, order=order)
        plt.xticks(rotation=90)
        plt.title('Sale Price Distribution by BOROUGH')
        plt.show()

        # limiting the price-square feet ratio
        data = data[(data['SALE PRICE'] < 1500000) & (data['GROSS SQUARE FEET'] < 5000) & (data['LAND SQUARE FEET'] < 5000)].append(data[(data['SALE PRICE'] >= 1500000)])

        # initializing bounderies for every borough
        num1,num2,num3,num4,num5 = (1500000, 3000000, 4000000, 3000000, 2000000)

        # amount below\above the bounderies for every borough
        len1 = len(data[(data['BOROUGH'] == 'Manhattan') & (data['SALE PRICE'] <= num1)])
        len2 = len(data[(data['BOROUGH'] == 'Bronx') & (data['SALE PRICE'] >= num2)])
        len3 = len(data[(data['BOROUGH'] == 'Brooklyn') & (data['SALE PRICE'] >= num3)])
        len4 = len(data[(data['BOROUGH'] == 'Queens') & (data['SALE PRICE'] >= num4)])
        len5 = len(data[(data['BOROUGH'] == 'Staten Island') & (data['SALE PRICE'] >= num5)])

        # debug
        print('''
        Manhattan below {num1}: {ans1} 
        Bronx above {num2}: {ans2}
        Brooklyn above {num3}: {ans3}
        Queens above {num4}: {ans4}
        Staten Island above {num5}: {ans5}'''.format(num1=num1, num2=num2, num3=num3,num4=num4,num5=num5,
                     ans1=len1, ans2=len2,ans3=len3,ans4=len4,ans5=len5))
        
        # data for every Borough
        manhattan_data = data[(data['BOROUGH'] == 'Manhattan') & (data['SALE PRICE'] >= num1)]
        bronx_data = data[(data['BOROUGH'] == 'Bronx') & (data['SALE PRICE'] <= num2)]
        brooklyn_data = data[(data['BOROUGH'] == 'Brooklyn') & (data['SALE PRICE'] <= num3)]
        queens_data = data[(data['BOROUGH'] == 'Queens') & (data['SALE PRICE'] <= num4)]
        staten_island_data = data[(data['BOROUGH'] == 'Staten Island') & (data['SALE PRICE'] <= num5)]

        dataframes = [manhattan_data, bronx_data, brooklyn_data, queens_data, staten_island_data]
        
        # connecting the data for every borough
        data = pd.concat(dataframes, axis=0)

        # The features that will stay
        columns = ['BOROUGH', 'BUILDING CLASS AT TIME OF SALE', 'COMMERCIAL UNITS','BUILDING AGE', 'SALE PRICE', 'GROSS SQUARE FEET', 'LAND SQUARE FEET', 'RESIDENTIAL UNITS', 'NEIGHBORHOOD', 'TAX CLASS AT TIME OF SALE', 'ZIP CODE'] #better without month or date - write about this
        data_model = data.loc[:,columns]

        
        # The Categorical features that need to be seperate 
        one_hot_features = ['BOROUGH', 'TAX CLASS AT TIME OF SALE', 'NEIGHBORHOOD', 'BUILDING CLASS AT TIME OF SALE']

        # For each categorical column, find the unique number of categories. This tells us how many columns we are adding to the dataset.
        longest_str = max(one_hot_features, key=len)
        total_num_unique_categorical = 0
        for feature in one_hot_features:
            num_unique = len(data[feature].unique())
            print('{col:<{fill_col}} : {num:d} unique categorical values.'.format(col=feature, 
                                                                                fill_col=len(longest_str),
                                                                                num=num_unique))
            total_num_unique_categorical += num_unique
        print('{total:d} columns will be added during one-hot encoding.'.format(total=total_num_unique_categorical))

        one_hot_encoded = pd.get_dummies(data_model[one_hot_features])

        one_hot_encoded.info(verbose=True, memory_usage=True, null_counts=True)

        # Delete the old columns
        data_model = data_model.drop(one_hot_features, axis=1)

        # Add the new one-hot encoded variables
        data_model = pd.concat([data_model, one_hot_encoded], axis=1)
        data_model.head()
        data_model['SALE PRICE'] = data['SALE PRICE']

        # Saving the changes
        data = data_model

        #seeing the data
        data_info = data.describe()
        data_info.to_csv(self.file_dir[0:-5] + '_data-info-after.csv')
        print(data_info)

        #saving new data
        data.to_csv(self.file_dir[0:-5] + '_new.csv')
        return data

    def normal_equation(self):
        '''
        This function is a Normal Equation Algorithm
        It searches the optimal theta vector
        '''
        X_transpose = self.features.T
        _x = X_transpose.dot(self.features)
        # print(np.shape(_x))
        self.theta = np.linalg.pinv(_x).dot(X_transpose).dot(self.y)
        return self.theta

    def gradient_descent(self, learning_rate, iter_num = NULL,  theta = NULL, scale_norm = False):
        '''
        This function is a Gradient Descent Algorithm
        It searches the optimal theta vector
        '''
        # initializing theta
        if theta == NULL:
            theta = np.ones(np.size(self.features, 1))

        # initializing cost history DataFrame
        cost_data = pd.DataFrame({'Cost' : [], 'Iteretion' : []})

        # saving starting time
        start_time = time.time()

        # checking if Scaling and Normalization is needed - if yes, runs scale_norm()
        if scale_norm:
            data = self.scale_norm(self.features)
        else:
            data = self.features

        # initializing starting cost and last cost and m - amount of training examples
        last_cost = self.cost_function(data, theta) + 1
        cost = self.cost_function(data, theta)
        m = np.size(data, 0)
        
        # looping 
        if iter_num != NULL:        
            for iter in range(1,iter_num + 1):
                print("Iteretion: ", iter, " Cost: ", self.cost_function(data, theta))
                
                # hypothesis calculation
                h = self.hypothesis(data, theta)
                theta = theta -( learning_rate/m * (h - self.y).T.dot(data))
                
                # entering current cost to cost history  
                cost = self.cost_function(data, theta)
                a_row = {'Cost' : cost, 'Iteration' : iter}
                cost_data = cost_data.append(a_row, ignore_index = True)

        else:
            # count - iteration variable
            count = 1

            while last_cost - cost > 0.00001:
                print("Iteretion: ", count, " Cost: ", self.cost_function(data, theta))
                
                # updating last cost
                last_cost = cost

                # hypothesis calculation
                h = self.hypothesis(data, theta)
                theta = theta -( learning_rate/m * (h - self.y).T.dot(data))
                
                # entering current cost to cost history  
                cost = self.cost_function(data, theta)
                a_row = {'Cost' : cost, 'Iteration' : count}
                cost_data = cost_data.append(a_row, ignore_index = True)
                count += 1

        self.theta = theta

        Graph.regplot(cost_data,'Cost', 'Iteration')
        cost_data.to_csv(self.file_dir[0:-5] + "_gradient-descent-data.csv")

        # saving ending time
        end_time = time.time()

        result = 'Gradient Descent\nTime spended : ' + str(int(end_time - start_time)) + " s"
        print(result)
        return (theta, result)

    def cost_function(self, x, theta):
        '''
        This function is an implemintation of Cost Function
        It return the cost of the model according to the data set
        '''
        m = np.size(self.features, 0)
        h = x.dot(theta)
        cost = np.sum(np.power(h-self.y, 2))/(2*m)
        return cost

    def scale_norm(self, data):
        '''
        This function is an implemintation of Feature Scaling
        and Mean Normalization.
        It return the data set after those changes
        '''
        new_data = data

        # going through all the columns
        for col in range(np.size(data, 1)):
            max = np.amax(data[:, col])
            min = np.amin(data[:, col])
            av = np.average(data[:, col])
            
            # debug
            print('''
            For column #{num}:
            \tmax value : {max}
            \tmin value : {min}
            \taverage value : {av}
            '''.format(num = col, max = max, min = min, av = av))

            scale = max - min
            new_data[:][col] -= av
            new_data[:][col] /= scale

        return new_data

    def hypothesis(self, x, theta):
        '''
        This function is an implementation of regression hypothesis
        It returns the hypothesis value for specific training example.
        or the hypothesis vector for number of training examples
        '''
        h = x.dot(theta)
        return h

    def error(self, x, y, theta):
        '''
        This function returns the squared error for specific
        training example.
        '''
        hypothesis = self.hypothesis(x, theta)
        err = pow(hypothesis - y , 2)
        return err

    def average_error(self, theta):
        '''
        This function returns the average error of the model.
        In percentage and in price
        '''
        error_data = pd.DataFrame({'ERROR %' : [], 'PRICE' : []})

        #initializing average error
        av_error_percent = 0
        av_price_error = 0

        # amount of training examples
        m = np.size(self.features, 0)

        #going through all the training sets
        for row in range(m):
            # x vector
            x = self.features[row]
            error_size = self.error(x, self.y[row], theta)
            error_sqrt = math.sqrt(error_size)
            av_price_error += error_sqrt

            av_error_percent += error_sqrt / self.y[row]

            a_row = {'ERROR %' : error_sqrt / self.y[row] * 100, 'PRICE' : self.y[row]}
            error_data = error_data.append(a_row, ignore_index = True)

        av_error_percent /= m
        av_price_error /= m

        print(error_data)
        result = 'average error percentage : {num}%\n'.format(num=av_error_percent * 100)
        result += 'average error in price : {num}$'.format(num=av_price_error)
        print(result)

        Graph.regplot(error_data,'ERROR %', 'PRICE')
        error_data.to_csv(self.file_dir[0:-5] + '_error-data.csv')
        return (result, av_error_percent, av_price_error)
    
    def run_check(self, theta):
        # dictionary to save the amount of encounters per percentage
        dict_below = {}

        # list of the percentage numbers
        below_percent_lst = [50, 40, 30, 20, 10, 5]

        # initializing dictionary
        for e in below_percent_lst:
            dict_below[e] = 0

        # amount of training examples
        m = np.size(self.features, 0)

        #going through all the training sets
        for row in range(m):
            # x vector
            x = self.features[row]
            error_size = self.error(x, self.y[row], theta)

            h = self.hypothesis(x, theta)
            #going through all the percentages
            for percent in below_percent_lst:
                if error_size < pow(percent/100 * h, 2):
                    dict_below[percent] += 1
        
        # print all the encounters per percentage
        result = ''

        for key in dict_below:
            result += 'difference below {col}% : {num} from {amount} training example, ({global_percent}%).\n'.format(col=key, num=dict_below[key], amount = m, global_percent = dict_below[key]/m * 100)

        print(result)
        return result
    
    def check_duplicates(self, data):
        # method for checking for duplicates and removing them
        dup_num = sum(data.duplicated(data.columns))
        print("Duplicates before:", dup_num)

        # delete duplicates if there are duplicates
        if dup_num != 0:
            data = data.drop_duplicates(data.columns, keep='last')
            print("Duplicates after:", sum(data.duplicated(data.columns)))

        return data

class GUI:
    def __init__(self):
        # The Main Menu window layout columns

        # First Column - training methods
        control_column = [
            [sg.Button("Normal Equation", key = "-NORMAL-")],
            [sg.Button("Gradient Descent", key = "-GD-")],
            [sg.Text("Enter Learning rate:")],
            [sg.In(size=(10, 1), enable_events=True, key="-LEARNING RATE-")],
            [sg.Text("Enter Iteration number:")],
            [sg.In(size=(10, 1), enable_events=True, key="-ITERATION-")],
            [sg.Button("Show theta", key = "-SHOW THETA-")],
            [sg.Button("Reset theta", key = "-RESET THETA-")],
            [sg.Text("")],
            [sg.Button("Clear", key = "-CLEAR-")],
            [sg.Button("Predict", key = "-PREDICT-", visible=False)],
        ]

        # Seconds Column - Model Checking methods and Result
        check_column = [
            [sg.Button("Run Check", key = "-RUNCHECK-")],
            [sg.Button("Average Error", key = "-AVERROR-")],
            [sg.Button("Model Selection Algorithm", key = "-MSA-")],
            [sg.Text("Result:")],
            [sg.Text("", key = "-RESULT-")],
        ]
        
        # ----- Main Menu Layout -----
        self.main_layout = [
            [
                sg.Column(control_column),
                sg.VSeperator(),
                sg.Column(check_column),
            ]
        ]


        # The Start Menu window layout columns
        # First Column - choosing folder and file viewer
        file_list_column = [
            [
                sg.Text("Data Folder"),
                sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
                sg.FolderBrowse(),
            ],
            [
                sg.Listbox(
                    values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
                )
            ],
        ]

        # Second Column - Text about the file and confirmation
        data_viewer_column = [
            [sg.Text("Choose a data from list on left:")],
            [sg.Text(size=(40, 1), key="-TOUT-")],
            [sg.Button("Confirm", visible=False, key="-CONFIRM-")],
        ]

        # ----- Full layout -----
        self.layout = [
            [
                sg.Column(file_list_column),
                sg.VSeperator(),
                sg.Column(data_viewer_column),
            ]
        ]

        # The check window layout columns
        # First Column - Titles for inputs
        enter_data_text_column = [
            [sg.Text("Enter the following data:")],
            [sg.Text("Enter Borough:")],
            [sg.Text("Enter Building Class:")],
            [sg.Text("Enter Tax Class:")],
            [sg.Text("Enter Neighborhood:")],
            [sg.Text("Enter year build:")],
            [sg.Text("Enter Zip Code:")],
            # No need in TOTAL UNITS because it should be the sum of commercial and residential
            [sg.Text("Enter Commercial Units:")],
            [sg.Text("Enter Residential Units:")],
            [sg.Text("Enter Gross Square feet:")],
            [sg.Text("Enter Land square feet:")],
            [sg.Button("Predict", key = "-CHECK-")]
        ]

        # Second Column - The Input Boxes
        enter_data_column = [
            [sg.In(size=(15, 1), enable_events=True, key="-BOROUGH-")],
            [sg.In(size=(15, 1), enable_events=True, key="-BUILDING-")],
            [sg.In(size=(15, 1), enable_events=True, key="-TAX-")],
            [sg.In(size=(15, 1), enable_events=True, key="-NEIGHBORHOOD-")],
            [sg.In(size=(15, 1), enable_events=True, key="-YEAR-")],
            [sg.In(size=(15, 1), enable_events=True, key="-ZIPCODE-")],
            [sg.In(size=(15, 1), enable_events=True, key="-COMMERCIAL-")],
            [sg.In(size=(15, 1), enable_events=True, key="-RESIDENTIAL-")],
            [sg.In(size=(15, 1), enable_events=True, key="-GROSS-")],
            [sg.In(size=(15, 1), enable_events=True, key="-LAND-")]
        ]

        # Third Column - The results
        result_column = [
            [sg.Text("Result:")],
            [sg.Text("", key= "-PRESULT-")]
        ]

        # ----- Full layout -----
        self.data_enter_layout = [
            [
                sg.Column(enter_data_text_column),
                sg.Column(enter_data_column),
                sg.VSeparator(),
                sg.Column(result_column),
            ]
        ]
    
    def start_window(self):
        window = sg.Window("Machine Learning NYC Property Model", self.layout)

        data_filename = ""
        self.model = NULL

        # Run the Event Loop
        while True:
            event, values = window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                break

            # Folder name was filled in, make a list of files in the folder
            if event == "-FOLDER-":
                folder = values["-FOLDER-"]
                try:
                    # Get list of files in folder
                    file_list = os.listdir(folder)
                except:
                    file_list = []

                fnames = [
                    f
                    for f in file_list
                    if os.path.isfile(os.path.join(folder, f))
                    and f.lower().endswith(".csv")
                ]
                window["-FILE LIST-"].update(fnames)

            elif event == "-FILE LIST-":  # A file was chosen from the listbox
                try:
                    filename = os.path.join(
                        values["-FOLDER-"], values["-FILE LIST-"][0]
                    )
                    window["-TOUT-"].update("Selected " + filename)
                    data_filename = filename
                    window["-CONFIRM-"].update(visible=True)
                except:
                    pass

            elif event == "-CONFIRM-":
                df = pd.read_csv (data_filename)
                self.model = Model(df, data_filename)
                break

        window.close()

    def main_menu(self):
        window = sg.Window("Main Menu", self.main_layout)
        
        iter = NULL
        learning_rate = 0.0000001

        # Run the Event Loop
        while True:
            event, values = window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                break

            if event == "-NORMAL-":
                self.model.normal_equation()
                window["-PREDICT-"].update(visible = True)

            elif event == "-GD-":
                print(learning_rate)
                print(iter)
                if type(learning_rate) == float and type(iter) == int or iter == NULL: 
                    window["-RESULT-"].update(self.model.gradient_descent(learning_rate,iter_num=iter)[1])
                    window["-PREDICT-"].update(visible = True)
                else:
                    sg.popup("Iteration number and Learning rate should contain only numbers and dot!")

            elif event == "-MSA-":
                window["-RESULT-"].update(model_selection(self.model))

            elif event == "-LEARNING RATE-":
                learning_rate = values["-LEARNING RATE-"]
                if learning_rate == '':
                    learning_rate = 0.0000001
                    continue
                try:
                    learning_rate = float(learning_rate)
                except:
                    pass

            elif event == "-ITERATION-":
                iter = values["-ITERATION-"]
                if iter == '':
                    iter = NULL
                    continue
                try:
                    iter = int(iter)
                    if iter <= 0:
                        iter = NULL
                except:
                    pass

            elif event == "-RESET THETA-":
                self.model.theta = np.ones(np.size(self.model.features, 1))
                window["-PREDICT-"].update(visible = False)

            elif event == "-RUNCHECK-":
                window["-RESULT-"].update(self.model.run_check(self.model.theta))

            elif event == "-AVERROR-":
                window["-RESULT-"].update(self.model.average_error(self.model.theta)[0])
            
            elif event == "-SHOW THETA-":
                window["-RESULT-"].update("Theta:\n" + str(self.model.theta))

            elif event == "-CLEAR-":
                window["-RESULT-"].update("")

            elif event == "-PREDICT-":
                self.data_enter_window()

        window.close()

    def data_enter_window(self):
        layout = copy.deepcopy(self.data_enter_layout)
        window = sg.Window("Machine Learning NYC Property Model", layout)

        sg.Popup("Be sure you trained the model as you wished before you check here the prediction!")
        # Run the Event Loop
        while True:
            event, values = window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                break

            # 
            if event == "-CHECK-":
                
                (borough, building, neighborhood, tax, c_units, r_units, g_feet, l_feet, zipcode, year) = (values["-BOROUGH-"],
                                                                                                 values["-BUILDING-"],
                                                                                                 values["-NEIGHBORHOOD-"],
                                                                                                 values["-TAX-"],
                                                                                                 values["-COMMERCIAL-"],
                                                                                                 values["-RESIDENTIAL-"],
                                                                                                 values["-GROSS-"],
                                                                                                 values["-LAND-"],
                                                                                                 values["-ZIPCODE-"],
                                                                                                 values["-YEAR-"])

                containOther = False
                for c in [tax, c_units, r_units, l_feet, g_feet, zipcode, year]:
                    if not str(c).isdecimal():
                        sg.Popup("Tax Class, Commercial Units, Residential Units, Zipcode, Year, Land Square feet and Gross Square feet should only contain numbers!")
                        containOther = True

                if containOther:
                    continue

                tax = int(tax)
                c_units = int(c_units)
                r_units = int(r_units)
                l_feet = int(l_feet)
                g_feet = int(g_feet)
                zipcode = int(zipcode)
                year = int(year)
                
                if str(borough).lower() not in BOROUGH:
                    sg.Popup("Enter Valid Borough!")
                    continue
                
                if str(borough).lower() not in BOROUGH:
                    sg.Popup("Enter Valid Borough!")
                    continue

                if str(building).upper() not in BUILDING_CLASSES:
                    sg.Popup("Enter Valid Building Class!")
                    continue

                print("TAX: ", tax)
                if tax not in range(1,5):
                    sg.Popup("Enter Valid Tax Class (1-4)!")
                    continue

                if year <= 0:
                    sg.Popup("Enter Valid Year (Above 0)!")
                    continue

                year = 2017 - year

                if zipcode <= 10000 or zipcode > 14975:
                    sg.Popup("Enter Valid NYC Zipcode (10001-14975)!")
                    continue

                zipcode -= 10001

                #if g_feet >= 20000 or l_feet >= 20000:
                #    sg.Popup("Enter Valid Gross and Land Square feet (<20000)!")
                #    continue

                use_data = self.model.data

                neigh_feature = "NEIGHBORHOOD_" + neighborhood.upper()
                b_class_feature = "BUILDING CLASS AT TIME OF SALE_" + building.upper()
                tax_class_feature = "TAX CLASS AT TIME OF SALE_" + str(tax)

                one_hot_features = list(use_data)[7:-1]

                if neigh_feature not in one_hot_features:
                    sg.Popup("Sorry, There is no enough data for such neighborhood!")
                    continue

                if b_class_feature not in one_hot_features:
                    sg.Popup("Sorry, There is no enough data for such building class!")
                    continue
                
                if tax_class_feature not in one_hot_features:
                    sg.Popup("Sorry, There is no enough data for such tax class!")
                    continue

                use_data = use_data.astype('float64')
                for col in use_data.columns:
                    use_data[col].values[:] = 0
                use_data["BOROUGH_" + borough] = 1
                use_data[neigh_feature] = 1
                use_data[tax_class_feature] = 1
                use_data[b_class_feature] = 1
                use_data['RESIDENTIAL UNITS'] = r_units
                use_data['COMMERCIAL UNITS'] = c_units
                use_data['ZIP CODE'] = zipcode
                use_data['BUILDING AGE'] = year
                use_data['GROSS SQUARE FEET'] = g_feet
                use_data['LAND SQUARE FEET'] = l_feet
                use_data = use_data.to_numpy()
                example = np.c_[np.ones(np.size(use_data, 0)), use_data[:, 0:2], use_data[:, 3:-1]] #adding X0
                example = example[0][:]
                
                hypothesis = self.model.hypothesis(example, self.model.theta)
                window["-PRESULT-"].update("Prediction: {h}$\n{b}, {b_c}, {tax}, {n}, {c}, {r}, {g}, {l}, {z}, {y}".format(h=hypothesis, b=borough, b_c = building, tax=tax, c = c_units, r = r_units, g = g_feet, l= l_feet, z = zipcode + 10001, y = 2017 - year, n = neighborhood))

        window.close()    

def model_selection(model):
    '''
    This function is a Model Selection Algorithm
    It searches the best feature combination
    '''
    start_time = time.time()

    every = np.c_[model.features, model.y]

    every_save = every
    theta_save = model.theta

    m_feat = model.features
    m_y = model.y

    # shuffle
    np.random.shuffle(every)

    m_feat = every[:, 0:np.size(m_feat, 1)]
    m_y = every[:, -1]

    # amount of training examples
    m = np.size(m_feat, 0)

    print(np.shape(m_feat))

    training_size = int(0.6 * m)
    csv_size = int(0.2 * m)
    test_size = m - training_size - csv_size

    # training set
    training_set = m_feat[0:training_size]
    training_y = m_y[0:training_size]

    # cross validation set
    csv_set = m_feat[training_size:training_size+csv_size]
    csv_y = m_y[training_size:training_size+csv_size]

    # test set
    test_set = m_feat[training_size + csv_size:m]
    test_y = m_y[training_size + csv_size:m]

    # amount of features
    x_num = np.size(model.features, 1)

    min_cost = LARGE_NUM
    min_theta = np.ones(x_num)
    min_comb = []

    possible_col = range(7)
    always_col = list(range(7, x_num))

    # creating a list of all combinations
    all_combinations = []
    for r in range(len(possible_col) + 1):
        combinations_object = itertools.combinations(possible_col, r)
        combinations_list = list(combinations_object)
        all_combinations += combinations_list

    # checking every features combination
    for comb in all_combinations:
        model.features = training_set[:, list(comb) + always_col]
        model.y = training_y
        np.set_printoptions(suppress=True)
        
        # finding theta
        theta = model.normal_equation()
        
        model.y = csv_y
        cost = model.cost_function(csv_set[:, list(comb) + always_col], theta)

        print(comb, "\ncost: ", cost, "\n\nmin comb:", min_comb, "\nmin cost: ", min_cost)

        # updating if better combination
        if cost < min_cost:
            min_cost = cost
            min_theta = theta
            min_comb = comb

    end_time = time.time()

    model.y = test_y

    result = '''Min comb : {comb}
             Min cost (Cross-Validation Set): {cost}
             Real cost (Test Set) : {r_cost}

             Time spended : {time}
            '''.format(comb = min_comb, cost = min_cost, r_cost = model.cost_function(test_set[:, list(min_comb) + always_col], min_theta), time = end_time - start_time)
    
    model.features = every_save[:, 0:np.size(m_feat, 1)]
    model.y = every_save[:, -1]
    model.theta = theta_save

    print(result)
    return result

def main():
    ui = GUI()
    
    ui.start_window()

    if ui.model == NULL:
        exit()
    
    ui.main_menu()

if __name__ == "__main__":
    main()