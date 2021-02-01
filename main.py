import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn import cluster

# Map all values to numeric data to prepare for one-hot preprocessing
def preprocess_data():
    # Get numeric data
    data = make_all_values_numeric()
    
    data = map_categorical_data(data)
    data = normalize_data(data)
    return data


def make_all_values_numeric():
    data = pd.read_csv("survey.txt")
    
    # Remove timestamp columns
    data.drop(labels=["Timestamp"], axis=1, inplace=True)

    possible_answers = [   ["Male", "Female", "Other"],
                    ["15-19", "20-24", "25-29", "30-34", "35-39", "40+"],
                    ["IT/Tech/Engineering", "Construction/Mechanic", "Economics/Busniess", "Healthcare/Medicine", "Design/Architect", "Service/Restaurant", "Student/Academics", "Unemployed", "Other"],
                    ["Running", "Fitness classes (e.g. spinning, yoga, bodypump)", "Gym", "Crossfit", "Mix of exercises", "Other", "I don't workout"],
                    ["0", "1-2", "3-4", "5+"],
                    ["0-1 hours", "1-2 hours", "2-3 hours", "4+ hours"],
                    ["Yes", "No"],
                    ["Health", "Apperance", "Achievements", "Enjoyment", "Combination of above", "None of above", "I don't workout"]
                ]

    for i, columns in enumerate(data):    

        for index, element in enumerate(possible_answers[i]): 
            data[columns].replace(to_replace=element, value=index, inplace=True)

    return data

'''
Encode categorical integer features using a one-hot aka one-of-K scheme.
'''
def map_categorical_data(data):
    mask = [True, False, True, True, False, False, True, True]
    enc = OneHotEncoder(categorical_features=mask, sparse=False)
    data = enc.fit_transform(data)
    
    return data

def normalize_data(data):
    return normalize(data, norm="max", axis=0)

def cluster_data(data):
    # Initiate the DBSCAN model
    #dbscan = cluster.DBSCAN(eps=1.5)
    dbscan = cluster.KMeans(n_clusters=5)
    dbscan.fit(data)

    return dbscan

def get_full_information_about_outliers(data):
    temp = pd.read_csv("survey.txt")
    temp = temp.values
    for i, e in enumerate(data):
        if(e == -1):
            print(temp[i])

def get_full_information_about_cluster(data):

    temp = pd.read_csv("survey.txt")
    temp = temp.values

    for n in (list(set(data))):
        print("Cluster number: {} = ".format(n))
        for i, e in enumerate(data):
            if(e == n):
                print(temp[i])

        print("--------------")
    
            

def main():
    
    # Get preprocessed data
    data = preprocess_data()

    # Cluster the data
    data_clustered = cluster_data(data)

    print("Number of samples = {}".format(len(data_clustered.labels_)))

    # Get number of outliers for DBSCAN
    outliers = [line for line in data_clustered.labels_ if line == -1]
    
    print(data_clustered.labels_)

    # Get number of clusters
    number_of_clusters = len(list(set(data_clustered.labels_)))
    
    # Print out all samples that are outliers for DBSCAN
    #get_full_information_about_outliers(data_clustered.labels_)
    get_full_information_about_cluster(data_clustered.labels_)

    print("Number of clusters: {}".format(number_of_clusters))
    print("Number of outliers = {}".format(len(outliers)))
if __name__== "__main__":
    main()
