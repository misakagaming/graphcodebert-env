import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

def get_ds_infos():
    """
    Read the file includes data subject information.
    
    Data Columns:
    0: code [1-24]
    1: weight [kg]
    2: height [cm]
    3: age [years]
    4: gender [0:Female, 1:Male]
    
    Returns:
        A pandas DataFrame that contains inforamtion about data subjects' attributes 
    """ 

    dss = pd.read_csv("data_subjects_info.csv")
    print("[INFO] -- Data subjects' information is imported.")
    
    return dss

def set_data_types(data_types=["userAcceleration"]):
    """
    Select the sensors and the mode to shape the final dataset.
    
    Args:
        data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration] 

    Returns:
        It returns a list of columns to use for creating time-series from files.
    """
    dt_list = []
    for t in data_types:
        if t != "attitude":
            dt_list.append([t+".x",t+".y",t+".z"])
        else:
            dt_list.append([t+".roll", t+".pitch", t+".yaw"])

    return dt_list


def creat_time_series(dt_list, act_labels, trial_codes, mode="mag", labeled=True):
    """
    Args:
        dt_list: A list of columns that shows the type of data we want.
        act_labels: list of activites
        trial_codes: list of trials
        mode: It can be "raw" which means you want raw data
        for every dimention of each data type,
        [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].
        or it can be "mag" which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)
        labeled: True, if we want a labeld dataset. False, if we only want sensor values.

    Returns:
        It returns a time-series of sensor data.
    
    """
    num_data_cols = len(dt_list) if mode == "mag" else len(dt_list*3)

    if labeled:
        dataset = np.zeros((0,num_data_cols+7)) # "7" --> [act, code, weight, height, age, gender, trial] 
    else:
        dataset = np.zeros((0,num_data_cols))
        
    ds_list = get_ds_infos()
    
    print("[INFO] -- Creating Time-Series")
    for sub_id in ds_list["code"]:
        for act_id, act in enumerate(act_labels):
            for trial in trial_codes[act_id]:
                fname = 'A_DeviceMotion_data/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'
                raw_data = pd.read_csv(fname)
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                vals = np.zeros((len(raw_data), num_data_cols))
                for x_id, axes in enumerate(dt_list):
                    if mode == "mag":
                        vals[:,x_id] = (raw_data[axes]**2).sum(axis=1)**0.5        
                    else:
                        vals[:,x_id*3:(x_id+1)*3] = raw_data[axes].values
                    vals = vals[:,:num_data_cols]
                if labeled:
                    lbls = np.array([[act_id,
                            sub_id-1,
                            ds_list["weight"][sub_id-1],
                            ds_list["height"][sub_id-1],
                            ds_list["age"][sub_id-1],
                            ds_list["gender"][sub_id-1],
                            trial          
                           ]]*len(raw_data))
                    vals = np.concatenate((vals, lbls), axis=1)
                dataset = np.append(dataset,vals, axis=0)
    cols = []
    for axes in dt_list:
        if mode == "raw":
            cols += axes
        else:
            cols += [str(axes[0][:-2])]
            
    if labeled:
        cols += ["act", "id", "weight", "height", "age", "gender", "trial"]
    
    dataset = pd.DataFrame(data=dataset, columns=cols)
    return dataset
#________________________________


ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]
TRIAL_CODES = {
    ACT_LABELS[0]:[1,2,11],
    ACT_LABELS[1]:[3,4,12],
    ACT_LABELS[2]:[7,8,15],
    ACT_LABELS[3]:[9,16],
    ACT_LABELS[4]:[6,14],
    ACT_LABELS[5]:[5,13]
}

## Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"
## attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
sdt = ["attitude", "userAcceleration"]
print("[INFO] -- Selected sensor data types: "+str(sdt))    
act_labels = ACT_LABELS
print("[INFO] -- Selected activites: "+str(act_labels))    
trial_codes = [TRIAL_CODES[act] for act in act_labels]
dt_list = set_data_types(sdt)
dataset = creat_time_series(dt_list, act_labels, trial_codes, mode="raw", labeled=True)
print("[INFO] -- Shape of time-Series dataset:"+str(dataset.shape))



feature_cols = [col for col in dataset.columns]
feature_cols.remove("act")
feature_cols.remove("id")
feature_cols.remove("trial")
X = dataset[feature_cols] # Features
y = dataset["act"] # Target variable

"""
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
"""

file1 = open("decision tree.txt", "w+")

file1.write("Results\n")

lines = []

print("decision tree")
lines.append("\n***********************\ndecision tree\n***********************\n")

for id in range(24):
    print("testing for user " + str(id + 1))
    lines.append("\ntesting for user " + str(id + 1)+"\n")
    test_dataset = dataset.loc[dataset['id'] == id]
    train_dataset = dataset.loc[dataset['id'] != id]
    X_test = test_dataset[feature_cols]
    y_test = test_dataset["act"]
    X_train = train_dataset[feature_cols]
    y_train = train_dataset["act"]
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)


    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    lines.append("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)) + "\n")

    step = 1/50
    timer = 0
    count = 0
    coefficients = [-25, -25, -20, -30, 1, 1]
    active = 0
    idle = 0
    for label in y_pred:
        change = step * coefficients[int(label)]
        timer += change
        if timer < 0:
            timer = 0
        if timer > 60:
            timer = 60
        count += 1
        if count == 50:
            count = 0
            if timer < 30:
                active += 1
            else:
                idle +=1

    print("active: " + str(active))
    lines.append("active: " + str(active) + "\n")
    print("idle: " + str(idle))
    lines.append("idle: " + str(idle) + "\n")

file1.writelines(lines)
file1.close()


file1 = open("knn.txt", "w+")

file1.write("Results\n")

lines = []
print("knn")
lines.append("\n***********************\nknn\n***********************\n")

for id in range(24):
    print("testing for user " + str(id + 1))
    lines.append("\ntesting for user " + str(id + 1)+"\n")
    test_dataset = dataset.loc[dataset['id'] == id]
    train_dataset = dataset.loc[dataset['id'] != id]
    X_test = test_dataset[feature_cols]
    y_test = test_dataset["act"]
    X_train = train_dataset[feature_cols]
    y_train = train_dataset["act"]
    # Create Decision Tree classifer object
    clf = KNeighborsClassifier(n_neighbors=3)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)


    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    lines.append("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)) + "\n")

    step = 1/50
    timer = 0
    count = 0
    coefficients = [-25, -25, -20, -30, 1, 1]
    active = 0
    idle = 0
    for label in y_pred:
        change = step * coefficients[int(label)]
        timer += change
        if timer < 0:
            timer = 0
        if timer > 60:
            timer = 60
        count += 1
        if count == 50:
            count = 0
            if timer < 30:
                active += 1
            else:
                idle +=1

    print("active: " + str(active))
    lines.append("active: " + str(active) + "\n")
    print("idle: " + str(idle))
    lines.append("idle: " + str(idle) + "\n")

file1.writelines(lines)
file1.close()

file1 = open("logistic regression.txt", "w+")

file1.write("Results\n")

lines = []

print("logistic regression")
lines.append("\n***********************\nlogistic regression\n***********************\n")

for id in range(24):
    print("testing for user " + str(id + 1))
    lines.append("\ntesting for user " + str(id + 1)+"\n")
    test_dataset = dataset.loc[dataset['id'] == id]
    train_dataset = dataset.loc[dataset['id'] != id]
    X_test = test_dataset[feature_cols]
    y_test = test_dataset["act"]
    X_train = train_dataset[feature_cols]
    y_train = train_dataset["act"]
    # Create Decision Tree classifer object
    clf = LogisticRegression()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)


    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    lines.append("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)) + "\n")

    step = 1/50
    timer = 0
    count = 0
    coefficients = [-25, -25, -20, -30, 1, 1]
    active = 0
    idle = 0
    for label in y_pred:
        change = step * coefficients[int(label)]
        timer += change
        if timer < 0:
            timer = 0
        if timer > 60:
            timer = 60
        count += 1
        if count == 50:
            count = 0
            if timer < 30:
                active += 1
            else:
                idle +=1

    print("active: " + str(active))
    lines.append("active: " + str(active) + "\n")
    print("idle: " + str(idle))
    lines.append("idle: " + str(idle) + "\n")

file1.writelines(lines)
file1.close()


file1 = open("naive bayes.txt", "w+")

file1.write("Results\n")

lines = []

print("naive bayes")
lines.append("\n***********************\nnaive bayes\n***********************\n")

for id in range(24):
    print("testing for user " + str(id + 1))
    lines.append("\ntesting for user " + str(id + 1)+"\n")
    test_dataset = dataset.loc[dataset['id'] == id]
    train_dataset = dataset.loc[dataset['id'] != id]
    X_test = test_dataset[feature_cols]
    y_test = test_dataset["act"]
    X_train = train_dataset[feature_cols]
    y_train = train_dataset["act"]
    # Create Decision Tree classifer object
    clf = GaussianNB()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)


    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    lines.append("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)) + "\n")

    step = 1/50
    timer = 0
    count = 0
    coefficients = [-25, -25, -20, -30, 1, 1]
    active = 0
    idle = 0
    for label in y_pred:
        change = step * coefficients[int(label)]
        timer += change
        if timer < 0:
            timer = 0
        if timer > 60:
            timer = 60
        count += 1
        if count == 50:
            count = 0
            if timer < 30:
                active += 1
            else:
                idle +=1

    print("active: " + str(active))
    lines.append("active: " + str(active) + "\n")
    print("idle: " + str(idle))
    lines.append("idle: " + str(idle) + "\n")

file1.writelines(lines)
file1.close()

file1 = open("svm.txt", "w+")

file1.write("Results\n")

lines = []

print("svm")
lines.append("\n***********************\nsvm\n***********************\n")

for id in range(24):
    print("testing for user " + str(id + 1))
    lines.append("\ntesting for user " + str(id + 1)+"\n")
    test_dataset = dataset.loc[dataset['id'] == id]
    train_dataset = dataset.loc[dataset['id'] != id]
    X_test = test_dataset[feature_cols]
    y_test = test_dataset["act"]
    X_train = train_dataset[feature_cols]
    y_train = train_dataset["act"]
    # Create Decision Tree classifer object
    clf = SVC(decision_function_shape='ovo')

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)


    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    lines.append("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)) + "\n")

    step = 1/50
    timer = 0
    count = 0
    coefficients = [-25, -25, -20, -30, 1, 1]
    active = 0
    idle = 0
    for label in y_pred:
        change = step * coefficients[int(label)]
        timer += change
        if timer < 0:
            timer = 0
        if timer > 60:
            timer = 60
        count += 1
        if count == 50:
            count = 0
            if timer < 30:
                active += 1
            else:
                idle +=1

    print("active: " + str(active))
    lines.append("active: " + str(active) + "\n")
    print("idle: " + str(idle))
    lines.append("idle: " + str(idle) + "\n")

file1.writelines(lines)
file1.close()
