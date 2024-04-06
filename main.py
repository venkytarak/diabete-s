import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pickle
def cal_median(df, target, var):
    temp = df[df[var].notnull()]
    temp = temp[[var, target]].groupby([target])[[var]].median().reset_index()
    return temp

def median_imputation(df, target, var, var_0, var_1):
    for i in range(len(df)):
        if df.loc[i, target] == 0 and df.loc[i, var] == 0:
            df.loc[i, var] = var_0
        if df.loc[i, target] == 1 and df.loc[i, var] == 0:
            df.loc[i, var] = var_1

# ... (rest of the data preprocessing code)

# Calculate performance function
def calculate_performance(test_num, pred_y, labels):
    tp, fp, tn, fn = 0, 0, 0, 0
    for index in range(test_num):
        if labels.iloc[index] == 1:
            if labels.iloc[index] == pred_y[index]:
                tp += 1
            else:
                fn += 1
        else:
            if labels.iloc[index] == pred_y[index]:
                tn += 1
            else:
                fp += 1

    acc = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp + 1e-06)
    npv = float(tn) / (tn + fn + 1e-06)
    sensitivity = float(tp) / (tp + fn + 1e-06)
    specificity = float(tn) / (tn + fp + 1e-06)
    mcc = float(tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-06)
    f1 = float(tp * 2) / (tp * 2 + fp + fn + 1e-06)

    return acc, precision, npv, sensitivity, specificity, mcc, f1

# Load the data
df = pd.read_csv("diabetes.csv")

# ... (rest of the data preprocessing code)
d = cal_median(df,"Outcome","Insulin")
d
median_imputation(df,"Outcome","Insulin",d.loc[0,"Insulin"],d.loc[1,"Insulin"])
d=cal_median(df,"Outcome","BloodPressure")
d
median_imputation(df,"Outcome","BloodPressure",d.loc[0,"BloodPressure"],d.loc[1,"BloodPressure"])
d = cal_median(df,"Outcome","BMI")
d
median_imputation(df,"Outcome","BMI",d.loc[0,"BMI"],d.loc[1,"BMI"])
#Note: We can also us the below approach for data imputation and can get 78% accuracy.
#df["Glucose").replace(0,np.nan, inplace=True)
#df["Glucose").replace(np.nan,df("Glucose"].median(), inplace=True)
#df["BloodPressure").replace(0, np.non, inplace=True)
#df["BloodPressure"].replace(np.nan,df["BloodPressure"].median(), inplace=True)
#df["SkinThickness").replace(0, np.nan, inplace=True)
#df["SkinThickness"].replace(np.nan, df["SkinThickness"].median(), inplace=True)
#df["Insulin"].replace(0,np.nan, inplace=True)
#df["Insulin"].replace(np.nan,df["Insulin"].median(), inplace=True)
#df["BMI"].replace(0, np.nan, inplace=True)
#df["BMI"].replace(np.nan,df ["BMI"].median(), inplace=True)

#DATA VISUALIZATION

import plotly.express as exp
import plotly.io as pio

def plot_data(df, varx, vary, target):
    pio.templates.default="simple_white"
    exp.defaults.template = "ggplot2"
    exp.defaults.color_continuous_scale = exp.colors.sequential.Blackbody
    exp.defaults.width = 800
    exp.defaults.height = 600
    fig = exp.scatter(df,x=varx,y=vary,color=target)
    fig.show()

# plot_data(df,"Glucose","Age","Outcome")
# df.loc[:,"N1"] = 1
# df.loc[(df['Age']<=30) & (df['Glucose']<=120), "N1"] = 0
# df.loc[(df['Age']>30) & (df['Age']<48) & (df['Glucose']<=88), "N1"]=0 # extra condition
# df.loc[(df['Age']>=63) & (df['Glucose']<=142), "N1"] = 0 # extra condition
# df.loc[:,'N2'] = 1
# df.loc[(df['BMI']<=30),'N2'] = 0
# plot_data(df,"Pregnancies","Age","Outcome")
# df.loc[:,'N3'] = 1
# df.loc[(df["Age"]<=27) & (df['Pregnancies']<=6),"N3"] = 0
# df.loc[(df["Age"]>60) & (df["Pregnancies"]>7.5),"N3"] = 0
# plot_data(df,"Glucose","BloodPressure","Outcome")
# df.loc[:,"N4"] = 1
# df.loc[(df["Glucose"]<=105) & (df['BloodPressure']<=80),"N4"]=0
# df.loc[(df["Glucose"]<=105) & (df["BloodPressure"]>83), "N4"]=0
# df.loc[:,"N5"] = 1
# df.loc[(df["SkinThickness"]<=20),"N5"] = 0
# plot_data(df,"SkinThickness","BMI","Outcome")
# df.loc[:,"N6"] = 1
# df.loc[(df["BMI"]<30) & (df["SkinThickness"]<=20),"N6"] = 0
# df.loc[(df["BMI"]>33) & (df["SkinThickness"]<=20),"N6"] = 0
# plot_data(df,"Glucose","BMI","Outcome")
# df.loc[:,"N7"]=1
# df.loc[(df["Glucose"]<=105) & (df["BMI"]<=30),"N7"]=0
# df.loc[(df["Glucose"]<=105) & (df["BMI"]>=40),"N7"]=0
# df.loc[:,"N9"] = 1
# df.loc[(df["Insulin"]<200),"N9"]=0
# df.loc[:,"N10"] = 1
# df.loc[(df["BloodPressure"]<80),"N10"]=0
# df.loc[:,"N11"] = 1
# df.loc[(df["Pregnancies"]<4)& (df["Pregnancies"]!=0),"N10"]=0
# df["N0"] = df["BMI"] * df["SkinThickness"]
# df["N8"] = df["Pregnancies"] / df["Age"]
# df["N13"] = df["Glucose"] / df["DiabetesPedigreeFunction"]
# df["N12"] = df["Age"] * df["DiabetesPedigreeFunction"]
# df.loc[:,"N15"] = 1
# df.loc[(df["N0"]<1034) , "N15"] = 0
df
from sklearn.preprocessing import LabelEncoder

# Assuming df is your DataFrame
encoder = LabelEncoder()
df["Outcome"] = encoder.fit_transform(df["Outcome"])

# Now df["Outcome"] is encoded, and you can proceed with x and y
y = df["Outcome"]
x = df.drop("Outcome", axis=1)

# y = pd.DataFrame(df["Outcome"])
# x= df.drop("Outcome",axis=1)
# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()
# for col in y.columns:
#     y[col]=encoder.fit_transform(y[col])
# y = y["Outcome"]
# cols=["N1","N2","N3","N4","N5","N6","N7","N9","N10","N11"]
# for col in cols:
#     x[col]=encoder.fit_transform(x[col])

# Model training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.11, random_state=16)
xgbc = xgb.XGBClassifier(
    learning_rate=0.01,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1e-5,
    objective='binary:logistic',
    nthread=-1,
    scale_pos_weight=1
).fit(x_train, y_train)

# Model evaluation and performance metrics
y_pred = xgbc.predict(x_test)
test_num = len(y_test)
'''x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=16)

param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [500, 1000, 1500],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [1e-5, 1e-4, 1e-3],
}

grid_search = GridSearchCV(estimator=xgb.XGBClassifier(), param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

best_model = grid_search.best_estimator_
best_accuracy = best_model.score(x_test, y_test)
print("Best Model Accuracy:", best_accuracy)

# Model evaluation and performance metrics
y_pred = best_model.predict(x_test)
test_num = len(y_test)'''

# Calculate performance metrics
accuracy, precision, npv, sensitivity, specificity, mcc, f1 = calculate_performance(test_num, y_pred, y_test)

# Display the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("NPV:", npv)
print("Sensitivity (Recall):", sensitivity)
print("Specificity:", specificity)
print("MCC:", mcc)
print("F1 Score:", f1)

# Classification report
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using seaborn
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
#             xticklabels=['No Diabetes', 'Diabetes'],
#             yticklabels=['No Diabetes', 'Diabetes'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()


Pkl_Filename = "diabetes_model.pkl"  
with open(Pkl_Filename,'wb') as file:  
    pickle.dump(xgbc ,file)