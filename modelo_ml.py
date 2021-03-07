# PIPELINE FOR MACHINE LEARNING
# 1. GET DATA
# 2. CLEAN, PREPARE and MANIPULATE DATA
# 3. TRAIN MODEL
# 4. TEST DATA
# 5. IMPROVE


# pip install Pipeline
# pip install sklearn
from sklearn.pipeline import Pipeline
from sklearn import datasets
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd

# 1. GET DATA
# read the dataset
df = pd.read_csv(
    'C:\PESSOAL\project_python\project_pipeline\pipeline_machine_learning/drives/adult.data')

print(df.head())

# 2. CLEAN, PREPARE and MANIPULATE DATA
# remove unnecessary columns, redundance with education
# axis: is a column, inplace: to modify my dataframe in memory
df.drop(['education'], axis=1, inplace=True)

# split data and classes
# x: data will receive all columns except the classe (income) as inplace=Flase will not drop the column, y: classes
x = df.drop('income', axis=1, inplace=False)
y = df.income

# TRAIN and TEST DATA
# split train and test 70x30
x_train, x_test, y_train, y_test = train_test_split(x, y)

# select non numeric columns - decision tree
#var = df.select_dtypes(include='object')
# print(var.head())
df.select_dtypes(include='object')

# OneHotEncoder: Categorical column to integer
ohe = OneHotEncoder(use_cat_names=True)
x_train = ohe.fit_transform(x_train)
x_train.head()

# StandardScaler - pre-processor to put numerical column in the same scale
scaler = StandardScaler().fit(x_train)

scaler

values_scale = scaler.transform(x_train)
values_scale[:10]
x_train = scaler.transform(x_train)

# generate the model - could be any model
# instance of the classifier decision tree and train the model
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(x_train, y_train)

# Apply object ohe and pre-processor on data for test
x_test = ohe.transform(x_test)
scaler_test = StandardScaler().fit(x_test)
x_test = scaler_test.transform(x_test)
x_test[:10]

# predict
clf_tree.predict(x_test)

# Validate the model
acuracy = clf_tree.score(x_test, y_test)

acuracy


# Pipeline
# will create a kind of alias for each method
pip_1 = Pipeline([
    ('ohe', OneHotEncoder()),
    ('scaler', StandardScaler()),
    ('clf', tree.DecisionTreeClassifier())
])

pip_1.steps

# flows of pipeline processes
x_train, x_test, y_train, y_test = train_test_split(x, y)

pip_1.fit(x_train, y_train)

# Model validation
acuracy = pip_1.score(x_test, y_test)

acuracy

# other examples that can be used with diferent configurations and params
pip_max_depth = Pipeline([
    ('ohe', OneHotEncoder()),
    ('scaler', MinMaxScaler()),
    ('clf', tree.DecisionTreeClassifier(max_depth=3))
])

pip_max_depth_std = Pipeline([
    ('ohe', OneHotEncoder()),
    ('scaler', StandardScaler()),
    ('clf', tree.DecisionTreeClassifier(max_depth=3))
])

# Validating the models
pip_max_depth.fit(x_train, y_train)
acuracy = pip_max_depth.score(x_test, y_test)
acuracy

pip_max_depth_std.fit(x_train, y_train)
acuracy = pip_max_depth_std.score(x_test, y_test)
acuracy


# Pipeline with step of median
mediana = Pipeline(steps=[
    ('mediana', SimpleImputer(strategy='median'))
])

# Pipeline with step of frequency
frequency = Pipeline(steps=[
    ('frequency', SimpleImputer(strategy='most frequent'))
])

# create a Pipeline that has both components
data_cleaning = ColumnTransformer(trasnformers[
    ('mediana', mediana, ['education-num']),
    ('frequent', frequente, ['race'])
])

# Final Pipeline
pipeline_final = Pipeline([
    ('datacleaning', data_cleaning),
    ('ohe', OneHotEncoder()),
    ('standardscaler', StandardScaler()),
    ('tree', tree.DecisionTreeClassifier())
])

pipeline_final.fit(x_train, y_train)
pipeline_final

# grid search
# define a dictionary
parameters_grid = dict(tree__max_depth=[3, 4, 5, 6, 7, 8, 9, 10])

# IMPROVE
# grid search find the best params for model
# object grisearch wiht defined params and configurations for cross validation with 5 folds
grid = GridSearchCV(
    pipeline_final, param_grid=parameters_grid, cv=5, scoring='accuracy')

# run grid search
grid.fit(x, y)

# result
grid.cv_results_

grid.best_params_

grid.best_score_


StandardScaler(copy=True, with_mean=True, with_std=True)
