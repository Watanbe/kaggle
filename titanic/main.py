# Start Python Imports
import math, time, random, datetime

# Data Manipulation
import numpy as np
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
plt.style.use('seaborn-whitegrid')

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

# Machine learning
import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool, cv

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
gender_submission = pd.read_csv('data/gender_submission.csv')

print(train.head(15))
print(test.head())
print(gender_submission.head())
print(train.describe())

train.Age.plot.hist()
missingno.matrix(train, figsize = (30,10))

print(train.isnull().sum())

df_bin = pd.DataFrame()
df_con = pd.DataFrame()


print(train.dtypes)

fig = plt.figure(figsize=(20,1))
sns.countplot(y='Survived', data=train)
print(train.Survived.value_counts())



df_bin['Survived'] = train['Survived']
df_con['Survived'] = train['Survived']

print(df_bin.head())
print(df_con.head())

sns.distplot(train.Pclass)
print(train.Pclass.isnull().sum())

df_bin['Pclass'] = train['Pclass']
df_con['Pclass'] = train['Pclass']

print(train.Name.value_counts())


plt.figure(figsize=(20,5))
sns.countplot(y='Sex', data=train)

print(train.Sex.isnull().sum())

df_bin['Sex'] = train['Sex']
df_bin['Sex'] = np.where(df_bin['Sex'] == 'female', 1, 0)
df_con['Sex'] = train['Sex']


fig = plt.figure(figsize=(20,20))
sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['Sex'], kde_kws={'label': 'Survived'})
sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['Sex'], kde_kws={'label': 'Did not survive'})



print(train.Age.isnull().sum())


def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20,5), use_bin_df=False):
    if use_bin_df:
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=bin_df)
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == 1][target_column],
                     kde_kws={"label": "Survived"})
        
        sns.distplot(data.loc[data[label_column] == 0][target_column],
                     kde_kws={"label": "Did not survived"})
        
    else:
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=data)
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == 1][target_column],
                     kde_kws={"label": "Survived"})
        
        sns.distplot(data.loc[data[label_column] == 0][target_column],
                     kde_kws={"label": "Did not survived"})


print(train.SibSp.isnull().sum())
print(train.SibSp.value_counts())

df_bin['SibSp'] = train['SibSp']
df_con['SibSp'] = train['SibSp']

plot_count_dist(train, bin_df = df_bin, label_column="Survived", target_column="SibSp", figsize=(20,10))


print(train.Parch.isnull().sum())
print(train.Parch.value_counts())
df_bin['Parch'] = train['Parch']
df_con['Parch'] = train['Parch']
plot_count_dist(train, bin_df = df_bin, label_column="Survived", target_column="Parch", figsize=(20,10))


print(train.head())
print(df_con.head())


print(train.Ticket.isnull().sum())
sns.countplot(y="Ticket", data=train)
print(train.Ticket.value_counts())

print("There are {} unique ticket values".format(len(train.Ticket.unique())))



print(train.Fare.isnull().sum())
sns.countplot(y="Fare", data=train)
print(train.Fare.value_counts())
print(train.Fare.dtype)
print("There are {} unique ticket values".format(len(train.Fare.unique())))
df_bin['Fare'] = pd.cut(train['Fare'], bins=5)
df_con['Fare'] = train['Fare']
plot_count_dist(train, bin_df = df_bin, label_column="Survived", target_column="Fare", figsize=(20,10), use_bin_df=True)

print(train.Cabin.isnull().sum())
print(train.Fare.value_counts())


print("Embarked")
print(train.Embarked.isnull().sum())
print(train.Embarked.value_counts())
sns.countplot(y="Embarked", data=train)
df_bin['Embarked'] = train['Embarked']
df_con['Embarked'] = train['Embarked']
print(len(df_con))
df_con = df_con.dropna(subset=["Embarked"])
df_bin = df_bin.dropna(subset=["Embarked"])
print(len(df_con))


print(df_bin.head())

one_hot_cols = df_bin.columns.tolist()
one_hot_cols.remove("Survived")
df_bin_enc = pd.get_dummies(df_bin, columns=one_hot_cols)
print(df_bin_enc.head())

df_embarked_one_hot = pd.get_dummies(df_con["Embarked"], prefix="embarked")
df_sex_one_hot = pd.get_dummies(df_con["Sex"], prefix="sex")
df_pclass_one_hot = pd.get_dummies(df_con["Pclass"], prefix="pclass")


df_con_enc = pd.concat([df_con, 
                        df_embarked_one_hot, 
                        df_sex_one_hot,
                        df_pclass_one_hot], axis=1)
df_con_enc = df_con_enc.drop(["Embarked", "Sex", "Pclass"], axis=1)
print(df_con_enc.head())

selected_df = df_con_enc

print(selected_df.head())

x_train = selected_df.drop("Survived", axis=1)
y_train = selected_df.Survived

print(x_train.shape)
print(x_train.head())

print(y_train.shape)

# function that runs a requested algorithm and returns the accuracy
def fit_ml_algo(algo, x_train, y_train, cv):
    model = algo.fit(x_train, y_train)
    acc = round(model.score(x_train, y_train) * 100, 2)
    
    train_pred = model_selection.cross_val_predict(algo, x_train, y_train, 
                                                   cv=cv, n_jobs = -1)
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)

    return train_pred, acc, acc_cv


# logistic regression 
print("\nLogistic Regression")
start_time = time.time()
train_pred_log, acc_log, acc_cv_log = fit_ml_algo(LogisticRegression(), 
                                                  x_train, y_train, 10)

log_time = (time.time() - start_time)
print("\nLogistic Regression")
print("Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))


print("\nKNN")
# k-Nearest Neighbours
start_time = time.time()
train_pred_knn, acc_knn, acc_cv_knn = fit_ml_algo(KNeighborsClassifier(), 
                                                  x_train, y_train, 10)

log_time = (time.time() - start_time)
print("Accuracy: %s" % acc_knn)
print("Accuracy CV 10-Fold: %s" % acc_cv_knn)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))


print("\nGaussian Naive Bayes")
# Gaussian Naive Bayes
start_time = time.time()
train_pred_gaussian, acc_gaussian, acc_cv_gaussian = fit_ml_algo(
                                                    GaussianNB(), 
                                                    x_train, y_train, 10)

log_time = (time.time() - start_time)
print("Accuracy: %s" % acc_gaussian)
print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))




print("\nLinear SVC")
# Linear SVC
start_time = time.time()
train_pred_svc, acc_svc, acc_cv_svc = fit_ml_algo(
                                                    LinearSVC(), 
                                                    x_train, y_train, 10)

log_time = (time.time() - start_time)
print("Accuracy: %s" % acc_svc)
print("Accuracy CV 10-Fold: %s" % acc_cv_svc)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))



print("\nStochastic Gradient Descendent")
# Stochastic Gradient Descendent
start_time = time.time()
train_pred_sgd, acc_sgd, acc_cv_sgd = fit_ml_algo(
                                                    SGDClassifier(), 
                                                    x_train, y_train, 10)

log_time = (time.time() - start_time)
print("Accuracy: %s" % acc_sgd)
print("Accuracy CV 10-Fold: %s" % acc_cv_sgd)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))



print("\nDecision Tree Classifier")
# Decision Tree Classifier
start_time = time.time()
train_pred_dtc, acc_dtc, acc_cv_dtc = fit_ml_algo(
                                                    DecisionTreeClassifier(), 
                                                    x_train, y_train, 10)

log_time = (time.time() - start_time)
print("Accuracy: %s" % acc_dtc)
print("Accuracy CV 10-Fold: %s" % acc_cv_dtc)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))



print("\nGradient Boost Trees")
# Gradient Boost Trees
start_time = time.time()
train_pred_gbt, acc_gbt, acc_cv_gbt = fit_ml_algo(
                                                    GradientBoostingClassifier(), 
                                                    x_train, y_train, 10)

log_time = (time.time() - start_time)
print("Accuracy: %s" % acc_gbt)
print("Accuracy CV 10-Fold: %s" % acc_cv_gbt)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))


print(x_train.head())
print(y_train.head())

print("CatBoost")
# CatBoost
cat_features = np.where(x_train.dtypes != np.float)[0]
print(cat_features)

train_pool = Pool(x_train, y_train, cat_features)

cat_boost_model = CatBoostClassifier(iterations=1000, custom_loss=["Accuracy"],
                                     loss_function='Logloss')
cat_boost_model.fit(train_pool, plot=True)
acc_catboost = round(cat_boost_model.score(x_train, y_train) * 100, 2)

start_time = time.time()
cv_params = cat_boost_model.get_params()
cv_data = cv(train_pool, cv_params, fold_count=10, plot=True)

catboost_time = (time.time() - start_time)
print(catboost_time)
acc_cv_catboost = round(np.max(cv_data['test-Accuracy-mean']) * 100, 2)

print("---CatBoost Metrics---")
print("Accuracy: {}".format(acc_catboost))
print("Accuracy cross-validation 10-fold: {}".format(acc_cv_catboost))
print("Running time: {}".format(datetime.timedelta(seconds=catboost_time)))

# Models Results
models = pd.DataFrame({
    "Model": ["Logistic Regression", "KNN", "Gaussian Naive Bayes",
              "Linear SVC", "Stochastic Gradient Descendent", 
              "Decision Tree Classifier", "Gradient Boost Trees",
              "CatBoost"],
    "Score": [acc_log, acc_knn, acc_gaussian, acc_svc, acc_sgd, acc_dtc,
              acc_gbt, acc_catboost]
    })

print("---Regular Accuracy Scores---")
models = models.sort_values(by="Score", ascending=False)
print(models)

models_cv = pd.DataFrame({
    "Model": ["Logistic Regression", "KNN", "Gaussian Naive Bayes",
              "Linear SVC", "Stochastic Gradient Descendent", 
              "Decision Tree Classifier", "Gradient Boost Trees",
              "CatBoost"],
    "Score": [acc_cv_log, acc_cv_knn, acc_cv_gaussian, acc_cv_svc, acc_cv_sgd, acc_cv_dtc,
              acc_cv_gbt, acc_cv_catboost]
    })
models_cv = models_cv.sort_values(by="Score", ascending=False)
print("---Cross Validation Accuracy scores---")
print(models_cv)



#Feature importance
def feature_importance(model, data):
    fea_imp = pd.DataFrame({"imp": model.feature_importances_, "col": data.columns})
    fea_imp = fea_imp.sort_values(["imp", "col"], ascending=[True, False]).iloc[-30:]
    _ = fea_imp.plot(kind="barh", x="col", y="imp", figsize=(20,10))
    return fea_imp

catboost_importance = feature_importance(cat_boost_model, x_train)
print(catboost_importance)


metrics = ["Precision", "Recall", "F1", "AUC"]
eval_metrics = cat_boost_model.eval_metrics(train_pool, metrics=metrics, plot=True)


for metric in metrics:
    print(str(metric)+": {}".format(np.mean(eval_metrics[metric])))


# Submission
test_embarked_one_hot = pd.get_dummies(test["Embarked"], prefix="embarked")
test_sex_one_hot = pd.get_dummies(test["Sex"], prefix="sex")
test_pclass_one_hot = pd.get_dummies(test["Pclass"], prefix="pclass")

test = pd.concat([test, test_embarked_one_hot, test_sex_one_hot, test_pclass_one_hot], axis=1)

wanted_test_columns = x_train.columns
print(wanted_test_columns)

predictions = cat_boost_model.predict(test[wanted_test_columns])
print(predictions[:20])

submission = pd.DataFrame()
submission["PassengerId"] = test["PassengerId"]
submission["Survived"] = predictions
print(submission.head())
submission["Survived"] = submission["Survived"].astype(int)


if len(submission) == len(test):
    print("Submission dataframe is the same length as test ({} rows).".format(len(submission)))
else:
    print("Dataframes mismatched, won't be able to submit to Kaggle.")

submission.to_csv('../catboost_submission.csv', index=False)
print('Submission CSV is ready!')



















