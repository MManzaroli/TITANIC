#Reference:
# Will Cukierski. (2012). Titanic - Machine Learning from Disaster. Kaggle. https://kaggle.com/competitions/titanic
#Kaggle competitions Titanic

import pandas as pd
import numpy as np

import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi()
api.authenticate()
api.competition_download_file('titanic', file_name='train.csv')
api.competition_download_file('titanic', file_name='test.csv')
api.competition_download_file('titanic', file_name='gender_submission.csv')

titanic_train_file_path = 'train.csv'
titanic_test_file_path = 'test.csv'

titanic_train_data = pd.read_csv(titanic_train_file_path)

titanic_test_data = pd.read_csv(titanic_test_file_path)

y=titanic_train_data.Survived
features = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
X = titanic_train_data[features]
X_test = titanic_test_data[features]
X[['Last_Name', 'First_Name']] = X['Name'].str.split(',', expand=True)
X_test[['Last_Name', 'First_Name']] = X_test['Name'].str.split(',', expand=True)
X.drop('Name', inplace=True, axis=1)
X_test.drop('Name', inplace=True, axis=1)

s = (X.dtypes == 'object')
object_cols = list(s[s].index)

from sklearn.preprocessing import OrdinalEncoder

# Make copy to avoid changing original data
label_X = X.copy()
# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X[object_cols] = ordinal_encoder.fit_transform(X[object_cols])

# Make copy to avoid changing original data
label_X_test = X_test.copy()
# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_test[object_cols] = ordinal_encoder.fit_transform(X_test[object_cols])

from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_X = pd.DataFrame(my_imputer.fit_transform(label_X))
imputed_X.columns = label_X.columns

X_plus = label_X.copy()
#.isnull() create boolean true and false whether the value is missing or not
#.isnull().any() controls whether in the column are all not missing or there is at least a missing value
#i.e. returns True whether there is at least one missing and False otherwise
cols_with_missing = [col for col in label_X.columns
                     if label_X[col].isnull().any()]


imputed_X_test = pd.DataFrame(my_imputer.fit_transform(label_X_test))
imputed_X_test.columns = label_X_test.columns

#missing values in each column
X_test_plus = label_X_test.copy()
cols_with_missing = [col for col in label_X_test.columns
                     if label_X_test[col].isnull().any()]

#add column to remmber whether changes has been made wrt missing values
for col in cols_with_missing:
    X_plus[col + '_was_missing'] = X_plus[col].isnull()
my_imputer = SimpleImputer()
#boolean to 1 and 0
imputed_X_plus = pd.DataFrame(my_imputer.fit_transform(X_plus))
imputed_X_plus.columns = X_plus.columns

for col in cols_with_missing:
    X_test_plus[col + '_was_missing'] = X_test_plus[col].isnull()
my_imputer = SimpleImputer()
imputed_X_test_plus = pd.DataFrame(my_imputer.fit_transform(X_test_plus))
imputed_X_test_plus.columns = X_test_plus.columns

#split into two subsets the data in order to have some validation ones

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(imputed_X_plus, y, random_state=23)
#, test_size=0.20


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


#divide in train_val in k_fold modi diversi, poi applica la cross_validation per ogni suddivisione e controlla acc media e std_deviation del modello scelto
def get_model_cross_validation(model, k_fold, X, y, cv):
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
    X = np.array(X)
    y = np.array(y)
    h_fold_mean={}
    h_fold_std_deviation={}
    s=0
    for train, val in kf.split(X):
        X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]
        model.fit(X_train, y_train)
        model.predict(X_val)
        h_fold_acc = cross_val_score(model, X_train, y_train, cv=cv)
        h_fold_mean[s] = h_fold_acc.mean()
        h_fold_std_deviation[s] = h_fold_acc.std()
        s=s+1
    return h_fold_mean, h_fold_std_deviation

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#my_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.006, penalty="l2", random_state=123))
my_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced",
    solver="liblinear",  penalty="l2", random_state=0))
#, max_iter=80



tentativo1_acc_mean, tentativo1_std_dev = get_model_cross_validation(my_model, 3,imputed_X_plus,y,2)
print('mean of accuracy tentativo 1', tentativo1_acc_mean, 'and standard deviation tentativo 1', tentativo1_std_dev)
tentativo2_acc_mean, tentativo2_std_dev = get_model_cross_validation(my_model, 5,imputed_X_plus,y,4)
print('mean of accuracy tentativo 2', tentativo2_acc_mean, 'and standard deviation tentativo 2', tentativo2_std_dev)
tentativo3_acc_mean, tentativo3_std_dev = get_model_cross_validation(my_model, 6,imputed_X_plus,y,8)
print('mean of accuracy tentativo 3', tentativo3_acc_mean, 'and standard deviation tentativo 3', tentativo3_std_dev)




#my_model = LogisticRegression(random_state=0, max_iter=10)
my_model.fit(X_train,y_train)
predictions_val = my_model.predict(X_val)
predictions_train = my_model.predict(X_train)

from sklearn.metrics import mean_absolute_error
print('mean_absolute_error for training data',mean_absolute_error(y_train, predictions_train))
print('mean_absolute_error for validation data',mean_absolute_error(y_val, predictions_val))

acc = accuracy_score(y_val, predictions_val)
print(acc, 'accuracy per validation_data')
acc_train = accuracy_score(y_train, predictions_train)
print(acc_train, 'accuracy per training data')

from mlxtend.evaluate import bias_variance_decomp
y_error, avg_bias, avg_var = bias_variance_decomp(my_model,
                                                  X_train.values, y_train.values,
                                                  X_val.values, y_val.values,
                                                  loss='0-1_loss',
                                                  random_seed=123)
print('Using Single Estimator')
print('Average expected loss: %.3f' % y_error)
print('Average bias: %.3f' % avg_bias)
print('Average variance: %.3f' % avg_var)

test_predictions = my_model.predict(imputed_X_test_plus)
final_with_sex = pd.DataFrame({'Sex': X_test.Sex, 'Survived': pd.Series(test_predictions)})
for s in range(418):
 if (final_with_sex.loc[s,'Sex']== 'female') and (final_with_sex.loc[s,'Survived']>=0.258) :
     final_with_sex.loc[s, 'Survived'] = 1
 elif (final_with_sex.loc[s,'Sex']== 'female') and (final_with_sex.loc[s,'Survived']<0.258) :
     final_with_sex.loc[s, 'Survived'] = 0
 elif (final_with_sex.loc[s, 'Sex'] == 'male') and (final_with_sex.loc[s, 'Survived'] >= 0.811):
     final_with_sex.loc[s, 'Survived'] = 1
 else:
# (final_with_sex.loc[s, 'Sex'] == 'male') and (final_with_sex.loc[s, 'Survived'] < 0.811):
     final_with_sex.loc[s, 'Survived'] = 0

final_with_sex.loc[:,'Survived'] = final_with_sex.loc[:,'Survived'].astype(int)
new_final = pd.DataFrame({'PassengerId': X_test.PassengerId, 'Survived': final_with_sex.Survived})

new_final.to_csv("death_previsions.csv", index=False)
Final = pd.DataFrame({'Name': X_test.First_Name+", "+X_test.Last_Name, 'sentence': pd.Series(np.round(test_predictions,0).astype(int))})

#print('test_predictions',pd.Series(test_predictions))
#arrotondiamo secondo la percentuale di sopravvivenza uomini-donne dei train data
print('nuovo conto sopravvissuti', new_final.Survived.sum())



#test_predictions = np.round(test_predictions,0).astype(int)
#print(pd.Series(test_predictions))
print('vecchio conto sopravvissuti', Final.sentence.sum())
#print(Final.columns)
#print(Final.shape)
#print(Final.head())
Ffinal = new_final = pd.DataFrame({'PassengerId': X_test.PassengerId, 'Survived': Final.sentence})
Ffinal.to_csv("Submission.csv", index=False)

