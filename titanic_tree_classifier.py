#Reference:
# Will Cukierski. (2012). Titanic - Machine Learning from Disaster. Kaggle. https://kaggle.com/competitions/titanic
#Kaggle competitions Titanic


import matplotlib.pyplot as plt
import pandas as pd

#https://github.com/Kaggle/kaggle-api/blob/main/kaggle/api/kaggle_api.py
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi()
api.authenticate()
api.competition_download_file('titanic', file_name='train.csv')
api.competition_download_file('titanic', file_name='test.csv')
api.competition_download_file('titanic', file_name='gender_submission.csv')
import zipfile
import os

def Unzip(filepath):
    with zipfile.ZipFile(filepath,'r') as zipref:
     zipref.extractall()
     #zip.extract('nome_file_specifico')
     #os.remove(filepath)

def DeleteFile(filepath): #scrivi il path fra ''
    if os.path.exists(filepath):
        os.remove(filepath)
        print('Deleted' + filepath)
    else:
        print('File', filepath, 'does not exists')

import numpy as np

titanic_train_file_path = 'train.csv'
titanic_test_file_path = 'test.csv'

titanic_train_data = pd.read_csv(titanic_train_file_path)

titanic_test_data = pd.read_csv(titanic_test_file_path)

features = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
#only for numerical we can ask the average/mean of survived/not survived with respect to given columns
#print(pd.pivot_table(titanic_train_data, index = 'Survived', values = ['Pclass','Age'] ) )



#percentuale donne e uomini sopravvissuti per i train data

print('survived in train data',titanic_train_data.loc[:,'Survived'].sum())
print('su un totale di',titanic_train_data.loc[:,'Survived'].shape)

k=0 #donne sopravvissute
l=0 #donne morte
m=0 #maschi vivi
for s in range(891):
 if (titanic_train_data.loc[s,'Survived'] == 1) and (titanic_train_data.loc[s,'Sex']== 'female'):
     k=k+1
 elif (titanic_train_data.loc[s,'Survived'] == 0) and (titanic_train_data.loc[s,'Sex']== 'female'):
     l= l+1
 elif (titanic_train_data.loc[s,'Survived'] == 1) and (titanic_train_data.loc[s,'Sex']== 'male'):
     m= m+1
 else:
     continue






#from IPython.display import display
#display(titanic_train_data)
#df = pd.DataFrame(titanic_train_data)

#X, y described from the training data set

y=titanic_train_data.Survived
#print(len(y[:]))
#print(len(titanic_train_data.loc[titanic_train_data.Survived[:] == 0]))
#print(len(titanic_train_data.loc[titanic_train_data.Survived[:] == 1]))
X = titanic_train_data[features]
X_test = titanic_test_data[features]

#split name into last and first name because this may reveil blood relations

X[['Last_Name', 'First_Name']] = X['Name'].str.split(',', expand=True)
X_test[['Last_Name', 'First_Name']] = X_test['Name'].str.split(',', expand=True)
X.drop('Name', inplace=True, axis=1)
X_test.drop('Name', inplace=True, axis=1)
#print(X['Age'].head(20))

#cerchiamo di capire qualcosa in più su X
#print(X['Age'].head(20))
#print(X.Age.value_counts())

#Clean and order data: missing, string to float, infinity

# 1) control all object data and transform it into float data
#print(X.dtypes)
s = (X.dtypes == 'object')
object_cols = list(s[s].index)
#print(object_cols)

from sklearn.preprocessing import OrdinalEncoder

# Make copy to avoid changing original data
label_X = X.copy()
# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X[object_cols] = ordinal_encoder.fit_transform(X[object_cols])
#print(label_X.dtypes)
#print(label_X.head())
#the following command inverse the float transformation
#label_X[object_cols]=ordinal_encoder.inverse_transform(label_X[object_cols])
#print(label_X.head())

# Make copy to avoid changing original data
label_X_test = X_test.copy()
# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_test[object_cols] = ordinal_encoder.fit_transform(X_test[object_cols])



#2) deal with missing values in columns/replaced with mean of the column by default

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
#print(X_plus['Fare'].tail(20))
#print(X_plus['Fare_was_missing'].tail(20))
#print(imputed_X_plus['Fare_was_missing'].tail(20))

#distribution of the data column by column after nmanipulation
#for i in imputed_X_plus.columns:
#    plt.hist(imputed_X_plus[i])
#    plt.title(i)
#    plt.show()
#some data should be normalized or scaled? This depends on the model we want to use
#correlations among data



for col in cols_with_missing:
    X_test_plus[col + '_was_missing'] = X_test_plus[col].isnull()
my_imputer = SimpleImputer()
imputed_X_test_plus = pd.DataFrame(my_imputer.fit_transform(X_test_plus))
imputed_X_test_plus.columns = X_test_plus.columns

#split into two subsets the data in order to have some validation ones

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(imputed_X_plus, y, random_state = 0)


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error

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

my_model = DecisionTreeClassifier(max_leaf_nodes=2,random_state=1)

tentativo1_acc_mean, tentativo1_std_dev = get_model_cross_validation(my_model, 3,imputed_X_plus,y,2)
print('mean of accuracy tentativo 1', tentativo1_acc_mean, 'and standard deviation tentativo 1', tentativo1_std_dev)
tentativo2_acc_mean, tentativo2_std_dev = get_model_cross_validation(my_model, 5,imputed_X_plus,y,4)
print('mean of accuracy tentativo 2', tentativo2_acc_mean, 'and standard deviation tentativo 2', tentativo2_std_dev)
tentativo3_acc_mean, tentativo3_std_dev = get_model_cross_validation(my_model, 6,imputed_X_plus,y,8)
print('mean of accuracy tentativo 3', tentativo3_acc_mean, 'and standard deviation tentativo 3', tentativo3_std_dev)




my_model.fit(X_train, y_train)
#, early_stopping_rounds=5


predictions_val = my_model.predict(X_val)
predictions_train = my_model.predict(X_train)

#def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
 #   model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
  #  model.fit(train_X, train_y)
   # preds_val = model.predict(val_X)
    #mae = mean_absolute_error(val_y, preds_val)
    #return(mae)
#for max_leaf_nodes in [2,5, 50, 500, 5000]:
 #   my_mae = get_mae(max_leaf_nodes, X_train, X_val, y_train, y_val)
  #  print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))



print('mean_absolute_error for training data',mean_absolute_error(y_train, predictions_train))
print('mean_absolute_error for validation data',mean_absolute_error(y_val, predictions_val))

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

new_final.to_csv("Submission.csv", index=False)
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
Final.to_csv("death_previsions.csv")



#print('tutti',len(Final.sentence))
#print('direi morti',len(Final.loc[Final.sentence < 0.5]))
#print('forse vivi',len(Final.loc[Final.sentence >= 0.5]))

#test_gender_submission_data
titanic_gender_file_path = 'gender_submission.csv'
titanic_gender_data = pd.read_csv(titanic_gender_file_path)
#print(titanic_test_data.loc[:,['PassengerId','Sex']].head(20))
#print(titanic_gender_data.head(20))
#total_cells=np.product(titanic_gender_data.shape)=418
#print('survived',titanic_gender_data.loc[:,'Survived'].sum())

#voglio fare un piccolo conto di quanto ci ho preso
print('accuracy of new_predictions',(new_final.Survived == titanic_gender_data.Survived).mean()*100)
print('accuracy of old_predictions',(Final.sentence == titanic_gender_data.Survived).mean()*100)


#p=0
#for s in range(418):
 #   if ( (new_final.loc[s:'Survived'] == 0) and (titanic_gender_data.loc[s: 'Survived'] == 0) ) or ((new_final.loc[s: 'Survived'] == 1) and (titanic_gender_data.loc[s: 'Survived'] == 1)):
  #      p=p+1
   # else:
   #     continue
#print('i dati non conidono',418-p, 'tra 418')




