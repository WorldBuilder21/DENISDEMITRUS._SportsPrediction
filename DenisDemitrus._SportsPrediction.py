#!/usr/bin/env python
# coding: utf-8

# In[30]:


# necessary imports required
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
import pickle as pkl


# ### Data Preprocessing

# In[32]:


df = pd.read_csv("male_players (legacy).csv")


# In[33]:


df.info()


# In[34]:


df.describe()


# In[35]:


df.isnull().sum()


# In[36]:


# gets all numeric column and replace their NaN values with the mean.
def cleaning_numeric(df):
    df_num = df.select_dtypes(include=[np.number])
    mean_values = df_num.mean()
    numeric = df_num.fillna(mean_values)
    return numeric


# In[37]:


# this function combines both one Hot encoding, and filling all NaN values with the mode of the column
def oneHotEconding(df):
    # Select the non-numeric columns
    non_numeric = df.select_dtypes(include = ['object'])
    # fill missing values with the mode
    for col in non_numeric.columns:
        mode_value = non_numeric[col].mode()[0]
        non_numeric[col] = non_numeric[col].fillna(mode_value)

    # OneHot encoding
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_data = encoder.fit_transform(non_numeric)
    cat = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(non_numeric.columns))
    # returns the encoded dataframe
    return cat


# In[38]:


# Main function that ties everything together, cleans the data, 
# fill all Null and NaN values in both numeric and categorical columns
def cleaning_data(df):
    # reads from the csv
    # reason for droping all of these columns was to:
    # 1. to reduce the size of the dataframe, to reduce the noise when training and the time
    # 2. After inspection and research all of these Attribute would not be needed to calculate the overall score of a player
    drop_col = ['player_id', 'player_url','fifa_version','fifa_update','fifa_update_date','short_name','long_name','dob','league_id','club_name','club_team_id',
    'club_jersey_number',
    'club_loaned_from',
    'club_joined_date',
    'club_position',
    'club_contract_valid_until_year',
    'nationality_id','nationality_name','nation_jersey_number','player_face_url', 'real_face','ls','st','rs','lw',
    'lf','cf','rf','rw','lam','cam','ram','lm','lcm','cm','rcm',    'rm','lwb','ldm','cdm','rdm','rwb','lb','lcb','cb',
    'rcb','rb','gk','player_positions','league_name','league_level',
    ]
    
    # drops redudant columns
    drop_col_existing = []
    for col in drop_col:
        if col in df.columns:
            drop_col_existing.append(col)
    
    df = df.drop(drop_col_existing, axis=1)
    # dropping the columns which have null values more than 50%
    L = []
    L_less = []
    for i in df.columns:
        if((df[i].isnull().sum()) < (0.5 * (df.shape[0]))):
            L.append(i)
        else:
            L_less.append(i)
            
    # spliting the numeric columns and filling all NaN values
    numeric = cleaning_numeric(df[L])
    cat = oneHotEconding(df[L])
    return pd.concat([numeric, cat], axis=1)


# ## Feature Engineering

# In[40]:


# pick the top 7 features with the higher corelation
def picking_best_features(df):
    new_df = cleaning_data(df)
    y = new_df['overall']
    X = new_df.drop('overall', axis=1)
    
    scaler= StandardScaler()
    X = scaler.fit_transform(X)

    Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,test_size=0.2,random_state=42)

    # used XGBRegressor over random forest because it is faster
    xgb_reg = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_reg.fit(Xtrain, Ytrain)

    feature_importances = pd.Series(xgb_reg.feature_importances_, index=new_df.drop('overall', axis=1).columns)
    predicting_var = feature_importances.sort_values(ascending=False)

    print(predicting_var)

    top10 = predicting_var.head(7)
    return top10

best_features = picking_best_features(df).index.tolist() # picks the names of variables
best_features


# ## Training Model
# According the rubric Training must be done with either RandomForest, XGBoost and Gradient Boost Regressor. Decided to use all 3.

# In[42]:


# spliting the data (uisng the variables with the highest correlation) to test in all the models
df_clean = cleaning_data(df)
X = df_clean[best_features]
y = df_clean['overall']

# split the the best features and the overall to be trained by the models
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,test_size=0.2,random_state=42)


# In[43]:


# main function to train all the models
def randomForestReg():
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(Xtrain, Ytrain)
    y_pred = rf.predict(Xtest)
    mae = mean_absolute_error(y_pred, Ytest)
    mse = mean_squared_error(y_pred, Ytest)
    rmse = np.sqrt(mean_squared_error(y_pred, Ytest))
    r2 = r2_score(y_pred, Ytest)
    print(f"""
    Mean Absolute Error={mean_absolute_error(y_pred, Ytest)}
    Mean Squared Error={mean_squared_error(y_pred,Ytest)}
    Root Mean Squared Error={np.sqrt(mean_squared_error(y_pred, Ytest))}
    R2 score={r2}
    model_name={rf.__class__.__name__}
    """) 
    return {'model': rf, 'MAE': mae}


# In[44]:


def xgbReg():
    xg = xgb.XGBRegressor()
    xg.fit(Xtrain, Ytrain)
    y_pred = xg.predict(Xtest)
    mae = mean_absolute_error(y_pred, Ytest)
    mse = mean_squared_error(y_pred, Ytest)
    rmse = np.sqrt(mean_squared_error(y_pred, Ytest))
    r2 = r2_score(y_pred, Ytest)
    print(f"""
    Mean Absolute Error={mean_absolute_error(y_pred, Ytest)}
    Mean Squared Error={mean_squared_error(y_pred,Ytest)}
    Root Mean Squared Error={np.sqrt(mean_squared_error(y_pred, Ytest))}
    R2 score={r2}
    model_name={xg.__class__.__name__}
    """) 
    return {'model': xg, 'MAE': mae}


# In[45]:


def grad_boost():
    gradReg = GradientBoostingRegressor()
    gradReg.fit(Xtrain, Ytrain)
    y_pred = gradReg.predict(Xtest)
    mae = mean_absolute_error(y_pred, Ytest)
    mse = mean_squared_error(y_pred, Ytest)
    rmse = np.sqrt(mean_squared_error(y_pred, Ytest))
    r2 = r2_score(y_pred, Ytest)
    print(f"""
    Mean Absolute Error={mean_absolute_error(y_pred, Ytest)}
    Mean Squared Error={mean_squared_error(y_pred,Ytest)}
    Root Mean Squared Error={np.sqrt(mean_squared_error(y_pred, Ytest))}
    R2 score={r2}
    model_name={gradReg.__class__.__name__}
    """) 
    return {'model': gradReg, 'MAE': mae}


# ## Evaluation

# In[47]:


# Picks the model with the the least mean absolute error, because:
#A high Mean Absolute Error (MAE) score indicates that, on average, 
# the predictions made by the model are significantly different from the actual values.
# Meaning a model with the lowest MAE is likely to be the most accurate.
def picking_best_model():
    rf_result = randomForestReg()
    gd_result = grad_boost()
    xgb_result = xgbReg()
    r2_results = [rf_result, gd_result,xgb_result]
    best_model = min(r2_results, key=lambda x: x['MAE'])
    print(best_model)
    return best_model


# In[48]:


best = picking_best_model()


# In[49]:


# hyper tuning using Grid Search with Cross Validation
def hyper_tuning(model):
    cv = KFold(n_splits=3)

    PARAMETERS_gb = {
        'n_estimators': [100],
        'max_depth': [10, 20],
        'min_samples_split': [2]
    }
    
    model_gs = GridSearchCV(model, param_grid=PARAMETERS_gb, cv=cv, scoring="neg_mean_absolute_error")
    model_gs.fit(Xtrain, Ytrain)
    y_pred = model_gs.predict(Xtest)
    
    # print(model_gs.__class__.__name__, confusion_matrix(Ytest, y_pred), classification_report(Ytest, y_pred))
    return model_gs.best_estimator_
    


# In[50]:


tuned_model = hyper_tuning(best['model'])


# In[51]:


tuned_model


# In[52]:


tuned_model.__class__.__name__


# In[53]:


# accessing the tuned models performance
def tuned_model_performance(model):
    model.fit(Xtrain, Ytrain)
    y_pred = model.predict(Xtest)
    mae = mean_absolute_error(y_pred, Ytest)
    mse = mean_squared_error(y_pred, Ytest)
    rmse = np.sqrt(mean_squared_error(y_pred, Ytest))
    r2 = r2_score(y_pred, Ytest)
    print(f"""
    Mean Absolute Error={mean_absolute_error(y_pred, Ytest)}
    Mean Squared Error={mean_squared_error(y_pred,Ytest)}
    Root Mean Squared Error={np.sqrt(mean_squared_error(y_pred, Ytest))}
    R2 score={r2}
    model_name={model.__class__.__name__}
    """)
result = tuned_model_performance(tuned_model)
result


# ## Test with new data set

# In[69]:


test_df = pd.read_csv("players_22-1.csv")
cln = cleaning_data(test_df)

def testing_tuned_model(model, test_df):
    cln = cleaning_data(test_df)
    best_feature = picking_best_features(cln)
    X = df_clean[best_features]
    y = df_clean['overall']
    Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,test_size=0.2,random_state=42)
    y_pred = model.predict(Xtest)

    r2 = r2_score(y_pred, Ytest)
    print(f"""
    Mean Absolute Error={mean_absolute_error(y_pred, Ytest)}
    Mean Squared Error={mean_squared_error(y_pred,Ytest)}
    Root Mean Squared Error={np.sqrt(mean_squared_error(y_pred, Ytest))}
    R2 score={r2} """)
    
testing_tuned_model(tuned_model, test_df)
tuned_model.__class__.__name__


# In[21]:


# Save the trained model to a pickle file
pkl.dump(tuned_model, open('C:/Users/Aggyr/OneDrive/Desktop/AI_HW/' + tuned_model.__class__.__name__ + '.pkl', 'wb'))
# with open('DenisDemitrus_SportsPrediction.pkl', 'wb') as file:
#     pkl.dump(tuned_model, file)

