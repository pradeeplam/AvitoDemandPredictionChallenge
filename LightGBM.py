'''

<Insert description here>
'''

import time

# General
import numpy as np
import pandas as pd
import zipfile

# Models 
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Clean-up
import gc

# Gradient boosting
import lightgbm as lgb
from sklearn.cross_validation import KFold

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string

NFOLDS = 5
SEED = 42
VALID = True

print("Data Load Stage")

# Load training data
zf = zipfile.ZipFile("/mnt/c/Users/pl199/Downloads/Avito_Data/train.csv.zip") 
training = pd.read_csv(zf.open("train.csv"), index_col = "item_id", parse_dates = ["activation_date"]) 
traindex = training.index
ntrain = training.shape[0]

kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED) # KFold -> Split training data into training + verification

# Load testing data
zf = zipfile.ZipFile("/mnt/c/Users/pl199/Downloads/Avito_Data/test.csv.zip") 
testing = pd.read_csv(zf.open("test.csv"), index_col = "item_id", parse_dates = ["activation_date"]) 
testdex = testing.index
ntest = testing.shape[0]

# Load meta data
zf = zipfile.ZipFile("/mnt/c/Users/pl199/Downloads/Avito_Data/meta.csv.zip") 
meta = pd.read_csv(zf.open("meta.csv"), index_col = "item_id") 

# Load contact data
zf = zipfile.ZipFile("/mnt/c/Users/pl199/Downloads/Avito_Data/contact.csv.zip") 
contact = pd.read_csv(zf.open("contact.csv"), index_col = "item_id") 

# Load op data
zf = zipfile.ZipFile("/mnt/c/Users/pl199/Downloads/Avito_Data/op_tag.csv.zip") 
op_data = pd.read_csv(zf.open("op_tag.csv"), index_col = "item_id") 

# Load city population
city_pop = pd.read_csv("/mnt/c/Users/pl199/Downloads/Avito_Data/city_population.csv") 

# Load region-city features
rc_features = pd.read_csv("/mnt/c/Users/pl199/Downloads/Avito_Data/region_city_features.csv")


print("Data Merging State")

y = training["deal_probability"].copy()
training.drop("deal_probability", axis=1, inplace=True)
data = training.append(testing)
data = data.join([meta,contact,op_data])

del training,testing,meta,contact,op_data
gc.collect()

print("Feature Engineering") # Well...Some of it (The rest was done w/ other scripts)

# Scale price
data["price"] = np.log(data["price"]+0.001)
# Fill missing price w/ price of median item from group
median = data.groupby(["parent_category_name","category_name"])['price'].transform('median')
data['price'].fillna(median,inplace=True)
# Fill missing image_top_1 w/
data["image_top_1"].fillna(-999,inplace=True) # Questionable choice
# Fill missing op_tag
data["op_tag"].fillna(False,inplace=True)
# Day of week in numerical form (Monday = 0, Sunday=6)
data["weekday"] = data['activation_date'].dt.weekday 

data.drop(["activation_date","image","description","title"],axis=1,inplace=True) # Can drop these col as already represented in other col

# Integrate region_city clusters into data (Added to categorical)
obs = ["lat_lon_hdbscan_cluster_05_03", "lat_lon_hdbscan_cluster_10_03", "lat_lon_hdbscan_cluster_20_03", "region", "city"]
data = data.reset_index().merge(rc_features[obs], how="left", on=["region","city"]).set_index("item_id")

del rc_features 
gc.collect()

# Integrate city population
cities = data["city"].unique()

unique_cities = []
for c in cities:
	if len(data[data["city"] == c]["region"].unique()) == 1:
		unique_cities.append(c)

# Fill in with available info
city_pop = city_pop[city_pop["city"].isin(unique_cities)]
data = data.reset_index().merge(city_pop[["city","population"]], how="left", on="city").set_index("item_id")

# Fill in with mean from cluster_20_03
mean = data.groupby(["lat_lon_hdbscan_cluster_20_03"])['population'].transform('mean')
data['population'].fillna(mean,inplace=True)

del city_pop
gc.collect()

# Encoder categorical data -> Convert to numerical
# Original
categorical = ["user_id","region","city","parent_category_name","category_name","user_type","image_top_1","param_1","param_2","param_3"]
# Added
categorical += ["op_tag","c_tag","lat_lon_hdbscan_cluster_05_03","lat_lon_hdbscan_cluster_10_03","lat_lon_hdbscan_cluster_20_03"]

lbl = preprocessing.LabelEncoder() # Init encoder
for col in categorical:
	data[col].fillna('unknown') 
	data[col] = lbl.fit_transform(data[col].astype(str)) # Use encoder on col 

print("Modeling Stage")

# Put data back into testing/training format

X = data.loc[traindex].values
testing = data.loc[testdex].values
fnames = data.columns.tolist()

del data
gc.collect()

modelstart = time.time()

lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt', # "gradient boosting decision tree"
    'objective': 'regression',
    'metric': 'rmse', # root mean square error
    # 'max_depth': 15,
    'num_leaves': 270,
    'feature_fraction': 1, 
    'bagging_fraction': 0.75,
    'bagging_freq': 2,
    'learning_rate': 0.016,
    'verbose': 0
} 

if VALID == False: # Using validation data 
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.10, random_state=23)
        
    # LGBM Dataset Formatting 
    lgtrain = lgb.Dataset(X_train, y_train,
                    feature_name=fnames,
                    categorical_feature = categorical)
    lgvalid = lgb.Dataset(X_valid, y_valid,
                    feature_name=fnames,
                    categorical_feature = categorical)
    del X, X_train; gc.collect()
    
    # Go Go Go
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=20000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    print("Model Evaluation Stage")
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
    del X_valid ; gc.collect()

else: # Using test data - Produce submittable output
    # LGBM Dataset Formatting 
    lgtrain = lgb.Dataset(X, y,
                    feature_name=fnames,
                    categorical_feature = categorical)
    del X; gc.collect()
    # Go Go Go
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=1550,
        verbose_eval=100
    )

# Feature Importance Plot
#f, ax = plt.subplots(figsize=[7,10])
#lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
#plt.title("Light GBM Feature Importance")
#plt.savefig('feature_import.png')

print("Model Evaluation Stage")
lgpred = lgb_clf.predict(testing) 

lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
lgsub.to_csv("/mnt/c/Users/pl199/Downloads/Avito_Data/lgsub.csv",index=True,header=True)

print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))