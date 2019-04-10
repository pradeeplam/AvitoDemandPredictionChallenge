
'''
Trying to tag items which have "outlier prices".
Current method: MAD z-score with log to make data less skewed
'''

import pandas as pd
import numpy as np
import zipfile


# Read Data from files
zf = zipfile.ZipFile('/mnt/c/Users/pl199/Downloads/Avito_Data/train.csv.zip')
train_data = pd.read_csv(zf.open('train.csv'))

zf = zipfile.ZipFile('/mnt/c/Users/pl199/Downloads/Avito_Data/test.csv.zip')
test_data = pd.read_csv(zf.open('test.csv'))

print("Done loading data!")

# Break data into chunks
obs = ["parent_category_name","category_name"] # Always going to group by (parent_category,catorgory)
train_grouped = train_data.groupby(obs)
test_grouped = test_data.groupby(obs)

# To be made into columns of a DF
item_id_col = np.array([])
op_tag_col = np.array([])

# For each chunk - Assumption is chunks are the same and cover all items
count = 1

for group in train_grouped.groups:
	train = train_grouped.get_group(group) 
	test = test_grouped.get_group(group)


	# Train data
	ids = train["item_id"].values
	prices = train["price"].values

	prices +=  (1.0)**-10 # Account for 0
	prices = np.log(prices) # Try to make data less skewed
	
	# Train using MAD
	nBool = np.isnan(prices)
	prices_train = prices[~nBool]
	thresh = 3.5 # Internet suggestion (Seems to be standard)
	median = np.median(prices_train)
	median_absolute_deviation = np.median(np.abs(prices_train-median))


	# Outlier detection -> Train data
	item_id_col = np.append(item_id_col,ids)
	modified_z_scores = 0.6745*(prices - median) / median_absolute_deviation
	nanI = np.isnan(modified_z_scores) # Takes care of error 
	modified_z_scores[nanI] = 0.0
	oTag = (np.abs(modified_z_scores) > thresh).astype(float)
	oTag[nanI] = np.nan
	op_tag_col = np.append(op_tag_col, oTag)


	# Test data
	ids = test["item_id"].values
	prices = test["price"].values


	# Outlier detection -> Test data
	item_id_col = np.append(item_id_col,ids)
	prices += (1.0)**-10 # Account for 0
	prices = np.log(prices) # Try to make data less skewed
	modified_z_scores = 0.6745*(prices - median) / median_absolute_deviation
	nanI = np.isnan(modified_z_scores) # Takes care of error 
	modified_z_scores[nanI] = 0.0
	oTag = (np.abs(modified_z_scores) > thresh).astype(float)
	oTag[nanI] = np.nan
	op_tag_col = np.append(op_tag_col, oTag)


	print(str(count) + " of " + str(len(train_grouped.groups)))
	count += 1

# Keep only item_id & tag
combined = {'item_id': item_id_col, 'op_tag': op_tag_col}
combined = pd.DataFrame(data=combined)

print("Saving data to file!")

combined.to_csv("/mnt/c/Users/pl199/Downloads/Avito_Data/op_tag.csv", encoding="utf-8-sig",index=False) # BUG: Compression isnt working - Mnually compress to save space

print("Done w/ tagging!")

