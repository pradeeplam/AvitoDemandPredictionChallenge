'''
Go through test/train.csv and create clean.csv file.
File will be used to collect descriptive 'tag' words for data.
Looking for text 'tags'. Don't worry about numbers for now.
'''

import pandas as pd
import zipfile
import string
from nltk.corpus import stopwords
import gc

# Read in data
zf = zipfile.ZipFile("/mnt/c/Users/pl199/Downloads/Avito_Data/train.csv.zip") 
train_data = pd.read_csv(zf.open("train.csv"), index_col="item_id")

zf = zipfile.ZipFile("/mnt/c/Users/pl199/Downloads/Avito_Data/test.csv.zip") 
test_data = pd.read_csv(zf.open("test.csv"), index_col="item_id")

print("Data reading complete")

# Combine into one table
train_data.drop(["deal_probability"], axis=1, inplace=True)
combined = train_data.append(test_data)

del train_data,test_data
gc.collect()

# Cleaning - Remember: Garbage in = Garbage out

text_obs = ["title","description","param_1","param_2","param_3"]

combined[text_obs] = combined[text_obs].astype(str)
combined[text_obs].fillna("")
combined["merged_dirty"] = combined["title"] + " " + combined["description"] + " " + combined["param_1"] + " " + combined["param_2"] + " " + combined["param_3"]
combined["merged"] = combined["merged_dirty"]
result = combined[["merged","merged_dirty"]]

del combined
gc.collect()

# Convert text to lower
result["merged"] = result["merged"].str.lower()
print("Done converting text to lower")

# Tale out punctuation, digits, and some weird characters
delete = string.punctuation + string.digits + "“”¨«»®´·º½¾¿¡§£₤‘’"
table = str.maketrans(dict.fromkeys(delete," "))
result["merged"] = result["merged"].apply(lambda text: text.translate(table))
print("Done taking out punctuation")

# Delete stopwords
stop = stopwords.words("russian") # Get list of fluff words
result["merged"] = result["merged"].apply(lambda text: " ".join([word for word in text.split() if word not in stop]))
print("Done deleting stopwords")

# Save to file
result.to_csv("/mnt/c/Users/pl199/Downloads/Avito_Data/clean.csv", encoding="utf-8-sig") # BUG: Compression isnt working - Mnually compress to save space