'''
Translate the parent_category_name and category_name for test.csv and train.csv.
Output to files.
'''

import pandas as pd
import zipfile
import textblob

# Read in data
zf = zipfile.ZipFile('/mnt/c/Users/pl199/Downloads/Avito_Data/train.csv.zip') 
train_data = pd.read_csv(zf.open('train.csv'))

zf = zipfile.ZipFile('/mnt/c/Users/pl199/Downloads/Avito_Data/test.csv.zip') 
test_data = pd.read_csv(zf.open('test.csv'))

print("Data reading complete!")

# Translate unique
parent_category_names = pd.concat([train_data["parent_category_name"],test_data["parent_category_name"]]).unique()
category_names = pd.concat([train_data["category_name"],test_data["category_name"]]).unique()

parent_category_names_t = {"Russian":[],"English":[]}
category_names_t = {"Russian":[],"English":[]}


for dat in parent_category_names:
    print("Translating: " + dat)
    try:
        parent_category_names_t["Russian"].append(dat)
        parent_category_names_t["English"].append(textblob.TextBlob(dat).translate(to="en"))
    except:
        print("Error! Couldn't translate!")
        print("Exiting now...")
        exit()

for dat in category_names:
    print("Translating: " + dat)
    try:
        category_names_t["Russian"].append(dat)
        category_names_t["English"].append(textblob.TextBlob(dat).translate(to="en"))
    except:
        print("Error! Couldn't translate!")
        print("Exiting now...")
        exit()

print("Translation complete!")

# Export
pd.DataFrame(data=parent_category_names_t).to_csv("/mnt/c/Users/pl199/Downloads/Avito_Data/parent_category_names.csv",index=False)
pd.DataFrame(data=category_names_t).to_csv("/mnt/c/Users/pl199/Downloads/Avito_Data/category_names.csv",index=False)

print("Export complete!")