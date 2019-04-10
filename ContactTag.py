'''
Whattsapp & Viber "in" (Simplest versions of these as to detect spelling mistakes)
Email: Regex @ followed by .
Phone: Regex common number formats
'''

import pandas as pd
import zipfile
import re

# Read Data from files
zf = zipfile.ZipFile('/mnt/c/Users/pl199/Downloads/Avito_Data/train.csv.zip')
train_data = pd.read_csv(zf.open('train.csv'))

zf = zipfile.ZipFile('/mnt/c/Users/pl199/Downloads/Avito_Data/test.csv.zip')
test_data = pd.read_csv(zf.open('test.csv'))

print("Done loading data!")


# Combine relevant data
obs = ["item_id","description"] # Only need these params for now
combined = pd.concat([train_data[obs],test_data[obs]],ignore_index=True)

print("Done combining/grouping data!")


# Go through data
cTag = []

for desc in combined["description"].tolist():

	tag = False

	if type(desc) == str:
		low = desc.lower()

		# Whattsapp & Viber in (Simplest versions of these words for spelling mistakes)
		if "viber" in low or "wats" in low or "what" in low:
			tag = True

		# Regex for email
		elif re.search(r"(\S+@\S+\.\S+)", low) != None:
			tag = True

		# Regex for phone numbers (Found online - Seems alright)
		elif re.search(r"((8|\+7)-?)?\(?\d{3}\)?-?\d{1}-?\d{1}-?\d{1}-?\d{1}-?\d{1}-?\d{1}-?\d{1}", low.replace(".","-")) != None:
			tag = True

	cTag.append(tag)



combined = combined.assign(c_tag=cTag)

print("'c_tag' col created!")

combined = combined.drop(columns=obs[1]) # Delete useless column

print("Saving data to file!")

combined.to_csv("/mnt/c/Users/pl199/Downloads/Avito_Data/contact.csv", encoding="utf-8-sig", index=False) # BUG: Compression isnt working - Mnually compress to save space

print("Done w/ tagging!")