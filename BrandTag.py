'''
English tag words (Not just brands)

Train tagger using training data ONLY!
'''

import pandas as pd
import zipfile
import re

# Need item_id for training data as clean contains both training and testing data
zf = zipfile.ZipFile('/mnt/c/Users/pl199/Downloads/Avito_Data/train.csv.zip')
for_id = pd.read_csv(zf.open('train.csv'))

# Read in cleaned text
zf = zipfile.ZipFile('/mnt/c/Users/pl199/Downloads/Avito_Data/clean.csv.zip')
clean_data = pd.read_csv(zf.open('clean.csv'))

train_id = for_id["item_id"].values
train_data = clean_data[clean_data["item_id"].isin(train_id)]
test_data = clean_data[~clean_data["item_id"].isin(train_id)]

print("Done loading data from file!")

# Get english words & frequncies
chunk_dicts = [] # Array of dictionaries (1 per group)
word_freq = {} # word -> frequency  

grouped = train_data.groupby(["parent_category_name","category_name"])
chunks = [grouped.get_group(x) for x in grouped.groups] # Each chunks is a grouping of data w/ same (parent_category_name, category_name)


for chunk in chunks:
	word_map = {} # word -> List of item_ids
	
	for _,row in chunk.iterrows():

		word_soup = row["merged"]
		word_salad = word_soup.split()
		
		# The same word can be used many time within the same item (But shouldn't count for more than 1)

		for word in set(word_salad): 

			# If string and english text (Everything should be lowercase)
			if type(word) is str and  re.match(r"([a-z]+)$", word) != None:
			
				if word not in word_map:
					word_map[word] = []
			
				if word not in word_freq: 
					word_freq[word] = 0
				
				word_map[word].append(row["item_id"])
				word_freq[word] += 1

	chunk_dicts.append(word_map)

print("Done organizing!")

# Calculate the important brand tags (Looking at categories)


print("Done w/ calculations!")

# Save to file
print("Saving to file!")




print("Tagging is complete!")