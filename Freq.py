'''
Just want to check things out.
Create a txt file with frequency of words.
Trying to understand the text data provided.
'''

import codecs
import pandas as pd
import zipfile

# Read in data
zf = zipfile.ZipFile('/mnt/c/Users/pl199/Downloads/Avito_Data/train.csv.zip') 
train_data = pd.read_csv(zf.open('train.csv'))

zf = zipfile.ZipFile('/mnt/c/Users/pl199/Downloads/Avito_Data/test.csv.zip') 
test_data = pd.read_csv(zf.open('test.csv'))

print("Data reading complete!")

# Combine into one table
obs = ["param_1","param_2","param_3","title","description"]
combined = pd.concat([train_data[obs],test_data[obs]], ignore_index=True) 

freq = {}

for col in obs:
	for dat in combined[col]:
		# Easier than regex
		words = str(dat).replace(";"," ").replace("."," ").replace(":"," ").replace(","," ").split()
		
		for w in words:
			if w not in freq:
				freq[w] = 1
			else:
				freq[w] += 1

	print("Done with: " + col)

print("Done w/ calculations!")

# Output to file
out = sorted(freq.items(), key=lambda x: x[1], reverse=True)

out_str = ""
for o in out:
	out_str += str(o) + "\n"

with codecs.open("freq.txt", "w", "utf-8-sig") as fp:
	fp.write(out_str)

print("Finished...")