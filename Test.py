'''
Random sanity checks for reference.
'''

import pandas as pd
import zipfile
import requests
import json

'''
Test 1: Only have 8 gb of RAM - Can I load everything?
'''
def test_1():
	zf = zipfile.ZipFile('/mnt/c/Users/pl199/Downloads/Avito_Data/train.csv.zip') 
	data = pd.read_csv(zf.open('train.csv'))

	print("Test 1")
	print(data.head())
	print("Success")


'''
Test 2: Are the <Insert col name here> for test data and training data the same?
'''
def test_2(cName):

	print("Test 2: " + cName)

	# Train data
	zf = zipfile.ZipFile('/mnt/c/Users/pl199/Downloads/Avito_Data/train.csv.zip') 
	data = pd.read_csv(zf.open('train.csv'))
	trainCol = data[cName].unique()
	print("Unique: " + str(len(trainCol)))
	print(trainCol)

	# Test data
	zf = zipfile.ZipFile('/mnt/c/Users/pl199/Downloads/Avito_Data/test.csv.zip') 
	data = pd.read_csv(zf.open('test.csv'))
	testCol = data[cName].unique()
	print("Unique: " + str(len(testCol)))
	print(testCol)

	if set(trainCol) == set(testCol):
		print("Same")
	else:
		print("Different")

'''
Test 3: Observe the json returned for a city in Russia.
'''
def test_3(city):

	print("Test 3: " + city)

	API_KEY = "AIzaSyAKDZyyGhWlFrNq9ELQyN0Bx2P5kS1h_9c"
	req = "https://maps.googleapis.com/maps/api/geocode/json?address=" + city + "&key=" + API_KEY
	response = requests.get(req)
	formatted = response.json()
	print(formatted)

'''
Test 4: Query and find region for a given city.
'''
def test_4(city):

	print("Test 4: " + city)

	# Train data
	zf = zipfile.ZipFile('/mnt/c/Users/pl199/Downloads/Avito_Data/train.csv.zip') 
	data = pd.read_csv(zf.open('train.csv'))
	print("From Train Data:")
	print(data[data['city'] == city]['region'])
	
	# Test data
	zf = zipfile.ZipFile('/mnt/c/Users/pl199/Downloads/Avito_Data/test.csv.zip') 
	data = pd.read_csv(zf.open('test.csv'))
	print("From Test Data:")
	print(data[data['city'] == city]['region'])

'''
Test 5: Understand which subcategories map to which categories.
'''
def test_5():
	print("Test 5")

	# Train data
	zf = zipfile.ZipFile('/mnt/c/Users/pl199/Downloads/Avito_Data/train.csv.zip') 
	train_data = pd.read_csv(zf.open('train.csv'))

	# Test data
	zf = zipfile.ZipFile('/mnt/c/Users/pl199/Downloads/Avito_Data/test.csv.zip') 
	test_data = pd.read_csv(zf.open('test.csv'))

	# Only params care about
	obs = ["parent_category_name","category_name"]

	combined = pd.concat([train_data[obs],test_data[obs]], ignore_index=True)

	pc_trans = pd.read_csv('/mnt/c/Users/pl199/Downloads/Avito_Data/parent_category_names.csv')
	c_trans = pd.read_csv('/mnt/c/Users/pl199/Downloads/Avito_Data/category_names.csv')
	
	# Want to translate to english
	combined['parent_category_name'] = combined['parent_category_name'].map(pc_trans.set_index('Russian')['English'])
	combined['category_name'] = combined['category_name'].map(c_trans.set_index('Russian')['English'])

	# Want unique pairs of categories & subcategories
	filtered = combined.groupby(obs).size()

	print(filtered)

'''
Test 6: How much time does it take to "filter" text(Take out special symbols and numbers)?
Note: Turns out Time command doesnt output to file using >> so had to manually copy & paste.
'''

def test_6():
	print("Test 6")

	# Train data
	zf = zipfile.ZipFile('/mnt/c/Users/pl199/Downloads/Avito_Data/train.csv.zip') 
	train_data = pd.read_csv(zf.open('train.csv'))

	obs = ["title","description","param_1","param_2","param_3"]

	replace = [":",",",";",".","!","?", # Punctuation
				"`","~","+","|","_","-","/","\\","#","$","@","%","^","&","*", # Other
				"(",")","[","]","{","}","<",">", # Brackets
				"0","1","2","3","4","5","6","7","8","9"] # Numbers

	# Please turn off prints if you're going to output time to file!
	for col in obs:
		count = 1
		for each in train_data[col]:
			#print(str(count) + " of " + str(train_data.size))
			
			if type(each) == str:
				for item in replace:
					each.replace(item," ")
				finished = each.lower()

			count += 1
		
		print("Col done!")



#test_1()
#test_2("region")
#test_2("city")
#test_3("Майский")
#test_4("Майский")
#test_3("Майский,Белгородская область")
#test_5()
#test_6()