'''
1. Count total characters before clean [ON: data]
2. Count number of words [ON: clean_data]
3. Count number of unique words [ON: clean_data]
4. Calc % of unique words to words 
5. Count number of char after clean [ON: clean_data]
6. Calc % char after clean to char 
7. Count "professional punctuation"(.,;) [ON: data]
8. Calc "professional punctuation" to total char ratio
9. Count number of punctuation [ON: data]
'''

import pandas as pd
import zipfile
import gc
import string

# Load data
zf = zipfile.ZipFile("/mnt/c/Users/pl199/Downloads/Avito_Data/clean.csv.zip") 
result = pd.read_csv(zf.open("clean.csv"), index_col="item_id")
print("Done loading data")

result["char_count_orig"] = result["merged_dirty"].apply(lambda text: len([c for c in text if c != " "])) # 1
print("Done w/ char_count_orig") 

result["num_words"] = result["merged"].apply(lambda text: len(text.split())) # 2
print("Done w/ num_words") 

result["num_unique_words"] = result["merged"].apply(lambda text: len(set(text.split()))) # 3
print("Done w/ num_unique_words") 

result["words_vs_unique"] = result["num_unique_words"] / result["num_words"] * 100 # 4
print("Done w/ words_vs_unique") 

result["char_count_clean"] = result["merged"].apply(lambda text: len([c for c in text if c != " "])) # 5
print("Done w/ char_count_clean") 

result["c_vs_u_char"] = result["char_count_clean"] / result["char_count_orig"] * 100 # 6
print("Done w/ c_vs_u_char") 

result["num_pro_punc"] = result['merged_dirty'].apply(lambda text: sum([str.count(text, c) for c in [",",".",";"]])) # 7
print("Done w/ num_po_punc")

result["pp_vs_char"] = result["num_pro_punc"] / result["char_count_orig"] * 100 # 8
print("Done w/ pp_vs_char")

result["num_punc"] = result["merged_dirty"].apply(lambda text: len([c for c in text if c in string.punctuation])) # 9 
print("Done w/ num_punc")

print("Saving data!")

result.drop(["merged","merged_dirty"], axis=1,inplace=True) # Don't need these anymore 
result.to_csv("/mnt/c/Users/pl199/Downloads/Avito_Data/meta.csv", encoding="utf-8-sig") # BUG: Compression isnt working - Mnually compress to save space