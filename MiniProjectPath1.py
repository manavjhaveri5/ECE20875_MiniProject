import pandas
import numpy as np

'''
 The following is the starting code for path1 for data reading to make your first step easier.
 'dataset_1' is the clean data for path1.
'''

with open('behavior-performance.txt','r') as f:
    raw_data = [x.strip().split('\t') for x in f.readlines()]
df = pandas.DataFrame.from_records(raw_data[1:],columns=raw_data[0])
df['VidID']       = pandas.to_numeric(df['VidID'])
df['fracSpent']   = pandas.to_numeric(df['fracSpent'])
df['fracComp']    = pandas.to_numeric(df['fracComp'])
df['fracPlayed']  = pandas.to_numeric(df['fracPlayed'])
df['fracPaused']  = pandas.to_numeric(df['fracPaused'])
df['numPauses']   = pandas.to_numeric(df['numPauses'])
df['avgPBR']      = pandas.to_numeric(df['avgPBR'])
df['stdPBR']      = pandas.to_numeric(df['stdPBR'])
df['numRWs']      = pandas.to_numeric(df['numRWs'])
df['numFFs']      = pandas.to_numeric(df['numFFs'])
df['s']           = pandas.to_numeric(df['s'])
dataset_1 = df
#print(dataset_1[15620:25350].to_string()) #This line will print out the first 35 rows of your data

print('Length of data set before 5 video filter: ', len(dataset_1))

#Initializes lists and dictionary
dataset_1 = dataset_1.values.tolist()
clean_data = []
new_users = []
id_counts = {}

#counts how many times each ID shows up in dataset_1, assigns count to ID in dictionary
for row in dataset_1:
    user_id = row[0]
    if user_id in id_counts:
        id_counts[user_id] += 1
    else: 
        id_counts[user_id] = 1

#makes new list of user IDs only if user_ID has five or more counts in dictionary
for user_id, count in id_counts.items():
    if count >= 5:
        new_users.append(user_id)

#makes new list of lists based off of filtered User IDs
for row in dataset_1:
    if row[0] in new_users:
        clean_data.append(row)

print('Length of data set after 5 video filter: ', len(clean_data))

#converts to array
X = np.array(clean_data)

#Example of filtered data (may be a problem in the future that values are strings and not floats/integers)
print('First two rows of cleaned data: ', X[:2])



#train data

#validate data

#test data
