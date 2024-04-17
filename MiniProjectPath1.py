import pandas
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
#print(dataset_1[0:1].to_string())  #This line will print out the first 35 rows of your data

# QUESTION 1 

video_completion = df[df['fracComp'] >= 0.9] #Filtering for fracComp >= 0.9

video_counts = video_completion.groupby('userID')['VidID'].nunique() #Counts how many videos each unique user has completed

users_with_5_videos = video_counts[video_counts >= 5].index #Filters to those who have completed 5 videos

dataset_1_filtered = video_completion[video_completion['userID'].isin(users_with_5_videos)] #Final Filtered Dataframe

dataset_1_filteredArray = dataset_1_filtered.to_numpy() #Final Filtered array

#for i in range(len(dataset_1)):
 #   if dataset_1[i][0] in new_users:
#        clean_data.append(dataset_1[i])

#Create points for each row
def makePointList(data):
    """Creates a list of points from initialization data.
    #This function is outside Point Class.
    Args:
      data: A p-by-d numpy array.

    Returns:
      A list of length p containing d-dimensional Point objects, each Point's
      coordinates correspond to one row of data.
    """
    list = []
    # fill in
    for x in data:
        a = Point(x.tolist())
        list.append(a)


    return list

#train data

#validate data

#test data


#Main function
if __name__ == "__main__":

    points = makePointList(clean_data)
    print(points[0])
