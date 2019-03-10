"""
Python 3.x
Titanic - Which type of people, given their attributes, were likely to survive the crash?
"""

__author__ = "Pruthvi Bheemarasetti"


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid') #for checking trends/skewing of data
#%matplotlib inline	#meant for viewing the seaborn plots instantaneously, easy with Jupyter notebook

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix	#primary metric for evaluating logistic regression learning algorithm



def main():

	#finding relative file path of the titanic.csv according to current working directory 
    dirpath = os.path.dirname(__file__)
    file_path = os.path.join(dirpath, "train_titanic.csv")
    print (file_path)

	#I/O error handler block
	try:

		#read the file into a pandas dataframe
		train = pd.read_csv(file_path)

		#exploratory analysis of the data set
		explore(train)

		#clean dataset of all missing/null values
		clean(train)

		#properly spltting dataset and training the logistic regression model
		train_and_evaluate(train)

	except IOError:
		print ("I/O Error: Could not read file")




def explore(train):

	train.info()	#check for missing data values

	# There are 891 entries in our train dataset with column name of the traveler information along with other information such as passenger 
	# class (Pclass), Fare, Ticket Cabin, and more. The Age column has 714 non-null whereas Cabin column has 204 non-null values. "Embarked" also has 
	# 889 non-null values. Data is clearly missing.

	#calculating percentage of data missing for the key columns/attributes
	pct_missing = round((train.isnull().sum())/(train.isnull().count())*100,1)
	pct_missing.sort_values(ascending=False).head()

	# Cabin column is missing 77.1% of its data
	# Age column is missing 19.9% of its data
	# Embarked column is missing 0.2% of its data

	sns.countplot(x='Survived', hue='Sex',data=train, palette='coolwarm')	#comparing number of people who survived vs. died (split by gender)
	# not many males survived as opposed to females

	sns.countplot(x='Embarked',data=train, hue='Survived')	#checking how many people survived based on port of embarkation
	# Southampton port was highest

	sns.countplot(x='Parch',data=train, hue='Survived')	#checking ratio of parents:children ("Parch") that survived 

	sns.distplot(train['Age'].dropna(),kde=False,color='green',bins=30)	#age distribution of all passengers with null values removed




def clean(train):

	#dropping this column, since almost of its data is missing
	train.drop('Cabin',axis=1,inplace=True)
	train.dropna(inplace=True)
	train.info()	#make sure there aren't anymore missing values

	
	#this chained method fills in missing values for the age column
	def impute_age(age_pclass): # passing age_pclass as ['Age', 'Pclass'] group of main dataset
	    
	    #passing age_pclass[0] which is 'Age' to variable 'Age'
	    Age = age_pclass[0]
	    
	    #passing age_pclass[2] which is 'Pclass' to variable 'Pclass'
	    Pclass = age_pclass[1]
	    
	    #applying condition based on the Age and filling in missing data respectively 
	    if pd.isnull(Age):

	        if Pclass == 1:
	            return 38

	        elif Pclass == 2:
	            return 30

	        else:
	            return 25

	    else:
	        return Age


	#applying custom method to the dataset        
    train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


    # from here on, we deal with categorical variables and merge back with original dataset

    #switching the column's non-numerical value into numberical values (0/1)
	sex = pd.get_dummies(train['Sex'])

	#switching the column's non-numberical values into numberical values (0/1), even though there are 3 ports
	embark = pd.get_dummies(train['Embarked'], drop_first=True)
	# if Q is 0, S is 0, the learning algorithm can predict C is 1 because at one time, a passenger can use one port only so Q, S or C, only one can be 1/True

	train = pd.concat([train,sex,embark],axis=1)	#since there are dummies now

	#however, we don't need these attributes in the dataset - their relationships might skew the model accuracy and key attribute relationships
	train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)




def train_and_evaluate(train):

	#splitting the feature columns to 'X' and target column to 'y'
	X = train.drop('Survived', axis = 1)
	y = train['Survived']

	X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

	#training the learning model
	logR = LogisticRegression()
	logR.fit(X_train, y_train)
	predictions = logR.predict(X_test)	#generating predicted set


	#evaluating the model
	print(classification_report(y_test,predictions))	#the classification report shows us the the precision, recall, f1-score and support cases for each class along with their averages
	print(confusion_matrix(y_test, predictions))	#the confusion matrix shows us all of the True Negatives/Positives and False Negatives/Positives of our learning algorithm





#main traceback/call
if __name__ == "__main__":
    start_time = time.time()
    main()
    #print("--- %s seconds ---" % (time.time() - start_time))   #to measure execution time
