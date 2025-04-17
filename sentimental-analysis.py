import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import re
import nltk
import nltk.corpus
import tensorflow as ts
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from collections import Counter 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from textblob import TextBlob





nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")



# Read the .csv file 
dataset=pd.read_csv("Instuments_Reviews.csv")
# Shape of dataset 
print(dataset.shape)
# Checking Null Values
print(dataset.isnull().sum())
# Filling Missing Values
dataset.reviewText.fillna(value=" ",inplace=True)
# Concatenate reviewText and summary Colums
dataset['reviews']=dataset['reviewText'] + " " + dataset["summary"]
dataset.drop(columns=["reviewText","summary"],axis=1,inplace=True)

print(dataset.describe(include="all"))

# Percentage of Rating Given from the Customer

dataset.overall.value_counts().plot(kind="pie",legend=False,autopct="%1.2f%%",fontsize=10,figsize=(8,8))
plt.title("percentage of Rating Given from the Customer",loc="center")
plt.show()

# Labeling Produsct Based on Rating Given

def Labelling(Rows):
    if(Rows["overall"]>3.0):
        Label="Positive"
    elif(Rows["overall"]<3.0):
        Label="Negative"
    else:
        Label="Neutral"
    return Label

dataset["sentiment"]=dataset.apply(Labelling,axis=1)
dataset["sentiment"].value_counts().plot(kind="bar",color="blue")
plt.title("Amount of Each Sentiments Based on Rating Given",loc="center",fontsize=15,color="red",pad=25)
plt.xlabel("Sentiments",color="green",fontsize=10,labelpad=15)
plt.xticks(rotation=0)
plt.ylabel("Amount of Sentiments",color="green", fontsize=10, labelpad=15)
plt.show()


# Text Cleaning

def Text_Cleaning(Text):
    Text=Text.lower()
    # Cleaning punctions in the text 
    punc=str.maketrans(string.punctuation,' '*len(string.punctuation))
    Text=Text.translate(punc)
    # Removing numbers in the Text 
    Text=re.sub(r'\d+','',Text)
    # Removing possible links 
    Text=re.sub(r'https?://\S+|www\.\S+', '', Text)
    # Deleting new lines
    Text=re.sub('\n','',Text)
    return Text

# Text Processing

Stopwords=set(nltk.corpus.stopwords.words('english'))-set(['not'])
def Text_Processing(Text):
    processed_Text=list()
    Lemmatizer=WordNetLemmatizer()
    # Tokens of Words
    Tokens=nltk.word_tokenize(Text)

    # Removing Stopwords and Lemmatizer Words
    # To reduce noises in our dataset, also to keep it simple and still
    # Powerful,we will only omit the word 'not' from the list of Stopwords

    for word in Tokens:
        if word not in Stopwords:
            processed_Text.append(Lemmatizer.lemmatize(word))
    
    return(" ".join(processed_Text))


dataset['reviews']=dataset['reviews'].apply(lambda Text: Text_Cleaning(Text))
dataset['reviews']=dataset['reviews'].apply(lambda Text: Text_Processing(Text))
print(dataset.head(n=10))

# Polarity Review Length and WordCounts

dataset['polarity']=dataset["reviews"].map(lambda Text: TextBlob(Text).sentiment.polarity)
dataset["polarity"].plot(kind='hist',bins=40,edgecolor='blue',linewidth=1,color='orange',figsize=(10,5))
plt.title("Polarity Score in Reviews",color='blue',pad=20)
plt.xlabel("Polarity",labelpad=15,color='red')
plt.ylabel("Amount of Reviews",labelpad=20,color='green')
plt.show()

dataset['length']=dataset['reviews'].astype(str).apply(len)
dataset['length'].plot(kind='hist',bins=40,edgecolor='blue',linewidth=1,color='orange',figsize=(10,5))
plt.title("Length of Reviews", color='blue',pad=20)
plt.xlabel("Length",labelpad=15,color='red')
plt.ylabel("Amount of Reviews",labelpad=20,color="green")
plt.show()

dataset['word_counts']=dataset['reviews'].apply(lambda x:len(str(x).split()))
dataset['word_counts'].plot(kind='hist',bins=40,edgecolor='blue',linewidth=1,color='orange',figsize=(10,5))
plt.title("Word Counts in Reviews", color='blue',pad=20)
plt.xlabel("Word Counts",labelpad=15,color='red')
plt.ylabel("Amount of Reviews",labelpad=20,color="green")
plt.show()

# N-Gram Analysis

# def Gram_Analysis(Corpus, Gram, N):
#     Vectorizer = CountVectorizer(stop_words=Stopwords, ngram_range=(Gram, Gram))
#     # N-Gram matrix
#     ngrams = Vectorizer.fit_transform(Corpus)
    
#     # N-Gram Frequency
#     Count = ngrams.sum(axis=0)
    
#     # List of Words
#     words = [(word, Count[0, idx]) for word, idx in Vectorizer.vocabulary_.items()]
    
#     # Sort Descending with key = Count
#     words = sorted(words, key=lambda x: x[1], reverse=True)
    
#     return words[:N]

# # Filter the DataFrame Based on Sentiments

# # Ensure that the sentiment column exists and has values "Positive", "Neutral", "Negative"
# Positive = dataset[dataset["sentiment"] == "Positive"].dropna()
# Neutral = dataset[dataset["sentiment"] == 'Neutral'].dropna()
# Negative = dataset[dataset["sentiment"] == "Negative"].dropna()


# # Unigram of Reviews Based on Sentiments
# # Unigram of Reviews Based on Positive Sentiments
# words = Gram_Analysis(Positive["reviews"], 1, 20)
# Unigram = pd.DataFrame(words, columns=["Words", "Counts"])

# # Visualization
# Unigram.groupby("Words").sum()["Counts"].sort_values().plot(kind='barh', color='green', figsize=(10, 5))
# plt.title("Unigram of Reviews with Positive Sentiments", loc='center', fontsize=15, color='blue', pad=25)
# plt.xlabel("Total Counts", color='magenta', fontsize=10, labelpad=15)
# plt.xticks(rotation=0)
# plt.ylabel("Top Words", color='cyan', fontsize=10, labelpad=15)
# plt.show()

# All remaining is in screen shot

Columns=["reviewerID","asin","reviewerName","helpful","unixReviewTime","reviewTime","polarity","length","word_counts","overall"]
dataset.drop(columns=Columns,axis=1,inplace=True)

print(dataset.head())

# **************************

Encoder=LabelEncoder()
dataset["sentiment"]=Encoder.fit_transform(dataset["sentiment"])
dataset["sentiment"].value_counts()

# TF-IDF Vectorizer

# Defining our vectorizer with total words of 5000 and with bigram model
TF_IDF=TfidfVectorizer(max_features=5000,ngram_range=(2,2))

# Fitting and transforming our reviews into a matrix of weighed words
# This will be our independent features 
X=TF_IDF.fit_transform(dataset["reviews"])

# Check our matrix shape
print(X.shape) 

# Declaring our target variable
y=dataset["sentiment"]

"""There are many ways to do resampling to an imbalanced dataset,such as SMOTE and Bootstrap Method .We will use SMOTE(Synthetic Minority
 Oversampling Technique )that will randomly generate new replicates of our undersampling data to balance our dataset"""
Balancer=SMOTE(random_state=42)
x_final,y_final=Balancer.fit_resample(X,y)
Counter(y_final)

# Splitting our Dataset

X_train,X_test,y_train,y_test,=train_test_split(x_final,y_final,test_size=0.25,random_state=42)
#We splitted our dataset into 75:25 portion respectively for training and test set

#******************************
# Model Selection and Evaluation
# *****************************

# Model Building

DTree=DecisionTreeClassifier()
LogReg=LogisticRegression()
SVC=SVC()
RForest=RandomForestClassifier()
Bayes=BernoulliNB()
KNN=KNeighborsClassifier()

Model=[DTree,LogReg,SVC,RForest,Bayes,KNN]
Model_Dict={0:"Decision Tree",1:"Logistic Regression",2:"SVC", 3:"Random Forest",4:"Naive Bayes",5:"K-Neighbours"}

for i, model in enumerate(Model):
    print("{} Test Accuracy: {}".format(Model_Dict[i],cross_val_score(model,X,y,cv=10,scoring="accuracy").mean()))


# Hyperparameter Tuning

Param={"C":np.logspace(-4,4,50),"penalty":['l1','l2']}
grid_search = GridSearchCV(estimator=LogisticRegression(random_state=42),param_grid=Param,scoring="accuracy",cv=10,n_jobs=-1)

grid_search.fit(X_train,y_train)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_

print("Best Accuracy : {:.2f}%".format(best_accuracy*100))
print("Best Prameters:",best_parameters)

Classifier=LogisticRegression(random_state=42,C=6866.488450042998,penalty='l2')
Classifier.fit(X_train,y_train)

Prediction =Classifier.predict(X_test)
# Metrics
print(accuracy_score(y_test,Prediction))

# Confusion matrix
ConfusionMatrix=confusion_matrix(y_test,Prediction)

# Visualizing Our Confusion Matrix
def plot_cm(cm,classes,title,normalized=False,cmap=plt.cm.Blues):
    
    plt.imshow(cm,interpolation="nearest",cmap=cmap)
    plt.title(title,pad=20)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes)
    plt.yticks(tick_marks,classes)

    if normalized:
        cm=cm.astype('float') / cm.sum(axis=1)[: np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print("Unnormalized Confusion Matrix")

    threshold=cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,cm[i,j],horizontalalignment="center",color="white" if cm[i,j]>threshold else "black")
    
    plt.tight_layout()
    plt.xlabel("Predicted Label",labelpad=20)
    plt.ylabel("Real Label",labelpad=20)

plot_cm(ConfusionMatrix,classes=["Positive,Neutral","Negative"],title="Confusion Matrix of Sentiment Analysis")
 
# Classification Scores
print(classification_report(y_test,Prediction))







