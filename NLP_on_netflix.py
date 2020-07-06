
### The Start
'''
This project started from a thought which came to my mind when i first completely watched a standup comedy show on Netflix last week by Ali Wong and i absolutely loved it.
I wanted to get some insights about the factors which make Ali Wong's routines different from other standup comedians.
Secondly, I was also curious about the reason why i found this show very interesting given my past history of avoiding standup comedy shows.
The best tool which i had was natural language processing (NLP) which helped me satisfy my thought.
Let us have a look at how i performed a complete Exploratory Data Analysis(EDA).

### Data Gathering
"""

# Web scraping, pickle imports
import requests
from bs4 import BeautifulSoup
import pickle

# Scrapes transcript data from scrapsfromtheloft.com
def url_to_transcript(url):
    '''Returns transcript data specifically from scrapsfromtheloft.com.'''
    page = requests.get(url).text # Get all data from URL
    soup = BeautifulSoup(page, "lxml") # Read as an HTML document
    text = [p.text for p in soup.find(class_="post-content").find_all('p')] # Pull out all text from post-content
    print(url)
    return text

# URLs of transcripts in scope
urls = ['http://scrapsfromtheloft.com/2017/05/06/louis-ck-oh-my-god-full-transcript/',
        'http://scrapsfromtheloft.com/2017/04/11/dave-chappelle-age-spin-2017-full-transcript/',
        'http://scrapsfromtheloft.com/2018/03/15/ricky-gervais-humanity-transcript/',
        'http://scrapsfromtheloft.com/2017/08/07/bo-burnham-2013-full-transcript/',
        'http://scrapsfromtheloft.com/2017/05/24/bill-burr-im-sorry-feel-way-2014-full-transcript/',
        'http://scrapsfromtheloft.com/2017/04/21/jim-jefferies-bare-2014-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/02/john-mulaney-comeback-kid-2015-full-transcript/',
        'http://scrapsfromtheloft.com/2017/10/21/hasan-minhaj-homecoming-king-2017-full-transcript/',
        'http://scrapsfromtheloft.com/2017/09/19/ali-wong-baby-cobra-2016-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/03/anthony-jeselnik-thoughts-prayers-2015-full-transcript/',
        'http://scrapsfromtheloft.com/2018/03/03/mike-birbiglia-my-girlfriends-boyfriend-2013-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/19/joe-rogan-triggered-2016-full-transcript/']

# Comedian names
comedians = ['louis', 'dave', 'ricky', 'bo', 'bill', 'jim', 'john', 'hasan', 'ali', 'anthony', 'mike', 'joe']

# Actually request transcripts (takes a few minutes to run)
transcripts = [url_to_transcript(u) for u in urls]

# Pickle (save) files for later use
# We pickle because we want to avoid performing API calls on site multiple times

# Make a new directory to hold the text files
!mkdir transcripts # anything after ! will be executed by the system command-line (not by the Python kernel) 

for i, c in enumerate(comedians):
    with open("transcripts/" + c + ".txt", "wb") as file:
        pickle.dump(transcripts[i], file)

# Load pickled files
data = {}
for i, c in enumerate(comedians):
    with open("transcripts/" + c + ".txt", "rb") as file:
        data[c] = pickle.load(file)

# Double check to make sure data has been loaded properly
data.keys()

# Just looking at data again
data['louis'][:2]

data['louis'][3]

# Let's take a look at our data again
next(iter(data.keys()))

# Notice that our dictionary is currently in the format of key: comedian, value: list of text
next(iter(data.values()))

"""### Data Preprocessing"""

# In the dict above, the values are actually a list of text
# We are going to change this one giant string of text. New format will be {key: comedian, value: string}
def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text

# Combine it!
data_combined = {key: [combine_text(value)] for (key, value) in data.items()}

# We can either keep it in dictionary format or put it into a Pandas DataFrame
# I will put it into a Pandas DataFrame
import pandas as pd
pd.set_option('max_colwidth',150)

data_df = pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns = ['transcript']
data_df = data_df.sort_index()

# Print the Corpus
data_df

# Let's take a look at the transcript for Ali Wong
data_df.transcript.loc['ali']

"""**Creating a Corpus**"""

# Apply a first round of text cleaning techniques
import re
import string

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower() # converting to lowercase
    text = re.sub('\[.*?\]', '', text) # removing text which is in square brackets
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Remove square brackets
round1 = lambda x: clean_text_round1(x)

# Let's take a look at the updated text
data_clean = pd.DataFrame(data_df.transcript.apply(round1))
data_clean

# Apply a second round of cleaning
def clean_text_round2(text):
    '''Get rid of some additional quotation marks and newline text that was missed the first time around.'''
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

round2 = lambda x: clean_text_round2(x)

# Let's take a look at the updated text
data_clean = pd.DataFrame(data_clean.transcript.apply(round2))
data_clean

# Let's take a look at our dataframe
data_df

# Let's add the comedians' full names as well
full_names = ['Ali Wong', 'Anthony Jeselnik', 'Bill Burr', 'Bo Burnham', 'Dave Chappelle', 'Hasan Minhaj',
              'Jim Jefferies', 'Joe Rogan', 'John Mulaney', 'Louis C.K.', 'Mike Birbiglia', 'Ricky Gervais']

data_df['full_name'] = full_names
data_df

# Let us now pickle it for later use in our project
data_df.to_pickle("corpus.pkl")

"""**Creating a document-term matrix**"""

# We are going to create a document-term matrix using CountVectorizer, and exclude common English stop words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.transcript) # fit count vectorizor to our CLEAN transcript data

# Convert it to an array and label all the columns
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index

# Document-Term matrix
data_dtm

# Let's pickle it for later use
data_dtm.to_pickle("dtm.pkl")

# Let's also pickle the cleaned data (before we put it in document-term matrix format) and the CountVectorizer object
data_clean.to_pickle('data_clean.pkl')
pickle.dump(cv, open("cv.pkl", "wb"))

"""### Exploratory Data Analysis"""

import pandas as pd
data=pd.read_pickle('dtm.pkl')
# let us take a look at our document term matrix
data

# Here, we can see that the comedians are represented in rows and the individual words in columns
# Let us transpose our data as it is difficult to work on rows as compared to columns
data=data.transpose()
data

# Now, let us find the top 30 words of each comedian
top_dict={}
for c in data.columns:
  top_word=data[c].sort_values(ascending=False).head(30)
  top_dict[c]=list(zip(top_word.index,top_word.values))
top_dict

# Printing top 15 words for each comedian
for comedian,top_words in top_dict.items():
  print(comedian)
  print(', '.join([word for word,count in top_words[0:15]]))
  print('-----')

'''
At this point, we can clearly see that the most often used words have little meaning and can be classified as stop words. 
So, we move on and add these words to the list of stopwords
'''
# Look at the most common top words --> add them to the stop word list
from collections import Counter

# Creating a combined list of all comedians' list of top 30 words used

words=[]
for comedian in data.columns:
  top=[word for word,count in top_dict[comedian]]
  for t in top:
    words.append(t)

words

# Aggregate this list and identify the most common words along with how many comedian's routines they occur in
Counter(words).most_common()

"""The Criteria which we will be using to add these words to the list of stop words is:

If more than half of the comedians( i.e. 6) have that word in their top 30 list, then add the word to the list of stopwords.
"""

add_stop_words=[word for word,count in Counter(words).most_common() if count>6]
add_stop_words

# Let us now update our document-term matrix with the new list of words
from sklearn.feature_extraction import text # Contains the stop words list
from sklearn.feature_extraction.text import CountVectorizer
# Read in cleaned data from corpus 
data_clean=pd.read_pickle('data_clean.pkl')

# Add new stop words
stop_words=text.ENGLISH_STOP_WORDS.union(add_stop_words)

# RECREATE DOCUMENT-TERM MATRIX WHICH EXCLUDES OUR ADDITIONAL STOP WORDS
cv=CountVectorizer(stop_words=stop_words)
data_cv=cv.fit_transform(data_clean.transcript)
data_stop=pd.DataFrame(data_cv.toarray(),columns=cv.get_feature_names())
data_stop.index=data_clean.index

# Now our data has been modified and cleaned after removing the updated stopwords

# pickle the results for later use
import pickle
pickle.dump(cv,open("cv_stop.pkl",'wb'))
data_stop.to_pickle("dtm_stop.pkl")

# Let's make some word clouds out of the data we just processed
from wordcloud import WordCloud
wc=WordCloud(stopwords=stop_words,background_color='white',colormap="Dark2",max_font_size=150,random_state=42)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=[16,6]
full_names=['Ali Wong', 'Anthony Jeselnik', 'Bill Burr', 'Bo Burnham', 'Dave Chappelle', 'Hasan Minhaj',
              'Jim Jefferies', 'Joe Rogan', 'John Mulaney', 'Louis C.K.', 'Mike Birbiglia', 'Ricky Gervais']
for index,comedians in enumerate(data.columns):
  wc.generate(data_clean.transcript[comedian])
  plt.subplot(3,4,index+1)
  plt.imshow(wc,interpolation="bilinear")
  plt.axis("off")
  plt.title([full_names[index]],color='white')

plt.show()

"""**Number of Words**"""

# Find the number of unique words that each comedian uses

# Identify the non-zero items in the document-term matrix, meaning that the word occurs at least once
unique_list = []
for comedian in data.columns:
    uniques = data[comedian].to_numpy().nonzero()[0].size
    unique_list.append(uniques)

# Create a new dataframe that contains this unique word count
data_words = pd.DataFrame(list(zip(full_names, unique_list)), columns=['comedian', 'unique_words'])
data_unique_sort = data_words.sort_values(by='unique_words')
data_unique_sort

# Let us calculate words per minute of each comedian
# First, find the total number of words a comedian uses
total_list=[]
for comedian in data.columns:
  totals=sum(data[comedian])
  total_list.append(totals)

# Comedy special run times from IMDB (in minutes)
run_times = [60, 59, 80, 60, 67, 73, 77, 63, 62, 58, 76, 79]

# Let's add some columns to our dataframe
data_words["total_words"]=total_list
data_words["run_time"]=run_times
data_words["words_per_minute"]=data_words["total_words"]/data_words["run_time"]

# Let us sort our newly created dataframe by words_per_minute criteria
data_wpm_sort=data_words.sort_values(by='words_per_minute')
data_wpm_sort

# Plotting our findings

import numpy as np
y_pos=np.arange(len(data_words))
plt.subplot(1,2,1)
plt.barh(y_pos,data_unique_sort.unique_words,align='center')
plt.yticks(y_pos,data_unique_sort.comedian,color='white')
plt.title("Number of Unique Words",fontsize=20,color='white')
plt.xticks(color='white')
plt.subplot(1,2,2)
plt.barh(y_pos,data_wpm_sort.words_per_minute,align='center')
plt.yticks(y_pos,data_wpm_sort.comedian,color='white')
plt.xticks(color='white')
plt.title("Number of Words per Minute",fontsize=20,color='white')

plt.tight_layout()
plt.show()

"""Ricky Gervais has the highest vocabulary and Anthony Jeselnik has the least vocabulary.

Whereas Joe Rogan is the fastest in talking speed while Anthony Jeselnik being the slowest.

Again, there is nothing prevalent about Ali Wong which is of our interest here.

**PROFANITY**
"""

Counter(words).most_common()

# Let us isolate Profanity
data_bad_words=data.transpose()[['fucking','fuck','shit']]
data_profanity=pd.concat([data_bad_words.fucking + data_bad_words.fuck,data_bad_words.shit],axis=1)
data_profanity.columns=['f_word','s_word']
data_profanity

data_profanity.index

#Let us create a Scatter Plot out of our findings on profanity
plt.rcParams['figure.figsize']=[10,8]
for i, comedian in enumerate(data_profanity.index):
  x=data_profanity.f_word.loc[comedian]
  y=data_profanity.s_word.loc[comedian]
  plt.scatter(x,y,color='blue')
  plt.text(x+1.5,y+0.5,full_names[i],fontsize=10) # offset the label to avoid overlapping of names and dots
  plt.xlim(-5,155)

plt.title("Number of Bad Words Used in Routine",fontsize=20,color='white')
plt.xlabel("Number of F Bombs",fontsize=15,color='white')
plt.ylabel("Number of S Words",fontsize=15,color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.show()

"""### Sentiment Analysis"""

import pandas as pd
data=pd.read_pickle("corpus.pkl")
data

# Here, we'll be using the textblob library to compute sentiment( in terms of polarity) and subjectivity
# Range of polarity is from -1(most negative) to +1(most positive) and that of subjectivity is from 0( i.e. highly factual) to 1(i.e. highly opiniated)
from textblob import TextBlob
pol=lambda x: TextBlob(x).sentiment.polarity
sub=lambda x: TextBlob(x).sentiment.subjectivity
data['Polarity']=data['transcript'].apply(pol)
data['Subjectivity']=data['transcript'].apply(sub)

data

# Let us visualize our findings
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=[10,8]
for index,comedian in enumerate(data.index):
  x=data.Polarity.loc[comedian]
  y=data.Subjectivity.loc[comedian]
  plt.scatter(x,y,color='blue')
  plt.text(x+0.001,y+0.001,data['full_name'][index],fontsize=10)
  plt.xlim(-.01,.12)
plt.title('Sentiment Analysis', fontsize=20,color='white')
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15,color='white')
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15,color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.show()

# Sentiment of routine over time
# Split each transcript into 10 parts
import numpy as np
import math

def split_text(text,n=10):
  length=len(text)
  size=math.floor(length/n)
  start=np.arange(0,length,size)
  split_list=[]
  for piece in range(n):
    split_list.append(text[start[piece]:start[piece]+size])
  return split_list

# Let's take a look at our data before splitting
data

#Let's perform the splitting using the function defined above
list_pieces=[]
for t in data.transcript:
  list_pieces.append(split_text(t))
list_pieces

# Now, after we have done splitting of text, let us calculate polarity and subjectivity for all of these pieces
polarity_transcript=[]
for lp in list_pieces:
  polarity_piece=[]
  for pp in lp:
    polarity_piece.append(TextBlob(pp).sentiment.polarity)
  polarity_transcript.append(polarity_piece)
polarity_transcript

# Visualizing our findings for a routine
plt.plot(polarity_transcript[0])
plt.title("Name Of Comedian : "+data["full_name"].index[0],color='white')
plt.xlabel("Split part of Routine",color='white')
plt.ylabel("Polarity/Sentiment value",color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.show()

# Let us now visualize and compare the results of all the comedians w.r.t. to polarity and subjectivity
plt.rcParams['figure.figsize']=[16,12]
for index,comedian in enumerate(data.index):
  plt.subplot(3,4,index+1)
  plt.plot(polarity_transcript[index])
  plt.plot(np.arange(0,10),np.zeros(10))
  plt.title(data["full_name"][index],color='white')
  plt.ylim(-0.2,0.3)
plt.xticks(color='white')
plt.yticks(color='white')
plt.show()

"""### Topic Modeling  ( Using LDA )"""

# Let's start by reading our document-term matrix
import pandas as pd
import pickle

data=pd.read_pickle('dtm_stop.pkl')
data

import logging
logging.basicConfig(filename='lda_model.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import matutils,models
import scipy.sparse # sparse matrix format is used for gensim

tdm=data.transpose() # term-document matrix, which is simply, a transpose of document-term matrix
tdm

sparse_counts=scipy.sparse.csr_matrix(tdm)
corpus=matutils.Sparse2Corpus(sparse_counts)

# Gensim also requires a dictionary of all the terms and their respective location in the term-document matrix
cv = pickle.load(open("cv_stop.pkl", "rb")) # CountVectorizor creates dtm
id2word = dict((v, k) for k, v in cv.vocabulary_.items())

'''
Now that we have the corpus (term-document matrix) and id2word (dictionary of location: term), 
we're ready to train the LDA model. We need to specify two other parameters - the number of topics and the number of training passes. 
Let's start the number of topics at 2, see if the results make sense, and increase the number from there.
'''

"""**Topic Modeling with full text**"""

import numpy as np
# LDA num_topics=2,passes=10
lda=models.LdaModel(corpus=corpus,id2word=id2word,num_topics=2,passes=10,random_state=np.random.RandomState(seed=10))

for topic,topwords in lda.show_topics():
  print("Topic",topic,"\n",topwords,"\n")

# LDA num_topics=3,passes=10
lda=models.LdaModel(corpus=corpus,id2word=id2word,num_topics=3,passes=10,random_state=np.random.RandomState(seed=10))
for topic,topwords in lda.show_topics():
  print("Topic",topic,"\n",topwords,"\n")

# LDA num_topics=4,passes=10
lda=models.LdaModel(corpus=corpus,id2word=id2word,num_topics=4,passes=10,random_state=np.random.RandomState(seed=10))
for topic,topwords in lda.show_topics():
  print("Topic",topic,"\n",topwords,"\n")

# As we can clearly see, that our analysis is not working here as the algorithm is unable to identify clear topics which pertain to different words
# and there is clear overlap of words between topics.
# Therefore, we try another approach called as POS Tagging, i.e. Part of Speech Tagging and then identify different topics

"""**Topic modeling with nouns**"""

# Let us create a function to pull out nouns from a string of text
from nltk import pos_tag,word_tokenize
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
def nouns(text):
  '''Given a string of text, tokenize the text and pull out only the nouns.'''
  is_noun=lambda pos: pos[:2]=='NN'
  tokenized=word_tokenize(text)
  all_nouns=[word for (word,pos) in pos_tag(tokenized) if is_noun(pos)]
  return ' '.join(all_nouns) # convert list of nouns returned to a string

# Read in the cleaned data, before the CountVectorizer step
data_clean = pd.read_pickle('data_clean.pkl')

data_nouns=pd.DataFrame(data_clean.transcript.apply(nouns))
data_nouns

#Now, create a new Document-term matrix using only nouns
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

# Re-add the additional stop words since we are recreating the document-term matrix
add_stop_words = ['like', 'im', 'know', 'just', 'dont', 'thats', 'right', 'people',
                  'youre', 'got', 'gonna', 'time', 'think', 'yeah', 'said']
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

cv_nouns=CountVectorizer(stop_words=stop_words)
data_cv_nouns=cv_nouns.fit_transform(data_nouns.transcript)
data_dtm_nouns=pd.DataFrame(data_cv_nouns.toarray(),columns=cv_nouns.get_feature_names())
data_dtm_nouns.index=data_nouns.index
data_dtm_nouns

# Create the Gensim Corpus but this time using only nouns
corpus_nouns=matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtm_nouns.transpose()))

# Create the vocabulary dictionary for all the terms and their respective location
id2word_nouns = dict((v, k) for k, v in cv_nouns.vocabulary_.items())

# Now, we can create our LDA model starting with 2 topics and then we can increase one by one
# LDA: num_topics=2,passes=10
lda_nouns=models.LdaModel(corpus=corpus_nouns,id2word=id2word_nouns,num_topics=2,passes=10)
lda_nouns.print_topics()

# Let's try for topics=3
lda_nouns=models.LdaModel(corpus=corpus_nouns,id2word=id2word_nouns,passes=10,num_topics=3)
lda_nouns.print_topics()

# Let's try for topics=4
lda_nouns=models.LdaModel(corpus=corpus_nouns,id2word=id2word_nouns,passes=10,num_topics=4)
lda_nouns.print_topics()

"""Unfortunately tuning the hyper-parameters did not yield any meaningful topics. I also tried including verbs and retraining the model with nouns, adjectives and verbs but that didn’t help it either.

### Why Topic modeling isn't working
The model assumes that every chunk of text that we feed into it contains words that are somehow related. So starting with the right corpus is crucial. However, comedy specials are inherently dynamic in nature with no fixed topics in most streams. Since the subject matter is constantly switching throughout a comedian’s routine there usually isn’t one centralized topic.

### Summary of Insights
Although Topic Modeling was not successful, we still got something out of this experiment i.e. intelligent insights which answered my queries that came to my mind as a thought i had mentioned in the starting.

Insights:


1.   Ali had the highest s-word to f-word ratio. I personally do not mind using/listening to the s-word but i do not like hearing the f-word.
2.   Ali Wong tends to be more positive and less opinionated which is quite similar to my interests.
3.   Based on the above two insights, i can consider watching other comedians which have similar characteristics to Ali Wong.

    *   Comedians who do not say the f-word more often: Mike Birbiglia (no curse words at all) and John Mulaney.
    *   Comedians with a similar Sentiment pattern: Louis C.K. and Mike Birbiglia.

After drawing inference from the above three insights, we can conclude that i would probably love watching Mike Birbiglia's comedy as well.
"""