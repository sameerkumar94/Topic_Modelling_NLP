'''

            \\SCRIPTERS//
            *************
            [1] Sameer Kumar
                [2] Parastou Yaghmai
                    [3] Ekaagar Singh Hara

    Authorship/Authority of different files are as folllowing :-
    ************************************************************
        1. Source Code :: Sameer Kumar
        2. Pre-processing Code (R) :: Parastou Yaghmai
        3. Baselines :: Parastou Yaghmai
        4. Human Annotation :: Parastou Yaghmai, Ekaagar Singh Hara, Sameer Kumar
        5. Origins :: Ekaagar Singh Hara
        6. Installs :: Sameer Kumar & Parastou Yaghmai
        7. Word Cloud :: Parastou Yaghmai & Ekaagar Singh Hara


    Program Title               :: Topic Modelling & Visualization on Election Tweet
    *************
    Problem                     :: To identify the purpose & Topic of Millions of tweet which were released at the time of 2016 American Elections.
    *******                        We identify the topics which were talked about in the huge set of tweets sent from 946 handles. It will help us understand 
                                   the whether there were any influence by these tweets into the 2016 US election.

    Program Run                 :: The program is executed in Python with importing the Semi-clean dataset created in R. 
    Original File Name          :: IRAA.csv
    Semi-cleaned File Name      :: Semicleantweet.csv
    Cleaned File Name           :: cleaneddata1.csv

    Steps involved in execution of the program & knowledge mining according to it :-
    ********************************************************************************
        [i] The dataset is imported as csv file and attribute dropping is implemented.
        
       [ii] Data pre-processing is done on the imported dataset thereby removing punctuations. The texts are converted into
            lower cases. Stopwords are removed using english stopword library. Lemmatization and Stemming worked on to convert the
            words into their canonical form. Frequent words are removed in R script by idetifying the stopwords separately.
            
      [iii] The cleaned and processed data gets converted into a dictionary. Tokenized dataset comes into play. Dictionary
            consists of every token(word) wwith its unique ID & the count of appearing in the document.
            
       [iv] LDA modelling technique using Uni-grams is applied onto the processed dictionary Bag of Words. The Optimum number 
            of appropriate topics to be modelled is calculated by calculating the coherence scores for each model processed from 1 to 15.
            The value is selected based on the plot of coherence score and number of topics. The optimum vaue is elected based on the drop &
            increase in the graph along with good coherence score. The model is selected with topic# = 14 and saved into a text file for later use.
            The LDA model selected is also saved in order to use the model for any other imports being performed on the data set.
            
        [v] Furthermore, the model is applied back to the tweet set to map each topic to the most probable topic number(topic#). The resulting 
            auto-mated mapping/annotatons are exported to a csv file.
            
       [vi] The annotated csv file is compared with the 3 different human/manual annotated files and accuracy is calculated for approx 300-350 tweets.
            Three diffferent manual annotations were done for minimizing the error percentage and to calculated precised accuracy. 

Note* :: 
********
    Human annotation used for accuracy calculation of the model built as the paradign used is unsupervised modelling technique. 
    300-400 approx. tweets were compared based on human and auto-mated annotations and accuracy was calculated to be 85%. 
'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
Packages/Module:
****************

        Modules/Packages list required for the installation & execution of various 
        python functions for carrying out the necessary programs. These include nltk,
        gensim, panda, numpy etc.
        
*Note* :: Some of the modules/packages are listed as not used because
          the commands are commented due to limited usage or used
          only at special times and places in order to check for 
          particualr result.
          
'''

import pandas as pd
import re
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from nltk.collocations import *
from nltk import FreqDist
import string
from sklearn.feature_extraction.text import CountVectorizer
import csv
import gensim
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
Data Input and Attribute Removal:
*********************************

        This section deals with the read IN function for the code.
        Input CSV file is read here and two un-necessary columns are 
        being dropped so that only the required attribute is available 
        for running the appropriate model. The file is imported from R 
        therefore it automatically saves extra columns at the time of 
        writing it to CSV. The dataset used for final results consisted
        of 1 Million tweets to process. 
        
'''

data=pd.read_csv("C:/Users/csame/Desktop/AIT 690/russian-troll-tweets/Semicleantweet.csv",header=None,encoding='Latin-1')
data.drop(data.columns[[0]], axis=1, inplace=True)
data.drop(data.columns[[0]], axis=1, inplace=True)
#data.head()
#print(data)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
Data Cleaning & Stopwords Removal:
**********************************

        Word lemmatizer imported and used. Removal of stopwords from the dataset
        thereby comparing it with the english stopwords bag of words which contains
        the most frequent stopwords that need to be removed. We compare the attribute
        with the english list of words and remove all those that gives a match.
        Punctuations are also removed from the dataset so that only the required
        words to model them into BoW(Bag of Words) remain and unnecessary characters, words, 
        punctuations are removed completely.
        
'''

ps=nltk.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
data.columns = ['content']
data_needed=pd.DataFrame({'content':data['content']})
def rem_stopwords(text):
    text= "".join([word for word in text if word not in string.punctuation])
    tokens= re.split('\W+',text)
    text=" ".join([ps.lemmatize(word) for word in tokens if word not in stopwords])
    return text
data_needed['cleaned_content'] = data_needed['content'].apply(lambda x: rem_stopwords(str(x).lower()))#
print(data_needed['cleaned_content'])
data_clean = data_needed[['cleaned_content']].copy()
data_clean.to_csv('C:/Users/csame/Desktop/AIT 690/russian-troll-tweets/cleaneddata1.csv', encoding = 'Latin-1')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
Data Lemmatizing, Stemming & Corpus Formation:
**********************************************   

    Steps invloved to read, convert and preprocess the file and to 
    Generate Dictionary of the processed documents:
    
    1.  The data file written to csv after cleaning/removal of stopwords
        is imported here again. The file contains a lot of special
        characters and symbols which were important for visualization
        purpose but were read as float. Due to this, the pre-processing map
        was not able to execute without errors, which termed the file as float 
        and not string for operations to be performed on. Therefore, the file 
        was read again as CSV but with one modification which is, the complete
        file is being converted into String format and then read IN for further
        execution as all the characters and words are considered to be String.

    2.  Document of the dataframe is created and document is processed 
        with lemmatizer and stemmer thereby cleaning the data a bit more 
        for trimming it for similar words and so on. Snowball Stemmer is
        utilized for the stemming of the document.
    
    3.  After the Stemming and Lemmatizing of the document,
        the processed document is converted into a dictionary
        using the gensim library/module. 
    
    4.  Applying a few filters on the dictionary and converting
        it into "Bag Of Words" format and saving that for future
        model building. Doc2bow method has been applied to carry out the 
        procedure. 
        
'''

data= []
with open('C:/Users/csame/Desktop/AIT 690/russian-troll-tweets/cleaneddata1.csv','r+') as reader_file:
    for read_ln in reader_file.read().split('\n')[:-1]:
        x = []
        for a in read_ln.split(','):
            try:
                x.append(int(a))
            except ValueError as ve:
                x.append(a.encode('utf-8').decode('utf-8'))
        data.append(x)
data = pd.DataFrame(data[1:],columns = ['index','cleaned_content'])

#print(data.dtypes)
#print(data)

data1 = data[['cleaned_content']]
data1['index'] = data1.index
documents = data1

#print(documents)

#len(documents)

np.random.seed(2018)
nltk.download('wordnet')
print(WordNetLemmatizer().lemmatize('went', pos='v'))

stemmer = SnowballStemmer('english')
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


processed_docs = (documents['cleaned_content']).map(preprocess)
processed_docs[:15]

dictionary = gensim.corpora.Dictionary(processed_docs)

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
Rough/Verification Code:
************************

        For verifying the execution of various commands performed 
        on the data set, there are usually few sets of commands to
        execute and check if the output is correct or verifying one
        small part of the dataset and code it to give the result.
        These are the few executale commands which were used in the code 
        to check for few results based upon our need at particular times.
        These are being commented so that it doesnt affect the running of
        the original code. 
        
'''

#doc_sample = documents[(documents['index']) == 123].values[0][0]
#print('original document: ')

#words = []

#for word in doc_sample.split(' '):
#    words.append(word)
#print(words)

#print('\n\n tokenized and lemmatized document: ')
#print(preprocess(doc_sample))

#count = 0
#for k, v in dictionary.iteritems():
#    print(k, v)
#    count += 1
#    if count > 40:
#        break

#bow_doc_145900 = bow_corpus[145900]
#
#for i in range(len(bow_doc_145900)):
#    print("Word {} (\"{}\") appears {} time.".format(bow_doc_145900[i][0], 
#                                                     dictionary[bow_doc_145900[i][0]], 
#                                                     bow_doc_145900[i][1]))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
Latent Dirichlet Allocation :: Topic Modelling
**********************************************

Step taken into consideration for carrying out LDA modelling
on the obtained document file:
    
    1. Choosing the size/limit of the model running or topic
       selection list thereby setting the upper and lower
       parameter of execution of the model run. The code
       executes for 14 values.
    
    2. Generating Different models based on the limits given
       and formulating several Coherence values for them. 
       14 different sets of Topic sets are created from
       running the model loop.
    
    3. Comparing the obtained coherence values in order to
       select the most appropriate model and the topics
       modelled is the next task. For that purpose, the
       plot between coherence value and model number is 
       obtained. The graph depicts value of coherence and model
       number and exhibits a technique to choose the optimum K.
       
    4. Checking the obtained result of the optimum model and
       saving it to .txt* format for future importing if used
       anywhere else for the same tweet. 
       
    5. Writing and saving the Topics formulated into a text file
       to use it futher for manual annotations of the tweet set and 
       to use it anytime later. 

Summary:
*******
    Formulating the model based on conherence score and selecting optimum value of Topic number based on the plot
    between coherence score and topics. Saving the Apporpriate LDA model and topic file for future uses. 
    
'''

limit=15 
start=1 
step=1
coherence_values = []
model_list = []
for num_topics in range(start, limit, step):
    #lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=2, workers=2)
    lda_model = LdaModel(corpus=bow_corpus, num_topics=num_topics, id2word=dictionary)
    model_list.append(lda_model)
    coherencemodel = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
    coherence_values.append(coherencemodel.get_coherence())
    
#print(coherence_values)                               #Cross-Verifying the Coherence value
#print(*coherence_values, sep = "\n") 

limit=15; start=1; step=1;
x = range(start, limit, step)
plt.xlabel("Topic_Count",)
plt.ylabel("Model_coherence_value")
plt.plot(x,coherence_values,'red')

#model_list[13]
#model_list[13].save("C:/Users/csame/Desktop/AIT 690/russian-troll-tweets/Optimum_LDA_Model.txt") 

#print(model_list[13].show_topics())



topics = lda_model.print_topics(num_words=10)           #Verifying and Printing the Optimum K-values
for i,topic in topics:
    print("Topic Number:", i , "(", topic, ")")
    print("\n")
    
#with open('C:/Users/csame/Desktop/AIT 690/russian-troll-tweets/TopicsSelected.txt', 'w') as f:
#    for item in model_list[13].show_topics():
#        f.write("%s\n" % item)
#model_list[13].show_topics().("C:/Users/csame/Desktop/AIT 690/russian-troll-tweets/TopicsSelected.csv")
 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
Mapping of topics & Sorting them to dataframe:
**********************************************

        The topics are formulated and used to annotate the original tweet
        file for topic modelling. The topic attribute or dataframe is 
        obtained and then mapped onto the original file for accuracy generation
        via comparison with manual annotations. 
        The Resulting dataframe is saved and written to a csv for carrying out the
        mapping of manual annotations and later formualizing accuracy of the model.

'''
#Sorting and assigning the Topic#
lda_new=lda_model[bow_corpus]													
i=0
sorted_lda = []
while i<len(lda_new): 
    sorted_lda.append(sorted(lda_new[i], key=lambda k:-k[1])[:1] )							
    i = i+1
Mapped_set = pd.DataFrame(sorted_lda)

#print(sorted_lda)      
#print(Mapped_set)   
                  
#Mapping of the Topic# column                    
Mapped_set.columns = ['Topic_#']
Mapped_set['Topic_#'] = Mapped_set['Topic_#'].astype(str)                   
Mapped_set["Topic_#"]= Mapped_set['Topic_#'].str.replace(r'(', '').str.strip()   
Mapped_set["Topic_#"]= Mapped_set['Topic_#'].str.replace(r')', '').str.strip()
Mapped_set['Topic_#'] = Mapped_set['Topic_#'].str.split(',').str[0]           
Mapping_text = pd.read_csv('C:/Users/csame/Desktop/AIT 690/russian-troll-tweets/lastcleantweet.csv', delimiter=',', encoding = 'latin-1')
result=Mapping_text.join(Mapped_set)

#Writing to CSV
result.to_csv('C:/Users/csame/Desktop/AIT 690/russian-troll-tweets/TopicModelledFile.csv')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



