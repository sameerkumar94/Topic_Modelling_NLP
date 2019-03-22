
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

Data Set URL of our project: https://www.kaggle.com/fivethirtyeight/russian-troll-tweets/version/1/

Github URL of our Project: https://github.com/ekaagar/Twitter-Topics/
