# %% [markdown]
# <div class="alert alert-block alert-danger">
# 
# # FIT5196 Task 1 in Assessment 1
# #### Student Name: Michael Xie, Weiliang Huang
# #### Student ID: 31482919, 34370862
# 
# Date: 8/24/2024
# 
# 
# Environment: Python 3.11.9
# 
# Libraries used:
# * os (for interacting with the operating system, included in Python xxxx) 
# * pandas 1.1.0 (for dataframe, installed and imported) 
# * multiprocessing (for performing processes on multi cores, included in Python 3.6.9 package) 
# * itertools (for performing operations on iterables)
# * nltk 3.5 (Natural Language Toolkit, installed and imported)
# * nltk.tokenize (for tokenization, installed and imported)
# * nltk.stem (for stemming the tokens, installed and imported)
# 
#     </div>

# %% [markdown]
# <div class="alert alert-block alert-info">
#     
# ## Table of Contents
# 
# </div>
# 
# [1. Introduction](#Intro) <br>
# [2. Importing Libraries](#libs) <br>
# [3. Examining Input File](#examine) <br>
# [4. Loading and Parsing Files](#load) <br>
# $\;\;\;\;$[4.1. Tokenization](#tokenize) <br>
# $\;\;\;\;$[4.2. Whatever else](#whetev) <br>
# $\;\;\;\;$[4.3. Genegrate numerical representation](#whetev1) <br>
# [5. Writing Output Files](#write) <br>
# $\;\;\;\;$[5.1. Vocabulary List](#write-vocab) <br>
# $\;\;\;\;$[5.2. Sparse Matrix](#write-sparseMat) <br>
# [6. Summary](#summary) <br>
# [7. References](#Ref) <br>

# %% [markdown]
# <div class="alert alert-block alert-success">
#     
# ## 1.  Introduction  <a class="anchor" name="Intro"></a>

# %% [markdown]
# This script tokenizes the review text extracted from task1 and generates two output files: a vocabulary dictionary of the tokens in a text format and a count vector text file that lists the token indices and their frequencies for each gmap_id.

# %% [markdown]
# <div class="alert alert-block alert-success">
#     
# ## 2.  Importing Libraries  <a class="anchor" name="libs"></a>

# %% [markdown]
# In this assessment, any python packages is permitted to be used. The following packages were used to accomplish the related tasks:
# 
# * **os:** to interact with the operating system, e.g. navigate through folders to read files
# * **re:** to define and use regular expressions
# * **pandas:** to work with dataframes
# * **multiprocessing:** to perform processes on multi cores for fast performance 
# * ...

# %%
import os
import re
import langid
import pandas as pd
import multiprocessing
from itertools import chain
import nltk
from nltk.probability import *
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import MWETokenizer
from nltk.stem import PorterStemmer
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer

# %% [markdown]
# -------------------------------------

# %% [markdown]
# <div class="alert alert-block alert-success">
#     
# ## 3.  Examining Input File <a class="anchor" name="examine"></a>

# %%
df = pd.read_json("task1_030.json")
df = df.T.reset_index() # Transpose the DataFrame and reset the index to convert JSON keys to columns
df.columns = ['gmap_id', 'reviews', 'earliest', 'latest'] # Rename columns for clarity
print(df.head(20))
# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Filter out reviews where 'review_text' is "None"
    filtered_reviews = [review for review in row["reviews"] if review.get("review_text") != "None"]
    df.at[index, "reviews"] = filtered_reviews

# %% [markdown]
# Justification: After reading the json file into a pandas dataframe. I found that the column and rows need to be transposed. Hence I used .T to ahieve that. Only the review contains more than 70 texts review is needed. So I used a list comprehension first to drop the review that is "None".

# %% [markdown]
# <div class="alert alert-block alert-success">
#     
# ## 4.  Loading and Parsing File <a class="anchor" name="load"></a>

# %% [markdown]
# In this section, ....

# %%
print(df.shape)
def extract_reviews(df):
    # Filter out rows where the 'reviews' list has fewer than 70 entries and reset the index
    df = df[df['reviews'].apply(len) >= 70].reset_index(drop=True)
    df['review_text'] = '' # Initialize a new column 'review_text' to store combined review text
    for index, row in df.iterrows():
        review_texts = [] # Initialize a list to store review texts for the current row
         # Iterate over each review in the 'reviews' list of the current row
        for review in row['reviews']:
            text = review.get('review_text', '').lower()
            text = re.sub(r'[^\x00-\x7F]+', '', text) # Remove non-ASCII characters
            review_texts.append(text)
        df.at[index, 'review_text'] = ' '.join(review_texts)
    print(df.shape)
    return df

df_review = extract_reviews(df)

# %% [markdown]
# After dropping all the review_text which is "None", I filtered the review by whose "reviews" list has more than 70 elements by applying len() method. After filtering the reviews, I created a new dataframe to store them. After that, a for loop is used to iterate through each row of the new data frame to store the review text to a new column "review_text". And then, join the review_text list to a string for further tokenization.

# %% [markdown]
# <div class="alert alert-block alert-warning">
#     
# ### 4.1. Tokenization <a class="anchor" name="tokenize"></a>

# %% [markdown]
# Tokenization is a principal step in text processing and producing unigrams. In this section, ....

# %%
#Correct version
stopwords_list = []
with open(r'Student Data/stopwords_en.txt', 'r') as file:
    for line in file:
        stopwords_list.append(line.strip()) # Read each line, strip whitespace, and add to the stopwords list
df_review["tokennized_review"] = None # Initialize a new column 'tokennized_review' with None values to store tokenized and processed reviews

for index,row in df_review.iterrows():
    # Tokenization
    review_text = row["review_text"]
    tokenizer = RegexpTokenizer(r"[a-zA-Z]+")
    unigram_tokens = tokenizer.tokenize(review_text) # Tokenize the review text into words

    # Remove tokens less than three
    unigram_tokens = [token for token in unigram_tokens if len(token) >= 3]

    # Remove independent stop words
    stopped_tokens = [w for w in unigram_tokens if w not in stopwords_list]
    #Stemming: Reduce tokens to their root form using the Porter stemmer
    stemmer = PorterStemmer()
    stemed_tokens = [stemmer.stem(token) for token in stopped_tokens]
    # Store the processed tokens in the 'tokennized_review' column of the DataFrame
    df_review.at[index, "tokennized_review"] = stemed_tokens
    # print(mwe_tokens)

# %% [markdown]
# 1. Read the stopwords txt and append them to a list called stopwords_lis
# 2. Iterate through the rows of df_reviews
# 3. Task a finished: Tokenize the review_text of each row by RegexpTokenizer
# - **Justification**: To perform further processing like remove stop words, Stemmentization etc, Words need to be tokenized first.
# 4. Task e finished: Used list comprehension to  filter out the tokens to be length longer than3
# - **Justification**: Words's Lenth less than 3 are always meanningless words. Remove than at the beginning can help improve efficency of data processing.
# 5. Task b1 finishied: Used list comprehension to filter the unigram_tokens by word not in the stop_word list.
# - **Justification**: Stop words are considered noise that needs to be removed. Remove them before stemmatization, because if the stopwords are stemmed first, the words would be transformed that not being able to match the stopword list.
# 6. Stemming the remaining tokens to their root forms using the Porter Stemmer.
# - **Justification**: To identify context dependent stopwords, words need to be stemmed before removing them

# %%
# Remove context-dependent stopwords
word_in_business = {}  # Initialize a dictionary to track the set of businesses in which each word appears
total_businesses = df_review.shape[0]  # Get the total number of businesses (documents) in the DataFrame

for index, row in df_review.iterrows():
    gmap_id = row['gmap_id']  
    unique_words = set(row['tokennized_review'])  # Convert the tokenized review into a set of unique words

    # Update the dictionary with each word and the set of businesses where it appears
    for word in unique_words:
        if word not in word_in_business:
            word_in_business[word] = set()  # Initialize an empty set for the word if it does not exist in the dictionary
        word_in_business[word].add(gmap_id)  # Add the current business ID to the set for the word

# Identify context-dependent stopwords that appear in more than 95% of businesses
context_dependent_stopwords = [word for word, business_set in word_in_business.items()
                                if len(business_set) / total_businesses > 0.95]

for index, row in df_review.iterrows():
    stopped_tokens = [w for w in row['tokennized_review'] if w not in context_dependent_stopwords]
    df_review.at[index, "tokennized_review"] = stopped_tokens


# %% [markdown]
# 1.Identifying and removing context-dependent stopwords, which are words that appear in more than 95% of businesses.
# - **Justification**: For example: "The words "restaurant" and "restaurants" are treated separately. "Restaurant" might appear in 80% of reviews, and "restaurants" in 20%. Individually, neither might exceed a frequency threshold (like 95%). Thus, neither is removed as a context-dependent stopword.

# %%
# Remove rare words
word_in_business = {}
total_businesses = df_review.shape[0]

for index, row in df_review.iterrows():
    gmap_id = row['gmap_id']
    unique_words = set(row['tokennized_review'])

    # Update the dictionary with each word and its associated businesses
    for word in unique_words:
        if word not in word_in_business:
            word_in_business[word] = set()
        word_in_business[word].add(gmap_id)

rare_words = [word for word, business_set in word_in_business.items()
                                if len(business_set) / total_businesses < 0.05]

# Iterate over each row again to remove rare words from tokenized reviews
for index,row in df_review.iterrows():
    rared_tokens = [w for w in row['tokennized_review'] if w not in rare_words]
    df_review.at[index, "tokennized_review"] = rared_tokens


# %% [markdown]
# 1. The purpose of this code is to remove rare words from the tokenized reviews. Rare words are defined as those that appear in less than 5% of the businesses in the dataset.
# - **Justification**: This step should be done after stemmatization. For example,"running," "ran," and "runs" all stem to "run." If rare word removal were done before stemming, each form would be treated as a separate entity, potentially missing the fact that collectively they might not be rare.

# %%
# Bigram words
# Combine all tokenized reviews into a single list of words
all_tokens = sum(df_review["tokennized_review"], [])

# Set up bigram association measures and finder
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = nltk.collocations.BigramCollocationFinder.from_words(all_tokens)

# Select the top 200 bigrams using Pointwise Mutual Information (PMI) as the scoring measure
bigram_token = finder.nbest(bigram_measures.pmi, 200) 

# Initialize a Multi-Word Expression (MWE) tokenizer with the identified bigrams
tokenizer = MWETokenizer(bigram_token)

for index, row in df_review.iterrows():
    # Tokenize the review using the MWE tokenizer to combine bigrams into single tokens
    mwe_tokens = tokenizer.tokenize(row["tokennized_review"])
    # Update the 'tokennized_review' column with the new tokens including bigrams
    df_review.at[index, "tokennized_review"] = mwe_tokens  

# %% [markdown]
# 1. Used MWETokenizer with pmi to find the first 200 bigrams and iterate through each row to identify the bigram and add bigram to it.
# 
# - **Justification**: The Pointwise Mutual Information (PMI) measure used to identify meaningful bigrams relies on the frequency and co-occurrence of words within a text. By removing unimportant or noisy words first, the PMI calculation becomes more accurate and reflective of meaningful word associations. If bigrams were generated earlier, PMI scores could be skewed by the presence of irrelevant or overly common tokens, leading to less meaningful bigrams.

# %%
# Export word count txt file
all_tokens_per_doc = [token for token in df_review["tokennized_review"]]

# %% [markdown]
# 1. Sorted the cleaned tokens and write them to test_vocab.txt by desired format.

# %% [markdown]
# At this stage, all reviews for each PID are tokenized and are stored as a value in the new dictionary (separetely for each day).
# 
# -------------------------------------

# %% [markdown]
# <div class="alert alert-block alert-warning">
#     
# ### 4.2. Generate numerical representation<a class="anchor" name="bigrams"></a>

# %% [markdown]
# One of the tasks is to generate the numerical representation for all tokens in abstract.  .....

# %%
# Initialize a CountVectorizer to convert text documents into a matrix of word counts
vectorizer = CountVectorizer(analyzer="word")

# Join tokens in each document into a single string to prepare for vectorization
documents = [' '.join(tokens) for tokens in all_tokens_per_doc]

# Fit the vectorizer to the documents and transform them into a document-term matrix
data_features = vectorizer.fit_transform(documents)

# Retrieve the list of unique words gained by the vectorizer
vocab = vectorizer.get_feature_names_out()


# Create a dictionary mapping each word to its index in the vocabulary
vocab_dist = {word: index for index, word in enumerate(vocab)}

print(vocab_dist)
output_countvec = []

# Iterate over each document and its corresponding Google Map ID
for doc_index, gmap_id in enumerate(df_review["gmap_id"]):
    doc_vector = data_features[doc_index]  # Get the count vector for the current document
    
    # Zip together the indices and frequencies of the non-zero elements in the count vector
    token_counts = zip(doc_vector.indices, doc_vector.data)
    
    # Format each word index and its frequency as "index:frequency"
    formatted_counts = [f"{index}:{frequency}" for index, frequency in token_counts]
    
    # Construct the output line with the gmap_id followed by the formatted count vector
    output_line = f"{gmap_id}," + ",".join(formatted_counts)
    
    # Append the formatted line to the output list
    output_countvec.append(output_line)

# %% [markdown]
# The code above processes a collection of text documents to create a numerical representation suitable for text analysis. It uses CountVectorizer to tokenize the documents and generate a document-term matrix, capturing the frequency of each word in each document. The vocabulary of unique words is extracted and written to a file, mapping each word to an index. Additionally, the code generates formatted output for each document, listing the frequency of each word by its index.

# %% [markdown]
# #### Whatever else <a class="anchor" name="whatev1"></a>

# %% [markdown]
# <div class="alert alert-block alert-success">
#     
# ## 5. Writing Output Files <a class="anchor" name="write"></a>

# %% [markdown]
# files need to be generated:
# * Vocabulary list
# * Sparse matrix (count_vectors)
# 
# This is performed in the following sections.

# %% [markdown]
# <div class="alert alert-block alert-warning">
#     
# ### 5.1. Vocabulary List <a class="anchor" name="write-vocab"></a>

# %% [markdown]
# List of vocabulary should also be written to a file, sorted alphabetically, with their reference codes in front of them. This file also refers to the sparse matrix in the next file. For this purpose, .....

# %%
with open("030_vocab.txt",'w') as file:
    for index,word in enumerate(vocab_dist):
        file.write(f"{word}:{ index}\n")

# %% [markdown]
# <div class="alert alert-block alert-warning">
#     
# ### 5.2. Sparse Matrix <a class="anchor" name="write-sparseMat"></a>

# %% [markdown]
# For writing sparse matrix for a paper, we firstly calculate the frequency of words for that paper ....

# %%
with open("030_countvec.txt","w") as file:
    for line in output_countvec:
        file.write(line + "\n")

# %% [markdown]
# -------------------------------------

# %% [markdown]
# <div class="alert alert-block alert-success">
#     
# ## 6. Summary <a class="anchor" name="summary"></a>

# %% [markdown]
# The provided code successfully accomplishes the objectives of generating vocab.txt and countvect.txt files by systematically filtering the review text and applying a series of well-structured text preprocessing steps. By following a logical sequence—starting with tokenization, then removing context-independent and context-dependent stopwords, followed by stemming, and finally removing rare words—the code effectively refines the dataset to focus on the most informative and meaningful tokens. This step-by-step approach ensures that the resulting vocabulary is both clean and relevant, enhancing the quality of the data for subsequent analysis or modeling tasks. The target of creating a concise and optimized vocabulary, as well as a corresponding count vector, is achieved efficiently and effectively through this methodical preprocessing pipeline.

# %% [markdown]
# -------------------------------------

# %% [markdown]
# <div class="alert alert-block alert-success">
#     
# ## 7. References <a class="anchor" name="Ref"></a>

# %% [markdown]
# [1] Pandas dataframe.drop_duplicates(), https://www.geeksforgeeks.org/python-pandas-dataframe-drop_duplicates/, Accessed 13/08/2022.
# 
# 

# %% [markdown]
# ## --------------------------------------------------------------------------------------------------------------------------


