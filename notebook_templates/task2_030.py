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

df = pd.read_json("task1_030.json")
df = df.T.reset_index() # Transpose the DataFrame and reset the index to convert JSON keys to columns
df.columns = ['gmap_id', 'reviews', 'earliest', 'latest'] # Rename columns for clarity
print(df.head(20))
# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Filter out reviews where 'review_text' is "None"
    filtered_reviews = [review for review in row["reviews"] if review.get("review_text") != "None"]
    df.at[index, "reviews"] = filtered_reviews

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

# Export word count txt file
all_tokens_per_doc = [token for token in df_review["tokennized_review"]]

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
    output_line = f"{gmap_id} " + ", ".join(formatted_counts)
    
    # Append the formatted line to the output list
    output_countvec.append(output_line)

with open("030_vocab.txt",'w') as file:
    for index,word in enumerate(vocab_dist):
        file.write(f"{word}:{ index}\n")

with open("030_countvec.txt","w") as file:
    for line in output_countvec:
        file.write(line + "\n")