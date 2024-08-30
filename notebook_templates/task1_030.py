import re
import pandas as pd
import datetime
import json 

# Open the sample_input_0.txt for testing
with open(r'Sample Input/sample_input_0.txt', 'r', encoding='utf-8') as file: 
     text = file.read()
     
#segment by review:
pattern = re.compile(r'<record>(.*?)</record>', re.DOTALL | re.IGNORECASE)

# Find all matches
matches = pattern.findall(text)

reviews = matches 

print(len(reviews))

print(reviews[0])

# Regex pattern for detecting qmap_id 
pattern = re.compile(r'(?:<\s?(?:gmapid|gmap_id)>\s*?)(.*?)(?:<)', re.IGNORECASE)

qmap_ids = [] 

# function to Find all matches
for review in reviews:
    matches = pattern.findall(review)
    if len(matches) > 1:
        print("ERROR")
    qmap_ids.append(matches)

print(len(qmap_ids))
print(qmap_ids)

# find gmap id function
def find_gmap_id(review):
    pattern = re.compile(r'(?:<\s?(?:gmapid|gmap_id)>\s*?)(.*?)(?:<)', re.IGNORECASE)
    matches = pattern.findall(review)
    if len(matches) > 1:
        print("ERROR1")
    return matches[0]

# Regex pattern for detecting user_id
pattern = re.compile(r'(?:<\s?(?:userid|user_id|user)\.?>\s?)(\d*?)(?:<)', re.IGNORECASE)

user_ids = [] 

# Find all matches
for review in reviews:
    matches = pattern.findall(review)
    if len(matches) > 1:
        print("ERROR")
    user_ids.append(matches)

print(len(user_ids))
print(user_ids)

# find user id function
def find_user_id(review):
    pattern = re.compile(r'(?:<\s?(?:userid|user_id|user)\.?>\s?)(\d*?)(?:<)', re.IGNORECASE)
    matches = pattern.findall(review)
    if len(matches) > 1:
        print("ERROR2")
    elif len(matches) == 0:
        return "none"
    return matches[0]

# find excel user id function
def find_excel_user_id(review):
    pattern = re.compile(r'(\d*)', re.IGNORECASE)
    matches = pattern.findall(review)
    if len(matches) == 0:
        return "none"
    return matches[0]

# Regex for parsing time 
pattern = re.compile(r'(?:<\s?(?:time|date)\.?>\s?)(\d*?)(?:<)', re.IGNORECASE)
times = [] 
# Find all matches
for review in reviews:
    matches = pattern.findall(review)
    # Change unix timestamp of millisecond to second
    matches_s = float(matches[0]) / 1000.0
    # Convert the matche_s of second to a datetime object
    matches_date_time = datetime.datetime.fromtimestamp(matches_s)
    # Convert the format of datetime object into what is needed
    matches_date_time = [matches_date_time.strftime('%Y-%m-%d %H:%M:%S')]
    if len(matches) > 1:
        print("ERROR")
    times.append(matches_date_time)

print(len(times))
print(times)

# txt time function
def find_time(review):
    pattern = re.compile(r'(?:<\s?(?:time|date)\.?>\s?)(\d*?)(?:<)', re.IGNORECASE)
    matches = pattern.findall(review)
    matches_s = float(matches[0]) / 1000.0
    matches_date_time = datetime.datetime.fromtimestamp(matches_s)
    matches_date_time = [matches_date_time.strftime('%Y-%m-%d %H:%M:%S')]
    if len(matches) > 1:
        print("ERROR")
    return matches_date_time[0]

# excel find time
def find_excel_time(review):
    matches_s = review / 1000.0
    matches_date_time = datetime.datetime.fromtimestamp(matches_s)
    matches_date_time = [matches_date_time.strftime('%Y-%m-%d %H:%M:%S')]
    return matches_date_time[0]

# Regex for parsing rating 
pattern = re.compile(r'(?:<\s?(?:rating|rate)\.?>\s?)(\d)(?:<)', re.IGNORECASE)

ratings = [] 

# Find all matches
for review in reviews:
    matches = pattern.findall(review)
    if len(matches) > 1:
        print("ERROR")
    ratings.append(str(matches))

print(len(ratings))
print(ratings)

# find rating function
def find_rating(review):
    pattern = re.compile(r'(?:<\s?(?:rating|rate)\.?>\s?)(\d)(?:<)', re.IGNORECASE)
    matches = pattern.findall(review)
    if len(matches) > 1:
        print("ERROR5")
    elif len(matches) == 0:
        return "none"
    else:
        return str(matches[0])
    
    # Regex for parsing reviews 
pattern = re.compile(r'(?:<\s?(?:text|review)\.?>\s?)([\s\S]*?)(?:<)', re.IGNORECASE)
# Regex for detecting emojis
emoji_pattern = re.compile("["
        "\U0001F600-\U0001F64F"  
        "\U0001F300-\U0001F5FF"  
        "\U0001F680-\U0001F6FF"  
        "\U0001F1E0-\U0001F1FF"  
        "\U00002702-\U000027B0"  
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)

## Regex to detect if review is translated
translation_pattern = r"(?<=\(Translated by Google\)).*?(?=\n)"
review_texts = [] 

# Find all matches
for review in reviews:
    matches = pattern.findall(review)
    if len(matches) > 1:
        print("ERROR")
        
    # Pre Process Text 
    if len(matches) == 1:
        text = matches[0] # Convert to str 
    elif len(matches) > 1:
        text = ' '.join(matches) # Concatenate Strings if more than one match
    else:
        text = 'none'
    # Extract translation
    translation = re.search(translation_pattern, text)
    if translation:
        text = translation.group(0)
    # Remove Emojis
    text = emoji_pattern.sub(r'', text)
    # Change to lower case 
    text = text.lower()
    
    review_texts.append(text)
    
print(len(review_texts))
print(review_texts)

# Regex for detecting emojis
emoji_pattern = re.compile("["
        "\U0001F600-\U0001F64F"  
        "\U0001F300-\U0001F5FF"  
        "\U0001F680-\U0001F6FF"  
        "\U0001F1E0-\U0001F1FF"  
        "\U00002702-\U000027B0"  
        "\U000024C2-\U0001F251"  
        "]+", flags=re.UNICODE)

# Regex for detecting if a review contains a translation by Google
translation_pattern = re.compile(r"(?<=\(Translated By Google\)).*?(?=\n)", re.IGNORECASE)

# Function to extract and clean review text from raw review input
def find_review_text(review):
    pattern = re.compile(r'(?:<\s?(?:text|review)\.?>\s?)([\s\S]*?)(?:<)', re.IGNORECASE)  # Regex to find text within <text> or <review> tags
    matches = pattern.findall(review)
    if len(matches) > 1:
        print("ERROR")  # Error if multiple matches are found
    if len(matches) == 0:
        return "None"  # Return "None" if no matches are found
    else: 
        text = matches[0]  # Extract the first match
        translation = re.search(translation_pattern, text)  # Check for translation text
        if translation:
            text = translation.group(0)  # If found, use the translated text
        text = emoji_pattern.sub(r'', text)  # Remove any emojis from the text
        if text != "None":
            text = text.lower()  # Convert text to lowercase for consistency
        return text

# Function to extract and clean review text from Excel data
def find_excel_review_text(review):
    if isinstance(review, float):
       return 'None'  # Return "None" if the review is not a string (e.g., NaN)
    matches = [review]  # Treat the review as a list to mimic the regex extraction logic
    if len(matches) > 1:
        print("ERROR6")  
    if len(matches) == 0:
        return "None"  
    else: 
        text = matches[0]  # Extract the review text
        translation = re.search(translation_pattern, text)  # Check for translation text
        if translation:
            text = translation.group(0)  # If found, use the translated text
        text = emoji_pattern.sub(r'', text)  # Remove any emojis from the text
        if text != "None":
            text = text.lower()  # Convert text to lowercase for consistency
        return text

# Regex pattern to parse picture-related tags in the review text
pattern = re.compile(r'(?:<\s?(?:pics|pic |pictures)\.?>\s?)(.*?)(?:<)', re.IGNORECASE)
pictures = [] 

# Extract picture information from each review
for review in reviews:
    matches = pattern.findall(review)
    if len(matches) > 1:
        print("ERROR")  # Error if multiple matches are found
    pictures.append(matches)

print(len(pictures))
print(pictures)

# Determine if a picture exists in each review
if_pic = []
for picture in pictures:
    n = ['N']
    y = ['Y']
    if picture == ['None']:
        if_pic.append(n)
    else:
        if_pic.append(y)

print(len(if_pic))
print(if_pic)

# Regex pattern to parse image dimensions
dimention_pattern = re.compile(r'=w(\d+)-h(\d+)-k', re.IGNORECASE)
pic_dim = []

# Extract dimensions for each picture
for picture in pictures:
    dimention = dimention_pattern.findall(picture[0])
    pic_dim.append(dimention)

print(len(pic_dim))
print(pic_dim)

# Check if a picture exists in a review text
def find_if_pic(review):
    pattern = re.compile(r'(?:<\s?(?:pics|pic |pictures)\.?>\s?)(.*?)(?:<)', re.IGNORECASE)
    matches = pattern.findall(review)
    if len(matches) > 1:
        print("ERROR")
    n = 'N'
    y = 'Y'
    if matches == ['None']:
        return n
    else:
        return y

# Check if a picture exists in Excel cell data
def find_excel_if_pic(review):
    matches = [review]
    if len(matches) > 1:
        print("ERROR7")
    n = 'N'
    y = 'Y'
    if matches == ['None']:
        return n
    else:
        return y
    
# Find image dimensions in review text
def find_pic_dim(review):
    if isinstance(review, float):
       return []
    dimention_pattern = re.compile(r'=w(\d+)-h(\d+)-k', re.IGNORECASE)
    dimentions = dimention_pattern.findall(review)
    dimentions_as_lists = [list(dimention) for dimention in dimentions]
    dimentions_as_lists
    return dimentions_as_lists

# Regex for parsing responses 
pattern = re.compile(r'(?:<\s?(?:response|resp)\.?>\s?)(.*?)(?:<)', re.IGNORECASE)
responses = [] 

# Find all matches

for review in reviews:
    matches = pattern.findall(review)
    if len(matches) > 1:
        print("ERROR")
    responses.append(matches)

print(len(responses))
print(responses)

# find response function
def find_response(review):
    if isinstance(review, float):
       return 'N'
    pattern = re.compile(r'(?:<\s?(?:response|resp)\.?>\s?)(.*?)(?:<)', re.IGNORECASE)
    matches = pattern.findall(review)
    if len(matches) > 1:
        print("ERROR8")
    n = 'N'
    y = 'Y'
    if matches == ['None']:
        return n
    else:
        return y

# .txt Parsing 
txt_data = []
for i in range(15):
    with open (f'Sample input/sample_input_{i}.txt', 'r', encoding='utf-8') as file:
    # with open (f'Student Data/group030_{i}.txt', 'r', encoding='utf-8') as file:
        text = file.read()
        #segment by review:
        pattern = re.compile(r'<record>(.*?)</record>', re.DOTALL | re.IGNORECASE)

        # Find all matches
        matches = pattern.findall(text)

        reviews = matches
        for review in reviews:   
            #gmap id
            gmap_id = find_gmap_id(review)
            #user id
            user_id = find_user_id(review)
            # time
            time = find_time(review)
            # revie rating
            review_rating = find_rating(review)
            #review text
            review_text = find_review_text(review)
            #if_pic
            if_pic = find_if_pic(review)
            #picture dimension
            pic_dim = find_pic_dim(review)
            #if response
            if_response = find_response(review)
            review_dict = {
                "gmap_id" : gmap_id, 
                "user_id" : user_id,
                "time" : time,
                "review_rating" : review_rating,
                "review_text" : review_text,
                "if_pic" : if_pic,
                "pic_dim" : pic_dim,
                "if_response" : if_response
            }
            txt_data.append(review_dict)
print(txt_data)

excel_data = []
# EXCEL PARSING
for i in range(15):
    excel_df = pd.read_excel("Sample Input/sample_input.xlsx", sheet_name = f'Sheet{i}')
    # excel_df = pd.read_excel("Student Data/group030.xlsx", sheet_name = f'Sheet{i}')
    # drop columns like "x1,x2,etc"
    colums_to_drop = [col for col in excel_df.columns if col.startswith('x')]
    excel_df.drop(columns=colums_to_drop, inplace=True)
    # drop rows that are all empty
    excel_df.dropna(axis=0, how='all', inplace=True)

    # loop through the rows
    for index, row in excel_df.iterrows():
        #gmap id
        gmap_id = row["gmap_id"]
        #user id
        user_id = find_excel_user_id(row["user_id"])
        # time
        time = find_excel_time(row["time"])
        # revie rating 
        review_rating = row["rating"]
        #review text
        review_text = find_excel_review_text(row["text"])
        #if_pic
        if_pic = find_excel_if_pic(row["pics"])
        #picture dimension
        pic_dim = find_pic_dim(row["pics"])
        #if response
        if_response = find_response(row["resp"])
        review_dict = {
                "gmap_id" : gmap_id, 
                "user_id" : user_id,
                "time" : time,
                "review_rating" : review_rating,
                "review_text" : review_text,
                "if_pic" : if_pic,
                "pic_dim" : pic_dim,
                "if_response" : if_response
            }
        excel_data.append(review_dict)
    print(excel_data)

combined_df = pd.DataFrame(txt_data + excel_data)

# For converting pic_dim column to hashatable type 
def convert_to_hashable(value):
    return tuple(tuple(sublist) for sublist in value)
# For converting pic_dim column back to list 
def convert_to_original(value):
    return [list(subtuple) for subtuple in value]

# Remove duplicate reviews 
combined_df["pic_dim"] = combined_df["pic_dim"].apply(convert_to_hashable)
combined_df.drop_duplicates()
combined_df["pic_dim"] = combined_df["pic_dim"].apply(convert_to_original)

print(combined_df.head())


combined_df.to_excel('test3.xlsx', index=False)