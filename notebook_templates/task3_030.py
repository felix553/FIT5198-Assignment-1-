import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import folium
from branca.colormap import linear
from collections import defaultdict

# Intialise Data 
review_df = pd.read_excel(r"combined_data.xlsx")
review_df.head()
review_df.shape
review_df.columns
review_df.dtypes
# Transform time to datetime object
review_df['time'] = pd.to_datetime(review_df['time'])
review_df.head()
# Generate summary statistics 
review_df.describe(include='all')
# Check for missing values 
review_df.isnull().sum()

# Extract year and month for aggregation 
review_df['year'] = review_df['time'].dt.year
review_df['month'] = review_df['time'].dt.to_period('M').astype(str)  

# Calculate monthly reviewss
monthly_reviews = review_df.groupby('month').size().reset_index(name='monthly_reviews')

# Add year for display purposes 
monthly_reviews['month'] = pd.to_datetime(monthly_reviews['month'], format='%Y-%m')

# Calculate position_yly reviews
yearly_reviews = review_df.groupby('year').size().reset_index(name='yearly_reviews')


# Generate line graph 
plt.figure(figsize=(12, 8))
sns.lineplot(data=monthly_reviews, x='month', y='monthly_reviews', marker='o', color='green')

# Generate position_yly total labels 
for index, row in yearly_reviews.iterrows():
    # Intialise month and day to 1 to display at beginning of Year   
    position_x = pd.to_datetime(f'{row["year"]}-01-01')

    # Vertical position for position_y total labels 
    position_y = monthly_reviews['monthly_reviews'].min() -50
    
    plt.text(position_x, position_y, f'{row["yearly_reviews"]}', color='green', ha='center', fontsize=12)
    # Add axis lines 
    plt.axvline(x=position_x, linestyle='--', alpha=0.3)
    
# Generate yearly x ticks 
years = [str(year) for year in range(2008, 2023)]
years_dt = pd.to_datetime(years)
plt.xticks(years_dt, years, fontsize=12)

plt.yticks(fontsize=12)
plt.title('Total Reviews between 2008 and 2022 with yearly totals', fontsize=20)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Reviews', fontsize=15)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.show()

# Generate bar graph for review_rating distribution with response categories
palette = {'Y': 'lime', 'N': 'salmon'} # Colour choice for graph
plt.figure(figsize=(12, 8))
ax = sns.countplot(data=review_df, x='review_rating', palette=palette, hue='if_response')

plt.title('Distribution of Reviews by ratings and response status', fontsize=20)
plt.xlabel('Review Rating', fontsize=14)
plt.xticks(fontsize=12)

plt.ylabel('Numer of Reviews', fontsize=14)
plt.yticks(fontsize=12)
plt.legend(title='Business Responded', title_fontsize=14, labels=['Yes', 'No'], fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Add value labels
for container in ax.containers:
    ax.bar_label(container, fontsize=12)

plt.show()

# Add new boolean column to check if review contains text body 
review_df['if_text'] = review_df['review_text'].apply(lambda x: 'Y' if x != 'none' else 'N')
palette = {'Y': 'lime', 'N': 'salmon'}
# Generate bar graph for count of reviews with review text 
plt.figure(figsize=(12, 8))
ax = sns.countplot(data=review_df, x='if_text', hue='if_response', hue_order=['Y', 'N'], palette=palette)

plt.title('Distribution of reviews on whether they contain text and response status ', fontsize=20)

plt.xlabel('Review contains text body', fontsize=16)
plt.xticks(['Y', 'N'], ['Yes', 'No'], fontsize=16)

plt.ylabel('Number of Reviews ', fontsize=14)
plt.yticks(fontsize=12)

plt.legend(title='Business Responded', loc='upper right', labels=['Yes', 'No'], title_fontsize=14, fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Add labels on bars
for container in ax.containers:
    ax.bar_label(container, fontsize=14)
    
plt.show()

# Aggregating reviews based on business 
aggregated_review_df = review_df.groupby('gmap_id').agg(
    review_count=('gmap_id', 'size'),
    user_count = ('user_id', 'size'),
    time_count = ('time', 'median'),
    avg_rating = ('review_rating', 'mean'),
    review_text_count=('review_text', lambda x: (x != 'none').sum()),
    pic_count = ('if_pic', lambda x: (x == 'Y').sum()),
    response_count=('if_response', lambda x: (x == 'Y').sum()),

)

plt.figure(figsize=(12, 6))
# Boxplot for avg review ratings 
sns.boxplot(data=aggregated_review_df, x='avg_rating', color='purple')
plt.title('Box Plot of Average Business Ratings', fontsize=18)
plt.grid(axis='y', alpha=0.3)
plt.xlabel('Average Review Rating', fontsize=12)
plt.xticks(fontsize=14)
plt.tight_layout()
plt.show()

meta_df = pd.read_json(r"meta-California.json", lines=True)

meta_df.shape
meta_df.describe()
meta_df.dtypes
meta_df.columns
meta_df.head()
# Check for nulls 
meta_df.isnull().sum()

# Merge metadata dataset with task1 dataset 
merged_df = pd.merge(review_df, meta_df, on='gmap_id', how='left')

# Drop redundant/unimportant columns
columns = ['avg_rating', 'num_of_reviews', 'relative_results']
merged_df = merged_df.drop(columns=columns)

merged_df.shape
merged_df.head(5)

# Final null value check 
merged_df.isnull().sum()

merged_df['MISC']

# Change column to dict type 
def convert_to_dict(row):
    # Convert to empty dict if 'None' 
    if row is None or row =='None':
        return {}
    else:
        return row

merged_df['transformed_MISC'] = merged_df['MISC'].apply(convert_to_dict)

merged_df['transformed_MISC']

# Function to extract MISC tags and encode them into new boolean attributes 
def extract_misc(misc_dict):
    flattened = {}
    for key, values in misc_dict.items():
        for value in values:
            flattened[f'{key}: {value}'] = 1
    return flattened

# Generate new dataframe with MISC tag encodings 
misc_encodings = merged_df['transformed_MISC'].apply(lambda x: pd.Series(extract_misc(x)))

# Fill NaN values with 0
misc_encodings = misc_encodings.fillna(0)

print(misc_encodings)

# Combine with original DataFrame
merged_df_combined = pd.concat([merged_df, misc_encodings], axis=1)
merged_df_combined = merged_df_combined.drop(columns='transformed_MISC')

merged_df_combined.head()


# New Dataframe to store average ratings for each tag 
tag_ratings = pd.DataFrame(columns=['tag', 'average_rating'])

# Aggregate based on tag 
tags = misc_encodings.columns
# Calculate the average review rating for each tag effect
for tag in tags:
    # Aggregate by tag and calculate average rating 
    avg_review = merged_df_combined.groupby(tag).agg(average_rating=('review_rating', 'mean'),
                                                     count_reviews=(tag, 'size')).reset_index()
    #print(avg_review)

    # We only want average rating for reviews with tag included 
    avg_review = avg_review[avg_review[tag]==1].drop(columns=[tag])
    #print(avg_review)
    
    # Append the results to the new Dataframe 
    tag_ratings = pd.concat([tag_ratings, pd.DataFrame({'tag': tag, 
                                                        'average_rating':avg_review['average_rating'], 
                                                        'reviews': avg_review['count_reviews']})], 
                            ignore_index=True)

print(tag_ratings.head())

# Filter tags that have at least appeared a significant amount of times  
median = tag_ratings['reviews'].median()
tag_ratings = tag_ratings[tag_ratings['reviews'] > median]
# Sort on rating 
tag_ratings = tag_ratings.sort_values(by='average_rating', ascending=False)

# filter top 10 tags 
top_10 = tag_ratings.head(10)

# Set the plot size
plt.figure(figsize=(12, 8))

# Bar Chart 
ax = sns.barplot(data=top_10, x='tag', y='average_rating', color='purple')
plt.title('Top 10 MISC Tags with the Highest Average Review Ratings', fontsize=16)
plt.xlabel('Tags', fontsize=14)
plt.xticks(rotation=60, fontsize=10)
plt.ylabel('Average Review Rating', fontsize=14)
plt.yticks(fontsize=10)
# Add labels on bars
for i, bar in enumerate(ax.patches): # [1]
    x_pos = bar.get_x() + bar.get_width() / 2
    y_pos = bar.get_height()/2
    reviews = top_10.iloc[i]['reviews']
    
    ax.text(
        x_pos, 
        y_pos, 
        f'{int(reviews)}', 
        ha='center', 
        va='bottom', 
        fontsize=12,
        color='white'
    )
    
plt.tight_layout()
plt.show()

# Aggregate based on business
aggregated_df = merged_df.groupby(['gmap_id', 'latitude', 'longitude', 'name']).agg(average_rating=('review_rating', 'mean'),
                                                                                    count_reviews=('gmap_id', 'size')).reset_index()

# Create a map which starts at average position from reviews 
map_center = [aggregated_df['latitude'].mean(), aggregated_df['longitude'].mean()]
business_map = folium.Map(location=map_center, zoom_start=6.5, tiles='cartodbpositron')

# Generate a continuous colour scale 
colormap = linear.YlGnBu_09.scale(aggregated_df['average_rating'].min(), aggregated_df['average_rating'].max()) # [2]

# Plot points 
for column, row in aggregated_df.iterrows(): 
    color = colormap(row['average_rating']) # Generate colour corresponding to average rating 
    radius = max(2, min(12, row['count_reviews'] ** 0.5)) # Generate corresponding radius depending on number of reviews 
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=radius, 
        color=color,
        fill=True,
        # Include popup for additional information 
        popup=(
            f"Gmap ID: {row['gmap_id']}<br>"
            f"Business: {row['name']}<br>"
            f"Average Rating: {row['average_rating']:.1f}<br>"
            f"Number of Reviews: {row['count_reviews']}"
        )
    ).add_to(business_map)

# Add legend 
colormap.caption = 'Average Rating'
colormap.add_to(business_map)


business_map


LA_position = [34.0522, -118.2437]
# Create a map which starts at average position from reviews 
map_center = [LA_position[0], LA_position[1]]
business_map = folium.Map(location=map_center, zoom_start=9, tiles='cartodbpositron')

# Generate a continuous colour scale 
colormap = linear.YlGnBu_09.scale(aggregated_df['average_rating'].min(), aggregated_df['average_rating'].max())

# Plot points 
for column, row in aggregated_df.iterrows():
    color = colormap(row['average_rating'])
    radius = max(2, min(15, row['count_reviews'] ** 0.5))
    
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=radius, 
        color=color,
        fill=True,

        popup=(
            f"Gmap ID: {row['gmap_id']}<br>"
            f"Business: {row['name']}<br>"
            f"Average Rating: {row['average_rating']:.1f}<br>"
            f"Number of Reviews: {row['count_reviews']}"
        )
    ).add_to(business_map)

# Add legend 
colormap.caption = 'Average Rating'
colormap.add_to(business_map)


business_map

merged_df["price"].unique()

def encode_price(price): # Label Encoding Price 
    encoding_map = {
        '₩': 1,  # We assume here that '₩' is equivalent to '$'
        '₩₩': 2,
        '$': 1,
        '$$': 2,
        '$$$': 3
    }
    return encoding_map.get(price, np.nan)  

# Aggregate based on price 
merged_df['encoded_price'] = merged_df['price'].apply(encode_price)
aggregated_df = merged_df.groupby(['encoded_price']).agg(
    count_price=('encoded_price', 'size'),
    average_review=('review_rating', 'mean')
).reset_index()

# Inverse mapping for price indiactor 
inverse_encoding_map = {1: r'₩', 2: r'₩₩', 3: r'₩₩₩'} # '$' has parsing conflicts matplotlib 
aggregated_df['price_indicator'] = aggregated_df['encoded_price'].map(inverse_encoding_map)


plt.figure(figsize=(12, 8))
# Generate Bar Chart 
sns.barplot(data=aggregated_df, y='average_review', x='price_indicator', palette='viridis')

# Add labels on bars correctly aligned
for index, row in aggregated_df.iterrows():
    plt.text(x=index, y=row['average_review'], s=f'{row["average_review"]:.2f}', 
             va='bottom', ha='center', fontsize=14)
    
# Scale y Axis 
plt.ylim(aggregated_df['average_review'].min() - 0.5, aggregated_df['average_review'].max() + 0.5)

# Customize the plot
plt.title('Average Business Rating by Price Indicator', fontsize=16)
plt.xlabel('Price Indicator', fontsize=14)
plt.xticks(fontsize=12)
plt.ylabel('Average Rating', fontsize=14)
plt.yticks(fontsize=12)
plt.show()

# Reinitialising merged_df to remove unwanted columns 
merged_df = pd.merge(review_df, meta_df, on='gmap_id', how='left')
columns = ['avg_rating', 'num_of_reviews', 'state', 'relative_results']
merged_df = merged_df.drop(columns=columns)

merged_df['category']

# Extracts categories from rows 
categories = [category for sublist in merged_df['category'].dropna() for category in sublist]
#print(categories)
# Create Series to count categories 
category_counts = pd.Series(categories).value_counts()
print(category_counts)

# Create a dictionary containing category counts
category_count_dict = category_counts.to_dict()

# Define a function to replace categories with the most frequent one
def assign_most_frequent_category(categories):
    if not categories:
        return None
    if len(categories) == 1:
        return categories[0]
    # Find the most frequently appearing category
    sorted_categories = sorted(categories, key=lambda x: category_count_dict.get(x, 0), reverse=True)
    #print(sorted_categories)
    return sorted_categories[0]

# Transform category to most frequent category
merged_df['most_frequent_category'] = merged_df['category'].apply(lambda x: assign_most_frequent_category(x))

# Aggregate data based on new category 
aggregated_df = merged_df.groupby(['most_frequent_category']).agg(
    count=('most_frequent_category', 'size'),
    average_review=('review_rating', 'mean')
).reset_index()

# Sort and filter
aggregated_df = aggregated_df.sort_values(by='average_review', ascending=False)
top_5 = aggregated_df.head(5)
bottom_5 = aggregated_df.tail(5)
combined = pd.concat([top_5, bottom_5])

# Define color code
colors = ['steelblue'] * 5 + ['orange'] * 5  # Blue for top 5, Orange for bottom 5


plt.figure(figsize=(12, 8))
# Generate bar chart 
sns.barplot(data=combined, y='most_frequent_category', x='average_review', palette=colors)

# Add labels
for index, value in enumerate(combined['average_review']):
    plt.text(value, index, f'{value:.2f}', va='center', ha='left', fontsize=10, color='black')

plt.title('Top and Bottom 5 categories by Average Review Rating', fontsize=16)
plt.xlabel('Average Review Rating', fontsize=14)
plt.ylabel('Category', fontsize=14)

plt.tight_layout()
plt.show()

# Reinitialise aggregated dataframe 
aggregated_review_df = review_df.groupby('gmap_id').agg(
    review_count=('gmap_id', 'size'),
    user_count = ('user_id', 'size'),
    time_count = ('time', 'median'),
    avg_rating = ('review_rating', 'mean'),
    review_text_count=('review_text', lambda x: (x != 'none').sum()),
    pic_count = ('if_pic', lambda x: (x == 'Y').sum()),
    response_count=('if_response', lambda x: (x == 'Y').sum()),

)

# Read the countvec_test.txt file
with open('countvec_test.txt', 'r') as file:
    count_vec_data = file.readlines()
# Convert to Dataframe 
count_vec_df = pd.DataFrame([line.strip().split() for line in count_vec_data])
count_vec_df.rename(columns ={0: 'gmap_id'}, inplace=True)

print(count_vec_df.head())

# Merge Dataframes
combined_vec_df = pd.merge(count_vec_df, aggregated_review_df, on='gmap_id', how='left')
combined_vec_df.head()

# Convert vocab list to dictionary 
word_dict = {}
with open('test_vocab.txt', 'r') as file:
    for line in file:
        word, index = line.strip().split(':')
        word_dict[index] = word

# Initialize dictionaries for word counts and ratings 
word_count = defaultdict(int)
word_rating_sum = defaultdict(float)

# Iterate through each row in the DataFrame
for column, row in combined_vec_df.iterrows():
    if pd.notna(row['avg_rating']):
        avg_rating = row['avg_rating']
        
        # Process each word count column
        for col in combined_vec_df.columns[1:1271]:
            value = str(row[col])
            for char in value.split(','):
                if ':' in char:
                    index, count = char.split(':')
                    word = word_dict.get(index)  
                    word_count[word] += int(count)
                    word_rating_sum[word] += avg_rating * int(count)

# Calculate average ratings for each word
word_avg_rating = {
    word: word_rating_sum[word] / word_count[word]
    for word in word_count
}

# Convert the results into a DataFrame
df_word_ratings = pd.DataFrame(
    list(word_avg_rating.items()), columns=['word', 'average_rating']
)

df_word_ratings['count'] = df_word_ratings['word'].map(word_count)

# Filter for words that appear more than ten times 
df_word_ratings = df_word_ratings[df_word_ratings['count'] > 10]

# Get the top 20 words with the highest and lowest average ratings
top_positive_words = df_word_ratings.sort_values(by='average_rating', ascending=False).head(20)
top_negative_words = df_word_ratings.sort_values(by='average_rating', ascending=True).head(20)

# Plotting for positive words
plt.figure(figsize=(12, 8))
sns.barplot(data=top_positive_words, x='word', y='average_rating', palette='viridis')
plt.title('Top 20 Most Positively Impactful Words', fontsize=16)
plt.xlabel('Word', fontsize=14)
plt.ylabel('Average Rating', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting for negative words
plt.figure(figsize=(12, 8))
sns.barplot(data=top_negative_words, x='word', y='average_rating', palette='magma')
plt.title('Top 20 Most Negatively Impactful Words', fontsize=16)
plt.xlabel('Word', fontsize=14)
plt.ylabel('Average Rating', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()