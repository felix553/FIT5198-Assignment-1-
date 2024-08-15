
import json
import pandas as pd

groupnum = input("Please input your group number:")

with open("task1_{}.json".format(groupnum.zfill(3)),"r",encoding='utf-8') as file:
        content=file.read()
try:
    data=json.loads(content)
except:
    raise TypeError("Invalid json: unable to load data!")

try:
    df=pd.DataFrame(data)
except Exception as e:
    raise ValueError("Invalid json data structure: check your data structure!")

df_index=["reviews","earliest_review_date","latest_review_date"]
assert len(df_index)==len(df.index), "Invalid json data structure:  check your data structure!"
for each in df_index:
    if each not in df.index:
          raise ValueError("key {} is missing".format(each))


df_reviews=pd.DataFrame(df.loc["reviews",:])
df_reviews.reset_index(inplace=True)
df_reviews.rename(columns={"index":"gmap_id"},inplace=True)
normalized_reviews=pd.json_normalize(df_reviews.to_dict(orient="records"),
        record_path=['reviews'],meta=["gmap_id"])

column_names=['user_id', 'time', 'review_rating', 'review_text', 'if_pic', 'pic_dim',
       'if_response', 'gmap_id']
assert len(column_names)==len(normalized_reviews.columns), "Invalid csv data structure: check your data structure!"
for each in column_names:
    if each not in normalized_reviews.columns:
        raise ValueError("key {} is missing".format(each))

print("Task 1 json file passed!")

df2 = pd.read_csv("task1_{}.csv".format(groupnum.zfill(3)))
df2_col=['gmap_id', 'review_count', 'review_text_count', 'response_count']
assert all(df2.columns == df2_col) == True, "check your csv columns!"
print("Task 1 csv file passed!")

