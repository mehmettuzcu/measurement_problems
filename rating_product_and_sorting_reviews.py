################ Data Understanding ################

##### Importing Libraries

import pandas as pd
import math
import scipy.stats as st

# Making the appearance settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


##### Importing Data

df = pd.read_csv("datasets/amazon_review.csv")

df.shape  # Dimension of dataframe

df.dtypes  # Data type of each variable

df.info()  # Print a concise summary of a DataFrame

df.head()  # First 5 observations of dataframe

df.tail()  # Last 5 observations of dataframe

##### Data Preparation

df.isnull().sum()  # Get number of Null values in a dataframe

# Remove missing observations from the data set
df.dropna(inplace=True)

################# Task-1 #################

# Let's check the number of rating awarded to product
df["overall"].count()

# Let's check how many of each rating there
df["overall"].value_counts()

# Distributions in variable "day_diff"
df["day_diff"].quantile([0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1])
df["day_diff"].mean()  # There is no difference between mode and median. It is seen that the distribution is in a regular way.

# Since the data is regularly distributed, I divided the days into quarters and multiplied according to these intervals.
df['days_segment'] = pd.qcut(df["day_diff"], 4, labels=["A", "B", "C", "D"])

df.loc[df["days_segment"] == "A", "overall"].mean() * 28 / 100 + \
df.loc[df["days_segment"] == "B", "overall"].mean() * 26 / 100 + \
df.loc[df["days_segment"] == "C", "overall"].mean() * 24 / 100 + \
df.loc[df["days_segment"] == "D", "overall"].mean() * 22 / 100

df["overall"].mean()
# When we calculate the Time-Based and Average rating scores, the rating score in the Time-Based method is observed a little more.


################# Task-2 #################

# Sorting Reviews

df = pd.read_csv("datasets/amazon_review.csv")

# Since there is no information about those who do not like the comment in the data set, we create it ourselves.
df["down_rating"] = df["total_vote"] - df["helpful_yes"]

df.head()


# We'll cover 3 methods for sorting reviews.

# 1-) Score
# 2-) Average rating
# 3-) Wilson Lower Bound Score

# 1-) Score = (up ratings) − (down ratings)


def score_up_down_diff(up, down):
    return up - down



# 2-) Score = Average rating = (up ratings) / (down ratings)


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


# 3-) Wilson Lower Bound Score


def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


# Case Study:

# Score Different
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["down_rating"]), axis=1)
df.sort_values("score_pos_neg_diff", ascending=False)

# Average rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["down_rating"]), axis=1)
df.sort_values("score_average_rating", ascending=False)

# Wilson Lower Bound Score
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["down_rating"]), axis=1)
df.sort_values("wilson_lower_bound", ascending=False)


final_df = df.loc[:, ["reviewText", "total_vote", "down_rating", "score_pos_neg_diff",  "score_average_rating",  "wilson_lower_bound"]]
final_df.head()

# Top 20 sorted by Wilson Lower Bound score
final_df.sort_values("wilson_lower_bound", ascending=False).head(20)