
################ Data Understanding ################

##### Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

##### Importing Data
df_control_group = pd.read_excel("datasets/ab_testing.xlsx", usecols=[0, 1, 2, 3], sheet_name="Control Group")
df_test_group = pd.read_excel("datasets/ab_testing.xlsx", usecols=[0, 1, 2, 3], sheet_name="Test Group")


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df_control_group)
check_df(df_test_group)


# Visualization
def create_displot(dataframe, col):
    sns.displot(data=dataframe, x=col, kde=True)
    plt.show()


def plot_summ(dataframe, dataframe2, col):
    plt.figure(figsize=(10, 8))
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.style.use('seaborn-deep')
    plt.hist([dataframe[col], dataframe2[col]], bins=15, label=["Control Group", "Test Group"])
    plt.legend(loc="upper right")
    plt.show()


[plot_summ(df_control_group, df_test_group, i) for i in df_control_group.columns]

# AB Testing (Independent Two-Sample T-Test)

# 1. Assumption Check
#   - 1. Normality Assumption
#   - 2. Variance Homogeneity
# 2. Implementation of the Hypothesis
#   - 1. Independent two-sample t-test if assumptions are met (parametric test)
#   - 2. Mannwhitneyu test if assumptions are not provided (non-parametric test)
# Not:
# -
# In cases where the assumption of normality is not met, non-parametric testing is applied.
# In cases where variance homogeneity is not provided, an argument (equal_var=False) is entered for the parametric test.

df_control_group["Purchase"].mean()  # Maximum Bidding
df_test_group["Purchase"].mean()  # Average Bidding

#
# As a result of the addition of this new feature, is there a statistically significant difference in the number of products purchased after the advertisements clicked on the website?
# H0: Satın alınan ürün sayısı arasında fark yoktur. (μ1 = μ2)
# H1: Satın alınan ürün sayısı arasında fark vardır. (μ1 != μ2)

# Assumption Check
#  1.Normality Assumption

# H0: There is no difference between the number of products purchased. (p-value < 0.05)
# H1: Normal distribution assumption not provided.

test_stat, pvalue = shapiro(df_control_group["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
create_displot(df_control_group, "Purchase")

# Since the pvalue is greater than 0.05, we cannot reject my Ho hypothesis. For this reason, I say it is the normal distribution for the 1st group.

test_stat, pvalue = shapiro(df_test_group["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
create_displot(df_test_group, "Purchase")

#
# Since the pvalue is greater than 0.05, we cannot reject my Ho hypothesis. For this reason, I say it is a normal distribution within the 2nd group..


# 2. Variance Homogeneity Assumption
# H0: Variances are Homogeneous
# H1: Variances Are Not Homogeneous
test_stat, pvalue = levene(df_control_group["Purchase"],
                           df_test_group["Purchase"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Since the pvalue is greater than 0.05, we cannot reject my Ho hypothesis. Variances are homogeneous. The distributions of the two groups are similar.


# Since the assumptions are met, an independent two-sample t-test (parametric test) will be applied..

test_stat, pvalue = ttest_ind(df_control_group["Purchase"],
                              df_test_group["Purchase"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Since the pvalue is greater than 0.05, we cannot reject my Ho hypothesis. There is no statistically significant difference in the number of products purchased after the advertisements clicked on the website.


# p-value < ise 0.05 'ten H0 can be rejected.
# p-value < değilse 0.05 H0 cannot be rejected.


####################### Conclusion #######################

""" As a result of the addition of this new feature, is there a statistically significant difference
in the number of products purchased after the advertisements clicked on the website?
H0: There is no difference between the number of products purchased. (μ1 = μ2)
H1: There is a difference between the number of products purchased. (μ1 != μ2)"""


""" As a result of the statistical inferences, there was a statistical difference between the control and test groups no significant difference was found. 
As a result of these tests, it can be said that the two groups are statistically similar to each other."""


""" The Shapiro test was applied to test whether the control and test groups conformed to the normality assumption.
 Since it provided the assumption of normality in both groups, the next step was the homogeneity of variance.

Levene test was used to observe the homogeneity of variance of the two groups. When the values obtained as a result of the test are examined
It was observed that there was a similarity between the homogeneity of variance of the groups. Since it also provides variance homogeneity, the difference between the two groups
Independent Two-Sample T-Test was applied to reveal statistical meanings.

The p-value obtained after the Independent Two-Sample T-Test was applied was examined.
H0 hypothesis could not be rejected because our value was not less than 0.05.
Even if there is a difference between the means of the control and test groups, it has been determined as a result of the analyzes that this is not statistically significant."""

"""A new test with more samples can be made to re-examine the new system added to the website."""
