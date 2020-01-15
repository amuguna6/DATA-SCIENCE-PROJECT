import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import markers
from numpy.core.tests.test_scalarinherit import C
from scipy.sparse.linalg.isolve.tests.demo_lgmres import x1, x2
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from random import randint

df = pd.read_csv('estatedata.csv')  # whole data extracted
print(df)
# fill the empty for basement
df['basement'].fillna(0.0, inplace=True)
# print("=====================")
# fill the empty for exterior_walls
df['exterior_walls'].fillna("N", inplace=True)

# # Fill empty for roof
df['roof'].fillna("N", inplace=True)
print(df.isnull().sum())

# replace string to integer property_type
df['property_type'].replace("Apartment / Condo / Townhouse", 1, inplace=True)
df['property_type'].replace("Single-Family", 2, inplace=True)
print(df.shape)

# replace string to integer exterior_walls
df['exterior_walls'].replace("Asbestos shingle", 1, inplace=True)
df['exterior_walls'].replace("Block", 2, inplace=True)
df['exterior_walls'].replace("Brick", 3, inplace=True)
df['exterior_walls'].replace("Brick veneer", 4, inplace=True)
df['exterior_walls'].replace("Combination", 5, inplace=True)
df['exterior_walls'].replace("Concrete", 6, inplace=True)
df['exterior_walls'].replace("Concrete Block", 7, inplace=True)
df['exterior_walls'].replace("Masonry", 8, inplace=True)
df['exterior_walls'].replace("Metal", 9, inplace=True)
df['exterior_walls'].replace("N", 10, inplace=True)
df['exterior_walls'].replace("Other", 11, inplace=True)
df['exterior_walls'].replace("Rock, Stone", 12, inplace=True)
df['exterior_walls'].replace("Siding (Alum/Vinyl)", 13, inplace=True)
df['exterior_walls'].replace("Stucco", 14, inplace=True)
df['exterior_walls'].replace("Wood", 15, inplace=True)
df['exterior_walls'].replace("Wood Siding", 17, inplace=True)
df['exterior_walls'].replace("Wood Shingle", 18, inplace=True)

# replace string to integer roof
df['roof'].replace("Asbestos", 1, inplace=True)
df['roof'].replace("Asphalt", 2, inplace=True)
df['roof'].replace("Built-up", 3, inplace=True)
df['roof'].replace("Composition", 4, inplace=True)
df['roof'].replace("Composition Shingle", 5, inplace=True)
df['roof'].replace("Gravel/Rock", 6, inplace=True)
df['roof'].replace("Metal", 7, inplace=True)
df['roof'].replace("N", 8, inplace=True)
df['roof'].replace("Other", 9, inplace=True)
df['roof'].replace("Roll Composition", 10, inplace=True)
df['roof'].replace("Shake Shingle", 11, inplace=True)
df['roof'].replace("Slate", 12, inplace=True)
df['roof'].replace("Wood Shake/ Shingles", 13, inplace=True)
df['roof'].replace("asphalt", 14, inplace=True)
df['roof'].replace("asphalt,shake-shingle", 15, inplace=True)
df['roof'].replace("composition", 16, inplace=True)
df['roof'].replace("shake-shingle", 17, inplace=True)

# Histogram Density
fig, ax = plt.subplots()
ax.hist(df['tx_price'], color="purple", bins=30),

ax.set_xlabel("tx_price for real estate in US")
ax.set_ylabel("Freq")
ax.set_title("Distribution of tx_price")
plt.show()
pandas.set_option('display.max_columns', 10)
print(df.describe())

# Scatter Plot
# Creating the plot object
fig, ax = plt.subplots()
# Plotting the data, setting the sizes color and transparency (alpha)alpha
ax.scatter(df['tx_year'], df['property_tax'], s=30, color='#539caf', alpha=0.75)
ax.set_title("Distribution of Year the transaction took place 'tx_year' by property_tax")
ax.set_xlabel("tx_year")
ax.set_ylabel("property_tax")
plt.show()

# Density Plot
fig, ax = plt.subplots()
ax.plot(df["median_age"], color="green", lw=2)
ax.set_ylabel("frequency")
ax.set_xlabel("Active_Life")
ax.set_title("Distribution of the median_age with active life for Real-Estate in US")
ax.legend(loc="best")
plt.show()

# seaborn heatmap

corr = (df.corr())
vmin = -1,
vmax = 1,
vars = ['tx_year', 'property_tax', 'insurance', 'restaurants', 'groceries', 'cafes', 'shopping', 'arts_entertainment',
        'beauty_spas']
sns.heatmap(corr.loc[vars, vars])
plt.xlabel("Values on X axis")
plt.show()

# Predict
array = df.values
X = array[:, 1:26]  # 0-26
Y = array[:, 0]  # target variable (OUTCOME) this is called supervised

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)

# Split to train, test
# X_train, Y_train will be 70%,
from sklearn import model_selection

# 0.30 is the test split %, for testing data
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(rescaledX, Y, test_size=0.30,
                                                                    random_state=20)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# ick one model
model = LinearRegression()
model.fit(X_train, Y_train)  # training process

# the model to predict the x_test , usually hide the y-test from the model DO NOT SHOW THE Y-TEST MODELS
predictions = model.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error

print("R aquared(%): ", r2_score(Y_test, predictions))
print("Mean square error: ", mean_squared_error(Y_test, predictions))

# Plot the scatter with best fit line X is the actual while Y is the predicted
fig, ax = plt.subplots()
ax.scatter(Y_test, predictions)
# line of the best fit THOSE THAT LIE ON THE POSITIVE PLOT LINE AND WHICH ARE FAR FROM THE LINE
ax.plot(Y_test, Y_test)
ax.set_xlabel("Y_test values (Real answer)")
ax.set_ylabel("Predicted values")
ax.set_title("Ytesy vs predictions")
plt.show()

# feature selection
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)

sub = df[['tx_price', 'year_built', 'median_age', 'property_tax', 'insurance']]
array = sub.values
X = array[:, 0:5]
# fit to k-means model the number of clusters are assigned to decide before the application

model = KMeans(n_clusters=5)
model.fit(X)
print(model)

# Use centronoids algorithm and load it to the pandas dataframe
centronoids = model.cluster_centers_
cluster = pandas.DataFrame(centronoids, columns=['tx_price', 'year_built', 'median_age', 'property_tax', 'insurance'])
print(cluster)

# these are people are in the mid thirties years. Need to understand these category seem to be an average age to acquire
# property into real estate in US
kmeans = KMeans(n_clusters=5).fit(X)
centronoids = kmeans.cluster_centers_
labels = kmeans.labels_
print(labels)
print(centronoids)
# Creating a new dataframe for the data
cluster = pandas.DataFrame(centronoids, columns=['tx_year', 'year_built', 'median_age', 'property_tax', 'insurance'])
print(cluster)
# Plotting
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=kmeans.labels_.astype(float))



def data():
    return [randint(0, 100) for _ in range(10)]


c1 = (data(), data(), data())
c2 = (data(), data(), data())
c3 = (data(), data(), data())
clusters = [c1, c2, c3]

# plot
colors = ['r', 'b', 'y', 'c']
for i, c in enumerate(clusters):
    ax.scatter(c[0], c[1], c[2], c=colors[i], label='cluster {}'.format(i))

ax.legend(bbox_to_anchor=(1.5, 1))
# Add title and axis names
plt.title('Distribution of Point of Clusters for Real Estate')
plt.xlabel('categories')
plt.ylabel('values')
plt.show()
