#!/usr/bin/env python
# coding: utf-8

#    ### EXPLORATORY DATA ANALYSIS AND PRICE ESTIMATION OF RESENDENTIAL RENT IN CALIFORNIA
#        

# **Library functions are imported**

# In[1]:


import warnings
warnings.filterwarnings(("ignore"))


# In[2]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


# **CSV file is Read**

# In[3]:


pd.set_option("display.max_columns", None) # enabling the option to inspect all accessible columns in the dataset while verifying
USA_Residence=pd.read_csv("./housing.csv") 
USA_Residence.head() #Examining the read file


# In[4]:


USA_Residence.dtypes #Analyzing the datatypes


# In[5]:


USA_Residence.shape #calculating the number of rows and columns


# In[6]:


USA_Residence.info() #Evaluating the information


# In[7]:


USA_Residence.describe() #examining each column's statistics


# The mean value of each column and the particular median value has huge difference and so we can be confident that there are many outliers in this dataset

# In[8]:


USA_Residence.price.value_counts() # Value counts are checked


# **STEP 1: DATA CLEANING**

# **Replacing the null and missing values with mode and mean**

# In[9]:


USA_Residence['laundry_options']=USA_Residence['laundry_options'].fillna(USA_Residence['laundry_options'].mode()[0]) # replacing with mode value
USA_Residence['parking_options']=USA_Residence['parking_options'].fillna(USA_Residence['parking_options'].mode()[0])
USA_Residence['lat']=USA_Residence['lat'].fillna(USA_Residence['lat'].mean()) # replacing lat and long with mean value
USA_Residence['long']=USA_Residence['long'].fillna(USA_Residence['long'].mean())
USA_Residence.isnull().sum() #check again....


# ** Taking the rent values between 1000 to 7500**

# In[10]:


USA_Residence = USA_Residence[(USA_Residence['price'] >= 1000) & (USA_Residence['price'] <= 7500)]


# **Dropping out the coloumns which are unnecessary**

# In[11]:


USA_Residence.drop(columns={"url","id","region_url","image_url","description"},inplace=True)


# ** Only for California..**

# In[12]:


california_res = USA_Residence[USA_Residence.state=='ca']
california_res #Reading all the California Data


# In[13]:


# Infering the shape of california dataset
california_res.shape


# **check for null or any missing values:**

# In[14]:


california_res.isnull().sum()


# In[15]:


california_res.region.unique() #Looking for regions in california


# **Using folium library to clean the Data using lat and Long**

# In[16]:


import folium # Folium easy to visualize the Geospatial Data
from folium import plugins 
California_Heatmap = folium.Map([36, -119], zoom_start = 6)
California_Heatmap_housing = california_res[np.isfinite(california_res['lat'])]
heatmap_with_location = California_Heatmap_housing[["lat", "long"]]
California_Heatmap.add_child(plugins.HeatMap(heatmap_with_location, radius = 15)) #Heatmap for all regions


# ** We can see the houses listed outside of the california we need to clean them out!!**

# In[30]:


# Filter the regions of California alone using Lat and Long :
california_res = california_res[(california_res['lat'] >= 32) & (california_res['lat'] <= 40.5)]
california_res = california_res[(california_res['long'] >= -120.41 ) & (california_res['long'] <= 172.5)]
california_res #Checking the dataframe with only California records


# In[31]:


california_res.shape #check size of matrix


# 
# 
# **Much of the wrong Data has been filtered out**

# **Look Map after filtering:**

# In[32]:


import folium
from folium import plugins
California_Heatmap = folium.Map(location=[36.7783,-119.4179],
                    zoom_start = 6, min_zoom=5)
California_Heatmap_housing = california_res[np.isfinite(california_res['long'])]
heatmap_with_location = California_Heatmap_housing[["lat", "long"]]
California_Heatmap.add_child(plugins.HeatMap(heatmap_with_location, radius = 40))


# In[34]:


# Statistical Measures for California:
california_res.describe()


# **Filter the data upon Sqfeet,number of beds,baths**

# **Size of the house** is assumed to be between **200 and 5000 sqfeet**

# In[35]:


california_res = california_res[(california_res['sqfeet'] >= 350) & (california_res['sqfeet'] <= 6500)]
california_res.describe() #check the data was intended or not


# In[36]:


california_res.shape #check for the size of the matrix


# After filtering data using size of the house we got 

# In[37]:


california_res.drop(columns={"state"},inplace=True) #Drop out the state column


# ** EXPLORATORY DATA ANALYSIS:**

# **Correlation Coefficients heat map for all pairings of variables**

# In[38]:


plt.figure(figsize=(20,12)) # increasing the plot size
sns.heatmap(data=california_res.corr(), cmap='GnBu', annot=True, annot_kws = {'fontsize':18}) # generating heatmap on the coefficients with values shown
plt.title("Heat map on correlation coefficients for all pairs of variables", 
          fontsize = 20)# marking legends 
plt.xlabel('',fontsize=18)
plt.ylabel('',fontsize=18)
plt.show() # unwanted line spacing are removed along the output


# We can see from this Correlation matrix that the price is connected with the other variables in the following order.

#   1. Size of the house (Correlation coefficient value = 0.47) <br>2. Number of baths(Correlation coefficient value = 0.35) <br>3. Electric vehicle charge(Correlation coefficient value = 0.3) 

# There are no good correlations correlations found from the matrix

# **QUESTION:What is the most common pricing range for houses?**

# **Analysis using Histogram:**

# In[39]:


plt.figure(figsize=(20,10)) #plot size is increased
plt.hist(california_res["price"], 
         bins=100, 
         edgecolor="k", 
         facecolor="blue",
         alpha=0.6,
        linewidth=0.6) # characterization based on histogram
plt.xlabel("Rent per month (in dollars)", fontsize= 18) # x-axis is labelled
plt.ylabel("The number of homes for sale", fontsize = 18) # y-axis is labelled
plt.title("Price Distribution Histogram", fontsize = 24) # #marking legends for the plot
plt.show() # line are negated from the output


# **ANSWER: The majority of properties have monthly prices ranging from $1000  to $2000 USD.**

# **QUESTION :What is the average size of the majority of the residences listed?**

# In[40]:


plt.figure(figsize=(20,10)) 
plt.hist(california_res["sqfeet"], 
         bins=50, 
         edgecolor="k", 
         facecolor="green",
         alpha=0.6,
        linewidth=0.6) 
plt.xlabel("House Dimensions (in Square feet)", fontsize= 18) #x-axis is labelled
plt.ylabel("the amount of records", fontsize = 18) #y-axis is labelled
plt.title("House Size Distribution Histogram (In square feet)", fontsize = 24) #marking legends for the plot
plt.show() 


# **ANSWER: The maximum number of houses listed are between the size of 600 to 1200 sq feet**

# **QUESTION: How many bathrooms do the majority of the homes for sale have?**

# In[41]:


plt.figure(figsize=(20,10))
sns.countplot(x=california_res['baths'])
plt.xlabel("Total bathrooms", fontsize= 18) #x-axis is labelled
plt.ylabel("Total houses for sale", fontsize = 18) #y-axis is labelled
plt.title("Number of baths distribution", fontsize = 20) #marking legends for the plot
plt.show()


# **ANSWER: Most of the houses listed in California has only 2 baths** 

# **QUESTION: How many houses listed have electric vehicle charge stations?**

# In[44]:


plt.figure(figsize=(20,10))
sns.countplot(x=california_res['electric_vehicle_charge'])
plt.xlabel("Electric vehicle stations", fontsize= 18) #x-axis is labelled
plt.ylabel("Houses Listed ", fontsize = 18) #y-axis is labelled
plt.title("EV charge station distribution", fontsize = 20) #marking legends for the plot
plt.show()


# **ANSWER: 500 houses listed have electric vehicle charge points in their houses? ** 

# **QUESTION:What are the most common types of parking alternatives in California?**

# In[49]:


plt.figure(figsize=(20,10))
sns.countplot(x=california_res['parking_options'])
plt.xlabel("Availability of Parking", fontsize= 18) #x-axis is labelled
plt.ylabel("Houses listed", fontsize = 18) #y-axis is labelled
plt.title("Availability of car Parking distribution", fontsize = 20) # marking legends for the plot
plt.show()


# **ANSWER:Car port parking is a popular parking option in California, with over 8500 residences offering it.**

# **QUESTION: What is the proportion of residences that are furnished?**

# In[51]:


plt.figure(figsize=(20,10))
sns.countplot(x=california_res['comes_furnished'])
plt.xlabel("Whether Furnished or Unfurnished?", fontsize= 18) #x-axis is labelled
plt.ylabel("Houses listed", fontsize = 18) #y-axis is labelled
plt.title("Furnished or Unfurnished distribution", fontsize = 20) #marking legends for the plot
plt.show()


# In[52]:


(california_res['comes_furnished'].value_counts()[1]) / (california_res['comes_furnished'].value_counts()[0])*100


# **ANSWER: In California, less than 3% of the total properties offered are furnished.**

# **MODELLING**

# **ANALYZING USING LINEAR REGRESSION METHODS**

# **Method 1: Price Estimation Based on House Size, Beds and electric_vehicle_charge:**

# In[54]:


import statsmodels.formula.api as smf #statstical model library is imported
model_1= smf.ols(formula = "price ~ sqfeet + beds + electric_vehicle_charge ",data = california_res).fit() # model is created
print(model_1.summary()) #overall summary is displayed


# **Calculating the Linear regression Analysis Statistics for Model Evaluation:**

# In[55]:


print("MSE:  ",model_1.mse_resid) # determing the Mean Squared Error
print("R2:  ",model_1.rsquared) # determining the Coefficient of Determination
print("R2_adj:  ",model_1.rsquared_adj) # determining the modified Coefficient of Determination


# **Method2: Predicting price based on all available variables**

# In[58]:


model_3= smf.ols(formula = "price ~ sqfeet + beds + baths + cats_allowed + dogs_allowed + smoking_allowed + wheelchair_access + electric_vehicle_charge + comes_furnished ",data = california_res).fit() # model is created 
print(model_3.summary()) #overall summary is displayed


# In[59]:


print("MSE:  ",model_3.mse_resid) # determing the Mean Squared Error
print("R2:  ",model_3.rsquared) # determining the Coefficient of Determination
print("R2_adj:  ",model_3.rsquared_adj) # determining the modified Coefficient of Determination


# **ENCODING CATEGORICAL COLUMNS FOR ANALYSIS:**

# In[60]:


california_res.head() #Idendfying the columns based on category


# In[61]:


from sklearn.preprocessing import LabelEncoder
Lab_Enc=LabelEncoder()
california_res_encoded=california_res


# In[62]:


california_res_encoded['region']=Lab_Enc.fit_transform(california_res['region'])
california_res_encoded['type']=Lab_Enc.fit_transform(california_res['type'])
california_res_encoded['laundry_options']=Lab_Enc.fit_transform(california_res['laundry_options'])
california_res_encoded['parking_options']=Lab_Enc.fit_transform(california_res['parking_options'])
california_res_encoded.head() #Examining the encoding of all five columns


# **Train and Test split**

# In[67]:



X=california_res_encoded.drop(columns=['price'])
y=california_res_encoded['price']


# In[68]:


#During analysis, all columns are scaled to ensure equal weightage distribution.
from sklearn.preprocessing import StandardScaler #import standard scaler
scalar=StandardScaler()
x_scaled=scalar.fit_transform(X)
x_scaled 


# In[69]:


X_version11 = california_res[['sqfeet' , 'beds' , 'baths' ,'cats_allowed' , 'dogs_allowed'  ,'smoking_allowed' ,'wheelchair_access' , 'electric_vehicle_charge' , 'comes_furnished']]
y_version1 = california_res['price']
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(x_scaled, 
                                                        y, 
                                                        test_size=.20, 
                                                        random_state = 69)
from sklearn import metrics
from sklearn import linear_model
li_reg = linear_model.LinearRegression()
li_reg.fit(X_train1, y_train1)
y_predict1 = li_reg.predict(X_test1)
Mean_square_E_version1 = metrics.mean_squared_error(y_test1, y_predict1)# mean squared error (square of the loss function)
print("MSE of Version 1:",Mean_square_E_version1)


# 

# In[70]:


r2_version1 = metrics.r2_score(y_test1, y_predict1)
print("R Squared of version 1:",r2_version1) 


# **RANDOM FOREST REGRESSION:**

# In[71]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_scaled,y,test_size=0.3,random_state=0) 


# In[72]:


from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators=100,random_state=5)
RFR.fit(X_train,y_train)


# **RFR EVALUATION:**

# In[73]:


y_estimate_forestreg = RFR.predict((X_test))
from sklearn import metrics
r_square = metrics.r2_score(y_test,y_estimate_forestreg)
print("Random Forest Model R Squared:",r_square) 


# **Displaying the model's True and Predicted values:**

# In[74]:


np.column_stack((y_test ,y_estimate_forestreg))# using column_stack


# In[75]:


mse = metrics.mean_squared_error(y_test,y_estimate_forestreg)
print("Random Forest Model Mean Square Error:",mse) 


# **DECISION TREE Version2:**

# In[76]:


from sklearn.tree import DecisionTreeRegressor
DTR= DecisionTreeRegressor(random_state=0)
DTR.fit(X_train,y_train) #Training the decision tree regression model


# In[77]:


y_predict_dtr=DTR.predict((X_test))
r_square1 = metrics.r2_score(y_test,y_predict_dtr)
print("Decision Tree Model R Squared:",r_square1) 


# In[78]:


mse1 = metrics.mean_squared_error(y_test,y_predict_dtr)
print("The Decision Tree Model's MSE is:",mse1) 


# **Showing the True and Predicted values from the model:**

# In[79]:


np.column_stack((y_test ,y_predict_dtr))


# **QUESTION:How does the average rent in California compare to the national average?**

# **California's Average Housing Rent Price:**

# In[81]:


california_res_meanprice=california_res['price'].mean()
california_res_meanprice


# The Mean price of housing rent in California = 1023.28 $

# In[ ]:




