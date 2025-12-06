#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


# ========== Load Data ==========
df = pd.read_csv('E:/courses/data analysis/Advertising data analysis/Egypt_Ads.csv')


# ## data cleaning

# In[4]:


# ========== Data Cleaning ==========
df.fillna({
    "impression_count_per_user": 0,
    "clicked": 0,
    "converted": 0,
    "spend_egp": 0,
    "conversion_value_egp": 0,
    "ad_text": "",
    "platform": "Unknown",
    "ad_type": "Unknown",
    "ad_category": "Unknown",
    "age": "Unknown",
    "gender": "Unknown",
    "city": "Unknown",
    "device_type": "Unknown",
    "os": "Unknown",
    "traffic_source": "Unknown"
}, inplace=True)

# Remove negative or extreme outliers
df = df[(df["impression_count_per_user"] >= 0) & 
        (df["clicked"] >= 0) & 
        (df["converted"] >= 0) & 
        (df["spend_egp"] >= 0) & 
        (df["conversion_value_egp"] >= 0)]

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["day"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.day_name()


# In[8]:


# ========== Helper function ==========
def compute_metrics(group):
    group["CTR"] = group["clicked"] / group["impression_count_per_user"].replace(0,1)
    group["Conversion_Rate"] = group["converted"] / group["clicked"].replace(0,1)
    group["ROAS"] = group["conversion_value_egp"] / group["spend_egp"].replace(0,1)
    group["CPC"] = group["spend_egp"] / group["clicked"].replace(0,1)
    group["CPM"] = group["spend_egp"] / group["impression_count_per_user"].replace(0,1) * 1000
    return group


# In[9]:


# ========== Descriptive Analysis ==========
descriptive_summary = df.groupby(["platform", "ad_type", "ad_category"]).agg(
    impression_count_per_user=("impression_count_per_user","sum"),
    clicked=("clicked","sum"),
    converted=("converted","sum"),
    spend_egp=("spend_egp","sum"),
    conversion_value_egp=("conversion_value_egp","sum")
).reset_index()

descriptive_summary = compute_metrics(descriptive_summary)


# In[10]:


# ========== Demographic Analysis ==========
demo_summary = df.groupby(["age","gender","city"]).agg(
    impression_count_per_user=("impression_count_per_user","sum"),
    clicked=("clicked","sum"),
    converted=("converted","sum"),
    spend_egp=("spend_egp","sum"),
    conversion_value_egp=("conversion_value_egp","sum")
).reset_index()

demo_summary = compute_metrics(demo_summary)


# In[11]:


# ========== Platform & Ad Type Analysis ==========
platform_summary = df.groupby("platform").agg(
    impression_count_per_user=("impression_count_per_user","sum"),
    clicked=("clicked","sum"),
    converted=("converted","sum"),
    spend_egp=("spend_egp","sum"),
    conversion_value_egp=("conversion_value_egp","sum")
).reset_index()

platform_summary = compute_metrics(platform_summary)

adtype_summary = df.groupby("ad_type").agg(
    impression_count_per_user=("impression_count_per_user","sum"),
    clicked=("clicked","sum"),
    converted=("converted","sum"),
    spend_egp=("spend_egp","sum"),
    conversion_value_egp=("conversion_value_egp","sum")
).reset_index()

adtype_summary = compute_metrics(adtype_summary)


# In[12]:


# ========== Campaign & Ad Analysis ==========
campaign_summary = df.groupby("campaign_id").agg(
    impression_count_per_user=("impression_count_per_user","sum"),
    clicked=("clicked","sum"),
    converted=("converted","sum"),
    spend_egp=("spend_egp","sum"),
    conversion_value_egp=("conversion_value_egp","sum")
).reset_index()

campaign_summary = compute_metrics(campaign_summary)

ad_summary = df.groupby("ad_id").agg(
    impression_count_per_user=("impression_count_per_user","sum"),
    clicked=("clicked","sum"),
    converted=("converted","sum"),
    spend_egp=("spend_egp","sum"),
    conversion_value_egp=("conversion_value_egp","sum")
).reset_index()

ad_summary = compute_metrics(ad_summary)


# In[16]:


# ========== Text Analysis ==========
df["ad_text_clean"] = df["ad_text"].astype(str).str.lower()

text_summary = df.groupby("ad_text_clean").agg(
    impression_count_per_user=("impression_count_per_user","sum"),
    clicked=("clicked","sum"),
    converted=("converted","sum"),
    spend_egp=("spend_egp","sum"),
    conversion_value_egp=("conversion_value_egp","sum")   # الحل هنا
).reset_index()

text_summary = compute_metrics(text_summary)

# Word frequency
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["ad_text_clean"])
word_counts = pd.DataFrame({
    "word": vectorizer.get_feature_names_out(),
    "count": X.toarray().sum(axis=0)
}).sort_values("count", ascending=False)


# In[14]:


# ========== Device & Traffic Analysis ==========
device_summary = df.groupby(["device_type","os"]).agg(
    impression_count_per_user=("impression_count_per_user","sum"),
    clicked=("clicked","sum"),
    converted=("converted","sum"),
    spend_egp=("spend_egp","sum"),
    conversion_value_egp=("conversion_value_egp","sum")
).reset_index()

device_summary = compute_metrics(device_summary)

traffic_summary = df.groupby(["platform","traffic_source","age","gender"]).agg(
    impression_count_per_user=("impression_count_per_user","sum"),
    clicked=("clicked","sum"),
    converted=("converted","sum"),
    spend_egp=("spend_egp","sum"),
    conversion_value_egp=("conversion_value_egp","sum")
).reset_index()

traffic_summary = compute_metrics(traffic_summary)


# In[15]:


# ========== Time Analysis ==========
daily_summary = df.groupby("day").agg(
    impression_count_per_user=("impression_count_per_user","sum"),
    clicked=("clicked","sum"),
    converted=("converted","sum"),
    spend_egp=("spend_egp","sum"),
    conversion_value_egp=("conversion_value_egp","sum")
).reset_index()

daily_summary = compute_metrics(daily_summary)

hourly_summary = df.groupby("hour").agg(
    impression_count_per_user=("impression_count_per_user","sum"),
    clicked=("clicked","sum"),
    converted=("converted","sum"),
    spend_egp=("spend_egp","sum"),
    conversion_value_egp=("conversion_value_egp","sum")
).reset_index()

hourly_summary = compute_metrics(hourly_summary)

weekday_summary = df.groupby("weekday").agg(
    impression_count_per_user=("impression_count_per_user","sum"),
    clicked=("clicked","sum"),
    converted=("converted","sum"),
    spend_egp=("spend_egp","sum"),
    conversion_value_egp=("conversion_value_egp","sum")
).reset_index()

weekday_summary = compute_metrics(weekday_summary)

# Peak hours & conversions
peak_click_hours = hourly_summary.sort_values("clicked", ascending=False).head(5)
peak_conversion_hours = hourly_summary.sort_values("converted", ascending=False).head(5)


# In[19]:


# ========== Save results for Power BI ==========
descriptive_summary.to_csv("Descriptive_Summary.csv", index=False)
demo_summary.to_csv("Demographic_Summary.csv", index=False)
platform_summary.to_csv("Platform_Summary.csv", index=False)
adtype_summary.to_csv("AdType_Summary.csv", index=False)
campaign_summary.to_csv("Campaign_Summary.csv", index=False)
ad_summary.to_csv("Ad_Summary.csv", index=False)
text_summary.to_csv("Text_Summary.csv", index=False)
word_counts.to_csv("Word_Counts.csv", index=False)
device_summary.to_csv("Device_Summary.csv", index=False)
traffic_summary.to_csv("Traffic_Summary.csv", index=False)
daily_summary.to_csv("Daily_Summary.csv", index=False)
hourly_summary.to_csv("Hourly_Summary.csv", index=False)
weekday_summary.to_csv("Weekday_Summary.csv", index=False)

print("تم إعداد جميع الجداول جاهزة للاستخدام في Power BI")


# In[ ]:





# ## 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




