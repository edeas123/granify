
# coding: utf-8

# In[1]:

# import required packages
import pyspark.sql.functions as f

# load data files
sessions = spark.read.json("sessions.gz")
features = spark.read.json("features.gz")
orders = spark.read.json("orders.gz")


# In[2]:

# cleaning files
# data cleaning
# 1- remove ssid duplicates in sessions and features dataset
sessions = sessions.drop_duplicates(["ssid"])
features = features.drop_duplicates(["ssid"])

# 2- replace unavailable sessions browser with "UNK"
sessions = sessions.fillna("UNK", ["browser"])

# 3 - remove x-character starting some of the ssid in orders dataset
orders = orders.withColumn("ssid", f.regexp_replace(orders.ssid, "^x", ""))


# In[6]:

# join all dataset by the ssid
data = sessions.join(features, on="ssid").join(orders, on="ssid", how="left")

# drop irrelevant columns
data = data.drop("st")

# split ssid into user_id, site_id, start_time
data = data.withColumn("site_id", data.ssid.substr(38,1))
data = data.withColumn("start_hour", f.from_unixtime(data.ssid.substr(40,10), format="yyyy-MM-dd HH"))

# cache data
data = data.cache()


# In[18]:

# prepare the report grouped by start_hour, site_id, gr, ad, and browser
# sessions, evenues, transactions and conversions

report_1 = data.groupby(["start_hour", "site_id", "gr", "ad", "browser"]) \
        .agg(f.countDistinct(data.ssid).alias("#sessions"), # unique session id
        f.countDistinct(f.when(f.isnull(data.revenue), None).otherwise(data.ssid)).alias("#conversion"),  # unique session id where revenue is not null
        f.sum(f.when(f.isnull(data.revenue), 0).otherwise(1)).alias("#transactions"),  # count where revenue is not null
        f.format_number(f.sum(f.when(f.isnull(data.revenue), 0).otherwise(data.revenue)), 2).alias("revenues")) # sum orders

report_1.show()
report_1.toPandas().to_csv("granify.tsv", sep=",", header=True, index=False)
#report_1.coalesce(1).write.csv(path="granify", sep="\t", mode="overwrite", header=True)


# In[20]:

# a list of means and standard deviations for each feature per every (site_id, ad) pair
# prepare the report of feature means and standard deviations grouped by site_id, ad
report_2 = data.groupby(["site_id", "ad"]) \
            .agg(f.mean(data['feature-1']).alias("f1_mean"),
                f.mean(data['feature-2']).alias("f2_mean"),
                f.mean(data['feature-3']).alias("f_mean"),
                f.mean(data['feature-4']).alias("feature-4_mean"),
                f.stddev(data['feature-1']).alias("feature-1_std"),
                f.stddev(data['feature-2']).alias("feature-2_std"),
                f.stddev(data['feature-3']).alias("feature-3_std"),
                f.stddev(data['feature-4']).alias("feature-4_std"))
    
report_2.show()
report_2.toPandas().to_json("granify.json",orient="records", lines=True)


# In[ ]:



