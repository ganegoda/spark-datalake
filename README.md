# Data Warehouse with Redshift

## Introduction
A music streaming startup, Sparkify, has grown their user base and song database. To better serve their users, sparkify is planning to move their processes and data onto the cloud. Sparkify's data files reside in amazon Simple Storage Service (s3). There are two primary datasets, a set of JSON logs on user activity and another with JSON metadata on songs in sparkify app.

This project is focused on sparkify cloud migration, where datasets are loaded into parquet files in amazon s3. Database schema are designed based on performance optimization of queries provided by the analytical team. songplay table is the fact table whereas songs, artists, users, and time are dimension tables. 

Two datasets from s3 are loaded and processed using Apache Spark. Amazon Elastic Map Reduce (EMR) cluster with spark has been utilized in the Extract, Transform, and Load (ETL) process. Data residing in s3 buckets were first loaded into spark dataframes, then processed based on table schema, and finally saved into parquet files in a different s3 bucket.

### Song Dataset
The song dataset is a subset of real data from the [**Million Song Dataset**](http://millionsongdataset.com). Song data resides in following s3 location:
```bash
s3://udacity-dend/song_data
```
The files are partitioned by the first three letters of each song's track ID. Example below shows the path to two files in the song dataset.
```bash
s3://udacity-dend/song_data/A/B/C/TRABCEI128F424C983.json
s3://udacity-dend/song_data/A/A/B/TRAABJL12903CDCF1A.json
```
JSON files contain song metadata in following format:
```json
{
 "num_songs": 1, 
 "artist_id": "ARBGXIG122988F409D", 
 "artist_latitude": 37.77916, 
 "artist_longitude": -122.42005, 
 "artist_location": "California - SF", 
 "artist_name": "Steel Rain", 
 "song_id": "SOOJPRH12A8C141995", 
 "title": "Loaded Like A Gun", 
 "duration": 173.19138, 
 "year": 0
}
```

### Log Dataset
Log dataset is generated by an **event simulator** based on the song dataset described above. Simulated app activity logs contain data in JSON format, that are generated based on configuration settings. Event log datasets are partitioned by year and month, as shown in the example below:
```bash
s3://udacity-dend/log_data/2018/11/2018-11-12-events.json
s3://udacity-dend/log_data/2018/11/2018-11-13-events.json
```
A single log file contains multiple records of data, each in the foramt shown below:
```json
{
    "artist":"Blue October \/ Imogen Heap",
    "auth":"Logged In",
    "firstName":"Kaylee",
    "gender":"F","itemInSession":7,
    "lastName":"Summers","length":241.3971,
    "level":"free","location":"Phoenix-Mesa-Scottsdale, AZ",
    "method":"PUT",
    "page":"NextSong",
    "registration":1540344794796.0,
    "sessionId":139,
    "song":"Congratulations",
    "status":200,"ts":1541107493796,
    "userAgent":"\"Mozilla\/5.0 (Windows NT 6.1; WOW64) AppleWebKit\/537.36 (KHTML, like Gecko) Chrome\/35.0.1916.153 Safari\/537.36\"",
    "userId":"8"
}

```

### Staging dataframes
Two staging dataframes song_df and log_df were used to store raw data read from s3 buckets. Dataframes were then modified to satisfy additional data requirements of final tables.

1. **song_df** - Stores data from event logs
    - *artist, auth, first_name, gender, items_in_session, last_name, length, level, location, method, page, registration, session_id, song, status, time_stamp, user_agent, user_id*
1. **log_df** - Stores songs information from songs files
    - *num_songs, artist_id, artist_latitude, artist_longitude, artist_location, artist_name, song_id, title, duration, year*


### Schema for Song Play Analytics
Star schema is suitable for this particular use case, which optimizes the song data analytics queries.

#### Fact Table
1. **songplays.parquet (partitioned by year and month)** - Records in log data associated with song play activities filtered by *page = NextSong*.
   - *songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location, user_agent*
    

#### Dimension Tables
1. **users.parquet** - Users of the app
    - *user_id, first_name, last_name, gender, level*
   
1. **songs.parquet (partitioned by year and artist)** - Songs available in the app
    - *song_id, title, artist_id, year, duration*
   
1. **artists.parquet** - Artist of songs
    - *artist_id, name, location, latitude, longitude*
   
1. **time.parquet** - Timestamps of songplay events in specific units
    - *start_time, hour, day, week, month, year, weekday*



## ETL Pipeline
The ```etl.py``` script implements an ETL pipeline to extract data from s3, load into staging dataframes, transform and write to parquet files in destination s3 bucket. Credentials for s3 bucket access are stored in dl.cfg, which is used by ```etl.py``` for s3 bucket access permissions.

## How to Load data
1. Edit the dl.cfg file based on your user credentials for accessing s3 bucket. You may have to open port 22 of the master node for SSH access. 

    ```bash
    [AWS]
    AWS_ACCESS_KEY_ID=Your_Key_ID
    AWS_SECRET_ACCESS_KEY=Your_Secret_Key
    ```
    <br />
1. Create an EMR cluster using AWS Web Interface or AWS CLI.
    <br />
1. When cluster is ready, execute following commands modified based on your configuration:
   
    - Copy your files to master node:
    ```bash
    scp -i your_key_file.pem etl.py hadoop@ec2-XXX-XXX-XXX-XXX.us-west-2.compute.amazonaws.com:/home/hadoop/.
    scp -i your_key_file.pem dl.cfg hadoop@ec2-XXX-XXX-XXX-XXX.us-west-2.compute.amazonaws.com:/home/hadoop/.
    
    ```
    <br />
    - Log in to master node
    ```bash 
    ssh -i your_key_file.pem hadoop@ec2-XXX-XXX-XXX-XXX.us-west-2.compute.amazonaws.com
    ```
    <br />
    - Run etl.py script 
    ```bash
    spark-submit --conf spark.dynamicAllocation.enabled=false etl.py
    ```
    




