import configparser
from datetime import datetime
import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, to_utc_timestamp, from_unixtime, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import DoubleType, LongType, IntegerType

# Read aws credentials from config file
config = configparser.ConfigParser()
config.read("dl.cfg")

os.environ['AWS_ACCESS_KEY_ID'] = config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    conf = (
        pyspark.SparkConf()
        .set("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0")
        .set("spark.hadoop.fs.s3a.awsAccessKeyId", config['AWS']['AWS_ACCESS_KEY_ID'])
        .set("spark.hadoop.fs.s3a.awsSecretAccessKey", config['AWS']['AWS_SECRET_ACCESS_KEY'])
        .set("fs.s3a.endpoint", "s3-us-west-2.amazonaws.com")
        .set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    )

    spark = (
        SparkSession
        .builder
        .config(conf=conf)
        .getOrCreate()
    )
    return spark


def process_song_data(spark, input_data, output_data):
    # get filepath to song data file
    # song_data = input_data+"song-data/A/A/Z/TRAAZYI12903CB2EE4.json" # Test1
    # Using a subset of data for testing and aws cost savings. Use "song-data/*/*/*/*.json" for full dataset
    song_data = os.path.join(input_data, "song-data/A/A/*/*.json")

    # read song data file
    song_df = spark.read.json(song_data)

    # Change column type
    song_df = (song_df.withColumn('artist_latitude', col('artist_latitude').cast(DoubleType()))
               .withColumn('artist_longitude', col('artist_longitude').cast(DoubleType()))
               )

    # extract columns to create artists table
    artists_table = (song_df.selectExpr(
        'artist_id', 'artist_name as name', 'artist_location as location',
        'artist_latitude as latitude', 'artist_longitude as longitude').dropDuplicates(['artist_id'])
    )

    # write artists table to parquet files
    artists_table.write.mode('overwrite').parquet(
        os.path.join(output_data, "artists.parquet"))

    # extract columns to create songs table
    songs_table = song_df.select('song_id', 'title', 'artist_id', song_df.year.cast(
        IntegerType()), 'duration').dropDuplicates(['song_id'])

    # write songs table to parquet files partitioned by year and artist
    songs_table.repartition("year", "artist_id").write.mode('overwrite').partitionBy(
        "year", "artist_id").parquet(os.path.join(output_data, "songs.parquet"))


def process_log_data(spark, input_data, output_data):

    # get filepath to log data file
    # log_data = 's3a://udacity-dend/log-data/2018/11/2018-11-01-events.json' # Use for testing
    log_data = os.path.join(input_data, "log-data/*/*/*.json")

    # read log data file
    log_df = spark.read.option("inferschema", "true").json(log_data)

    # Chage column names
    new_names = ['artist', 'auth', 'first_name', 'gender', 'items_in_session', 'last_name',
                 'length', 'level', 'location', 'method', 'page', 'registration', 'session_id',
                 'song', 'status', 'ts', 'user_agent', 'user_id']

    log_df = log_df.toDF(*new_names)

    # In JSON user_id is defiend as string. Chage that to integer
    log_df = log_df.withColumn('user_id', col('user_id').cast(LongType()))

    # Add new timestamp column by converting 'ts' column. Assumed 'PST' timezone
    log_df = log_df.withColumn("time_stamp", to_utc_timestamp(
        from_unixtime(col("ts")/1000, 'yyyy-MM-dd HH:mm:ss'), 'PST'))

    # filter by actions for song plays
    log_df_song_play = log_df.filter(log_df.page == 'NextSong')

    # extract columns for users table
    users_table = log_df_song_play.select(
        'user_id', 'first_name', 'last_name', 'gender', 'level').dropDuplicates(['user_id'])

    # write users table to parquet files
    users_table.write.mode('overwrite').parquet(
        os.path.join(output_data, "users.parquet"))

    # extract columns to create time table
    time_table = log_df_song_play.selectExpr('time_stamp', 'hour(time_stamp) as hour', 'day(time_stamp) as day',
                                             'weekofyear(time_stamp) as week', 'month(time_stamp) as month',
                                             'year(time_stamp) as year',
                                             'weekday(time_stamp) as weekday').dropDuplicates(['time_stamp'])

    # write time table to parquet files partitioned by year and month
    time_table.write.mode('overwrite').parquet(
        os.path.join(output_data, "times.parquet"))

    # read in song data to use for songplays table
    song_data_path = os.path.join(output_data, "songs.parquet")
    songs_df = spark.read.parquet(song_data_path)

    # read in song data to use for songplays table
    artist_data_path = os.path.join(output_data, "artists.parquet")
    artists_df = spark.read.parquet(artist_data_path)

    # join songs, artists, and log data tables to combine data required to create songplays table

    l = log_df_song_play.alias("l")
    a = artists_df.alias("a")
    s = songs_df.alias("s")

    songplays_df = l.join(a, col("l.artist") == col("a.name"), "left")\
        .join(s, [(col("l.song") == col("s.title")) & (col("l.length") == col("s.duration"))], "left")

    # extract columns from joined song and log datasets to create songplays table
    songplays_table = songplays_df.select(monotonically_increasing_id().alias('songplay_id'), 'l.time_stamp', 'l.user_id', 'l.level',
                                          's.song_id', 'a.artist_id', 'l.session_id', 'a.location', 'l.user_agent')
    songplays_table = songplays_table.withColumn("year", year(
        col("time_stamp"))).withColumn("month", month(col("time_stamp")))

    # write songplays table to parquet files partitioned by year and month
    songplays_table.repartition("year", "month").write.mode('overwrite').partitionBy(
        "year", "month").parquet(os.path.join(output_data, "songplays.parquet"))


def main():
    spark = create_spark_session()

    input_data = "s3a://udacity-dend/"

    # Input your bucket name here:
    output_data = "s3a://hasitha-datalake/sparkify/"

    # Write songs and artists tables to parquet file. Function returns artists & songs tables for further processing
    process_song_data(spark, input_data, output_data)

    process_log_data(spark, input_data, output_data)

    spark.stop()


if __name__ == "__main__":
    main()
