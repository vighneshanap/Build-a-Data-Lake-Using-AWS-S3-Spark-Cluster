import configparser
import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, from_unixtime, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import IntegerType, TimestampType, DateType


config = configparser.ConfigParser()
config.read('dl.cfg')

# get AWS credentials from dl.cfg file
os.environ['AWS_ACCESS_KEY_ID']=config.get('AWS','AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY']=config.get('AWS','AWS_SECRET_ACCESS_KEY')


def create_spark_session():
    """create and config instance of SparkSession
    
    returns: spark object 
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    
    """
    extracts song data from source s3 bucket in json format process it using spark to 
    extract song and artist data then load the result into parquet files in destination 
    s3 bucket 
    
    :param spark: allows to initiate SparkSession
    :param input_data: input local data or s3 bucket address 
    :param output_data: output stores locally or s3 bucket address provided
    :return: parquet files
    """
    
    # get filepath to song data file
    song_data = input_data
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select('song_id', 'title', 'artist_id', 'year', 'duration').where(df.song_id != '').dropDuplicates(['song_id'])
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy(['year', 'artist_id']).parquet(output_data + "/songs_table/songs_table.parquet") 
    print(f"INFO: songs_table with {songs_table.count()} rows processed and stored inside {output_data}.")
    
    # extract columns to create artists table
    artists_table = df.select('artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude').where(df.artist_id != '').dropDuplicates(['artist_id'])
    
    # write artists table to parquet files
    artists_table.write.parquet(output_data + "/artists_table/artists_table.parquet")
    print(f"INFO: artists_table with {artists_table.count()} rows processed and stored inside {output_data}.")

def process_log_data(spark, input_data, output_data):
    """
    extracts log data from source s3 bucket in json format process it using spark to 
    extract users, time and songplays data then load the result into parquet files in
    destination s3 bucket 
    
    :param spark: allows to initiate SparkSession
    :param input_data: input local data or s3 bucket address 
    :param output_data: output stores locally or s3 bucket address provided
    :return: parquet files
    """
    
    # get filepath to log data file
    log_data = input_data

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df[df.page == 'NextSong']

    # extract columns for users table    
    users_table = df.sort(df.ts.desc()).select('userId', 'firstName', 'lastName', 'gender', 'level').where(df.userId != '').dropDuplicates(['userId'])
    
    # write users table to parquet files
    users_table.write.parquet(output_data + "/users_table/users_table.parquet")
    print(f"INFO: users_table with {users_table.count()} rows processed and stored inside {output_data}.")
    
    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: int(x / 1000.0), IntegerType())
    df = df.withColumn("timestamp", get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    df = df.withColumn("start_time", from_unixtime(df.timestamp))
    
    # extract columns to create time table
    time_table = df.dropDuplicates(['start_time']).select('start_time').withColumn('hour', hour(df.start_time)) \
                .withColumn('day', date_format(df.start_time,'d')).withColumn('week', weekofyear(df.start_time)) \
                .withColumn('month', month(df.start_time)).withColumn('year', year(df.start_time)) \
                .withColumn('weekday', date_format(df.start_time,'u'))

    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy(['year', 'month']).parquet(output_data + "/time_table/time_table.parquet")
    print(f"INFO: time_table with {time_table.count()} rows processed and stored inside {output_data}.")
    
    # read in song data to use for songplays table
    songdf = spark.read.parquet(output_data + "/songs_table/songs_table.parquet")
    song_df = df.join(songdf, df.song==songdf.title, how='left')

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = song_df.select('start_time', 'userId', 'level', 'song_id', 'artist_id', 'sessionId', 'location', 'userAgent')
    songplays_table = songplays_table.withColumn("songplay_id", monotonically_increasing_id()).withColumn('month', month(df.start_time)).withColumn('year', year(df.start_time))

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy(['year','month']).parquet(output_data + "/songplays_table/songplays_table.parquet")
    print(f"INFO: songplays_table with {songplays_table.count()} rows processed and stored inside {output_data}.")

def main():
    spark = create_spark_session()
    
    # for testing dataset stored locally 
    #song_data = "song_data/*/*/*/*.json"
    #log_data = "log_data/*.json"
    #output_data = "sparkify_data" # local storage
    
    # for testing on subset of dataset stored in s3 bucket
    #song_data = "s3a://udacity-dend/song-data/A/A/*/*.json"
    #log_data = "s3a://udacity-dend/log-data/*/*/*.json"
    
    # for testing full dataset stored in s3 bucket
    song_data = "s3a://udacity-dend/song-data/A/A/*/*.json"
    log_data = "s3a://udacity-dend/log-data/*/*/*.json"
    
    # make sure to provide the active s3 bucket address
    output_data = "s3a://udacitydends"  # s3 bucket 
    
    
    process_song_data(spark, song_data, output_data)    
    process_log_data(spark, log_data, output_data)


if __name__ == "__main__":
    main()
