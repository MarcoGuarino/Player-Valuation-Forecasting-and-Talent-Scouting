import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pre_processing

def create_spark_session():
    # Create the session
    conf = SparkConf(). \
        set('spark.ui.port', "4050"). \
        set('spark.executor.memory', '10G'). \
        set('spark.driver.memory', '45G'). \
        set('spark.driver.maxResultSize', '30G'). \
        setAppName("PySparkTutorial"). \
        set('spark.executor.cores', "8"). \
        setMaster("local[*]")

    sc = pyspark.SparkContext.getOrCreate(conf=conf)
    #sc.stop()
    spark = SparkSession.builder.getOrCreate()


def run_preprocessing():
    final_table = pre_processing.create_final_table()

if __name__ == "__main__":
    create_spark_session()
    run_preprocessing()