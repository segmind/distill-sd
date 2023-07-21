from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import rand
from img2dataset import download
import shutil
import os


def repartition():
    spark = SparkSession.builder.config("spark.driver.memory", "16G") .master("local[16]").appName('spark-repart').getOrCreate() 
    df = spark.read.parquet("dataset/data.parquet")
    # df = df.filter((df.WIDTH >= 1024) & (df.HEIGHT >= 1024))
    # df = df.filter((df.AESTHETIC_SCORE > 7))
    df = df.orderBy(rand(seed = 0)) # this line is important to have a shuffled dataset
    print(df.count())
    df.repartition(10).write.parquet("dataset/laion_small")


def download_images(output_dir="dataset/laion_small_images", url = "dataset/laion_small/part-00002-195faf27-0776-429e-a03b-a6aba71d4f16-c000.snappy.parquet"):
    output_dir = os.path.abspath(output_dir)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    spark = (
        SparkSession.builder.config("spark.driver.memory", "16G").master("local[16]").appName("spark-stats").getOrCreate()
    )

    download(
        processes_count=1,
        thread_count=32,
        url_list= url,
        image_size=512,
        output_folder=output_dir,
        output_format="webdataset",
        input_format="parquet",
        url_col="url",
        caption_col="generated_caption",
        enable_wandb=True,
        number_sample_per_shard=1000,
        distributor="pyspark",
    )

if __name__ == "__main__":
    repartition()
    download_images()


"""
Common Error Handling,

1. If an error with connecting to JAVA port,  Paste this in terminal `export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64`
2. Untar Files `for f in dataset/laion_small_images/*.tar; do tar -xvf "$f" -C data/; done`
3. Copy Files `for f in full_images/*.txt; do cp -v "$f" new_images/"${f//\//_}"; done` this is for text do .jpg for images

"""
