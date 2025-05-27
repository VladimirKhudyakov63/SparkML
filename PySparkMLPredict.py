import argparse

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

MODEL_PATH = 'spark_ml_model'


def main(data_path, model_path, result_path):
    """
    :param data_path: path to dataset
    :param model_path: path to model
    :param result_path: path to prediction result
    """
    spark = _spark_session()
    test_df = spark.read.parquet(data_path)
    model = PipelineModel.load(model_path)
    prediction = model.transform(test_df)
    result = prediction.select("session_id","prediction")
    result.write.mode("overwrite").parquet(result_path)

def _spark_session():
    # Create spark session
    return SparkSession.builder.appName('PySparkMLPredict').getOrCreate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Please set model path.')
    parser.add_argument('--data_path', type=str, default='test.parquet', help='Please set datasets path.')
    parser.add_argument('--result_path', type=str, default='result', help='Please set result path.')
    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    result_path = args.result_path
    main(data_path, model_path, result_path)
