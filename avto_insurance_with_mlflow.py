import argparse
import os
import mlflow
from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer,StringIndexerModel,VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline,PipelineModel
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

LABEL_COL = "has_car_accident"

os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://storage.yandexcloud.net'
os.environ['AWS_ACCESS_KEY_ID'] = '33kU43UzyCYfV1jgKUPL'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'WPZnfkNEOlpdZ32hwVGhQ6PNiPPjmFZEajnWUMRe'

mlflow.set_tracking_uri("https://mlflow.lab.karpov.courses")
mlflow.set_experiment(experiment_name="vladimir-hudjakov-fpa5335")

def create_pipeline(train_alg) -> Pipeline:
    sex_index = StringIndexer(inputCol="sex", outputCol="sex_index")
    car_class_index = StringIndexer(inputCol="car_class", outputCol="car_class_index")
    features = VectorAssembler(inputCols=["age", "driving_experience", "speeding_penalties",
                                     "parking_penalties", "total_car_accident", "sex_index", "car_class_index"],
                          outputCol= "features")
    return Pipeline(stages=[sex_index, car_class_index, features, train_alg])

def optimization(pipeline, gbt, train_df, evaluator):
    grid = ParamGridBuilder() \
            .addGrid(gbt.maxDepth, [3,5]) \
            .addGrid(gbt.maxBins, [16, 32]) \
            .addGrid(gbt.maxIter, [20, 30]).build()
    tvs = TrainValidationSplit(estimator=pipeline,
                               estimatorParamMaps=grid,
                               evaluator=evaluator,
                               trainRatio=0.8)
    models = tvs.fit(train_df)
    return  models.bestModel

def evaluation(evaluator, predict, metric_list):
    metrics = {}
    for metric in metric_list:
        evaluator.setMetricName(metric)
        score = evaluator.evaluate(predict)
        metrics[metric] = score
        print(f'{metric} score = {score}')
    return metrics

def process(spark, train_path, test_path):
    mlflow.start_run()
    train = spark.read.parquet(train_path)
    test = spark.read.parquet(test_path)
    start_df = spark.read.parquet(test_path)
    train.cache()
    test.cache()
    gbt = GBTClassifier(labelCol=LABEL_COL)
    pipeline = create_pipeline(gbt)
    evaluator = MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="f1")
    model = optimization(pipeline, gbt, train, evaluator)
    predict = model.transform(test)
    metrics = evaluation(evaluator, predict, ["f1", "accuracy", "weightedRecall", "weightedPrecision"])
    print("Best model saved")

    # Logging params and metrics to the mlflow
    for metric, score in metrics.items():
        mlflow.log_metric(metric, score)

    mlflow.log_param("maxDepth", model.stages[-1].getMaxDepth())
    mlflow.log_param("maxBins", model.stages[-1].getMaxBins())
    mlflow.log_param("maxIter", model.stages[-1].getMaxIter())

    for i in range(0, len(model.stages)):
        stage = model.stages[i]
        mlflow.log_param(f'stage_{i}', stage.__class__.__name__)
        if type(stage) == StringIndexerModel:
            mlflow.log_param(f'stage_{i}_input', stage.getInputCol())
            mlflow.log_param(f'stage_{i}_output', stage.getOutputCol())
        elif type(stage) == VectorAssembler:
            mlflow.log_param(f'stage_{i}_input', stage.getInputCols())
            mlflow.log_param(f'stage_{i}_output', stage.getOutputCol())
        else:
            mlflow.log_param(f'feature', stage.getFeaturesCol())
            mlflow.log_param(f'target', stage.getLabelCol())

    mlflow.log_param('target', LABEL_COL)
    mlflow.log_param('input_columns', list(start_df.columns))
    mlflow.log_param('features', ["age", "sex_index", "car_class_index", "driving_experience",
                "speeding_penalties", "parking_penalties", "total_car_accident"])

    # Start logging to mlflow
    mv = mlflow.spark.log_model(model,
                                artifact_path="vladimir-hudjakov-fpa5335",
                                registered_model_name="vladimir-hudjakov-fpa5335")
    mv
    mlflow.end_run()

def _spark_session() -> SparkSession:
    return SparkSession.builder.appName('avto_insurance').getOrCreate()

def main(train_path, test_path) -> None:
    spark = _spark_session()
    process(spark, train_path, test_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='taxi_train.parquet', help='Set train dataset')
    parser.add_argument('--test', type=str, default='taxi_test.parquet', help='Set test dataset')
    args = parser.parse_args()
    train = args.train
    test = args.test
    main(train, test)
