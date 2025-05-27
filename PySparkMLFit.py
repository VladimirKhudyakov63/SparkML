import argparse

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier

MODEL_PATH = 'spark_ml_model'
LABEL_COL = 'is_bot'


def process(spark, data_path, model_path):
    """
    :param data_path: path to dataset
    :param model_path: path to model
    :param result_path: path to prediction result
    """
    # Read the data
    df = spark.read.parquet(data_path)
    train, test = df.randomSplit([0.8, 0.2])

    # Cache datasets
    train.cache()
    test.cache()

    # Pre
    user_type_indexer = StringIndexer(inputCol="user_type", outputCol="user_type_index")
    platform_indexer = StringIndexer(inputCol="platform", outputCol="platform_index")
    feature = VectorAssembler(inputCols=["duration", "item_info_events", "select_item_events",
                                         "make_order_events", "events_per_min", "user_type_index", "platform_index"],
                              outputCol="features")

    # Evaluator
    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="is_bot", predictionCol="prediction",
                                                           metricName="accuracy")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="is_bot", predictionCol="prediction", metricName="f1")

    # Models
    rf_classifier = RandomForestClassifier(labelCol="is_bot", featuresCol="features")
    dt_classifier = DecisionTreeClassifier(labelCol="is_bot", featuresCol="features")
    gbtc_classifier = GBTClassifier(labelCol="is_bot", featuresCol="features")

    # Pipeline creation
    pipeline_rf = Pipeline(stages=[user_type_indexer, platform_indexer, feature, rf_classifier])
    pipeline_dt = Pipeline(stages=[user_type_indexer, platform_indexer, feature, dt_classifier])
    pipeline_gbtc = Pipeline(stages=[user_type_indexer, platform_indexer, feature, gbtc_classifier])

    # Parameters grid
    rf_paramGrid = (ParamGridBuilder()
                    .addGrid(rf_classifier.numTrees, [10, 20])
                    .addGrid(rf_classifier.maxDepth, [5, 10])
                    .addGrid(rf_classifier.maxBins, [32, 64])
                    .build())

    dt_paramGrid = (ParamGridBuilder()
                    .addGrid(dt_classifier.maxDepth, [5, 10])
                    .addGrid(dt_classifier.maxBins, [32, 64])
                    .build())

    gbtc_paramGrid = (ParamGridBuilder()
                      .addGrid(gbtc_classifier.maxIter, [10, 20])
                      .addGrid(gbtc_classifier.maxDepth, [5, 10])
                      .build())

    # Cross-validation
    crossval_rf = CrossValidator(estimator=pipeline_rf,
                                 estimatorParamMaps=rf_paramGrid,
                                 evaluator=accuracy_evaluator,
                                 numFolds=3)

    crossval_dt = CrossValidator(estimator=pipeline_dt,
                                 estimatorParamMaps=dt_paramGrid,
                                 evaluator=accuracy_evaluator,
                                 numFolds=3)

    crossval_gbtc = CrossValidator(estimator=pipeline_gbtc,
                                   estimatorParamMaps=gbtc_paramGrid,
                                   evaluator=accuracy_evaluator,
                                   numFolds=3)

    # Training model
    model_rf = crossval_rf.fit(train)
    model_dt = crossval_dt.fit(train)
    model_gbtc = crossval_gbtc.fit(train)

    # Testing model
    models = [
        ("RandomForestClassifier", model_rf),
        ("DecisionTreeClassifier", model_dt),
        ("GBTClassifier", model_gbtc)
    ]

    best_model = None
    best_accuracy = 0.0
    best_model_name = ""

    for model_name, model in models:
        predictions = model.transform(test)
        accuracy = accuracy_evaluator.evaluate(predictions)
        f1 = f1_evaluator.evaluate(predictions)
        print(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        # Best model choose
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model.bestModel
            best_model_name = model_name

    # Save best model
    if best_model:
        print(f"Saving best model ({best_model_name}) with accuracy {best_accuracy:.4f} to {model_path}")
        best_model.write().overwrite().save(model_path)
    else:
        print("No model was selected as the best.")

    # Clear cache
    train.unpersist()
    test.unpersist()


def main(data_path, model_path):
    spark = _spark_session()
    process(spark, data_path, model_path)
    spark.stop()


def _spark_session():
    # create spark session
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='session-stat.parquet', help='Please set datasets path.')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Please set model path.')
    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    # main
    main(data_path, model_path)