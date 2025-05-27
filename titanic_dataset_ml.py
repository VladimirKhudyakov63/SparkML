from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.sql.types import IntegerType


def model_params(rf):
    return ParamGridBuilder() \
        .addGrid(rf.maxDepth, [5, 6, 7, 8, 9]) \
        .addGrid(rf.maxBins, [4, 5, 6, 7]) \
        .build()


def prepare_data(df: DataFrame, assembler) -> DataFrame:
    df = df.withColumn("is_married", df.married.cast(IntegerType()))
    sex_index = StringIndexer(inputCol="sex", outputCol="sex_index")
    df = sex_index.fit(df).transform(df)
    df = assembler.transform(df)
    return df


def vector_assembler() -> VectorAssembler:
    features = ["age", "salary", "successfully_credit_completed", "credit_completed_amount", "active_credits",
                "active_credits_amount", "sex_index", "is_married"]
    assembler = VectorAssembler(inputCols=features, outputCol='features')
    return assembler

def build_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(labelCol="credit_amount", featuresCol="features")

def build_evaluator() -> RegressionEvaluator:
    return RegressionEvaluator(labelCol="credit_amount", predictionCol="prediction", metricName="rmse")

def build_tvs(rand_forest, evaluator, model_params) -> TrainValidationSplit:
    return TrainValidationSplit(
        estimator=rand_forest,
        estimatorParamMaps=model_params,
        evaluator=evaluator,
        trainRatio=0.8
    )


def train_model(train_df, test_df) -> (RandomForestRegressionModel, float):
    assembler = vector_assembler()
    train_pdf = prepare_data(train_df, assembler)
    test_pdf = prepare_data(test_df, assembler)
    rf = build_random_forest()
    evaluator = build_evaluator()
    tvs = build_tvs(rf, evaluator, model_params(rf))
    models = tvs.fit(train_pdf)
    best = models.bestModel
    predictions = best.transform(test_pdf)
    rmse = evaluator.evaluate(predictions)
    print(f"RMSE: {rmse}")
    print(f'Model maxDepth: {best._java_obj.getMaxDepth()}')
    print(f'Model maxBins: {best._java_obj.getMaxBins()}')
    return best, rmse


if __name__ == "__main__":
    spark = SparkSession.builder.appName('PySparkMLJob').getOrCreate()
    train_df = spark.read.parquet("train.parquet")
    test_df = spark.read.parquet("test.parquet")
    train_model(train_df, test_df)
