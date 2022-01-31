import com.johnsnowlabs.nlp.base.EmbeddingsFinisher
import org.apache.spark.ml.classification.{LinearSVC, NaiveBayes, OneVsRest}
import org.apache.spark.ml.clustering.BisectingKMeans
import org.apache.spark.ml.evaluation.{ClusteringEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{CountVectorizer, IDF, MinMaxScaler, StopWordsRemover, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{Column, Row, SparkSession}
import org.apache.spark.sql.catalyst.dsl.expressions.{StringToAttributeConversionHelper, intToLiteral}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

import scala.Console.in
import scala.collection.mutable


object Task5_6 {

  def get_Names(): List[String] = {
    var num = List("01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18")
    return num
  }

  def readFile(filename: String): Seq[String] = {
    val bufferedSource = scala.io.Source.fromFile(filename)
    val lines = (for (line <- bufferedSource.getLines()) yield line).toList
    bufferedSource.close
    lines
  }

  def main(args: Array[String]): Unit = {
    // Basic Inits
    val ss = SparkSession.builder().master("local[3]").appName("Task6").getOrCreate()
    ss.sparkContext.setLogLevel("ERROR")
    val names = get_Names()
    var df_for_6 = ss.read.parquet("processed.parquet/part-00000-98b58bde-4095-40d6-abcc-8edc5a0cc0c7-c000.snappy.parquet")

    names.foreach(names => {
      println("Starting file " + names)
      val df1 = ss.read.parquet("processed.parquet/part-000" + names + "-98b58bde-4095-40d6-abcc-8edc5a0cc0c7-c000.snappy.parquet").sample(0.1)
      df_for_6 = df_for_6.union(df1)
    })

    val stopwords = readFile("stop.txt")

    //df_for_6 = df_for_6.drop(cols: _*).na.drop("any")

    val partiesOfInterest = Seq("κομμουνιστικο κομμα ελλαδας","νεα δημοκρατια",
      "συνασπισμος ριζοσπαστικης αριστερας","πανελληνιο σοσιαλιστικο κινημα")


    df_for_6 = df_for_6.withColumn("cleaned_speech",
      concat_ws(",",col("cleaned_speech")))
    df_for_6 = df_for_6.withColumn("cleaned_speech", lower(col("cleaned_speech")))
    df_for_6 = df_for_6.drop( "parliamentary_period", "parliamentary_sitting", "member_region", "member_gender")

    // with "withColumn" I can kee all the column I want, while with "select" all the columns are droped, except the one that I perform the transformation on.
    df_for_6 = df_for_6.withColumn("cleaned_speech", split(col("cleaned_speech"),",").as("cleaned_speech"))

    df_for_6 = df_for_6.withColumn("cleaned_speech", expr("filter(cleaned_speech, x -> not(length(x) < 3))")).where(size(col("cleaned_speech")) > 0)

    var subOfParties = df_for_6.filter(col("political_party").isin(partiesOfInterest:_*))

    val remover = new StopWordsRemover()
      .setStopWords(stopwords.toArray)
      .setInputCol("cleaned_speech")
      .setOutputCol("removed")
    subOfParties = remover.transform(subOfParties)
    //subOfParties.show()

    subOfParties = subOfParties.filter(size(col("removed")) > 1)
    //subOfParties.show()

    val vectorizer = new CountVectorizer()
      .setInputCol("removed")
      .setOutputCol("raw")
      .setMaxDF(0.5)
      .setVocabSize(5000)
    val vec = vectorizer.fit(subOfParties)

    subOfParties = vec.transform(subOfParties)


    // Create tf-idf features
    /*
    val hashingTF = new HashingTF().setInputCol("tokens").setOutputCol("RawFeatures").setNumFeatures(5000)
    val featurizedDF = hashingTF.transform(wordsDF)
*/
    val idf = new IDF().setInputCol("raw").setOutputCol("Features")
    val idfM = idf.fit(subOfParties)

    subOfParties = idfM.transform(subOfParties)

    //val completeDF = embeddingsFinisher.transform(semicompleteDF)
    subOfParties = subOfParties.select("political_party","Features")

    val indexer = new StringIndexer()
      .setInputCol("political_party")
      .setOutputCol("pp_Index")

    val indexed = indexer.fit(subOfParties).transform(subOfParties).select("pp_Index","Features")

    val Array(trainingData, testData) = indexed.randomSplit(Array(0.75, 0.25))

    // -------------------------NAIVE BAYES-----------------------------

    val model = new NaiveBayes().setLabelCol("pp_Index").setFeaturesCol("Features").fit(trainingData)
    val preds_nb = model.transform(testData)
    preds_nb.show()

    val evaluatorf1_nb = new MulticlassClassificationEvaluator()
      .setLabelCol("pp_Index")
      .setPredictionCol("prediction")
      .setMetricName("f1")

    val evaluatorAcc_nb = new MulticlassClassificationEvaluator()
      .setLabelCol("pp_Index")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val f1_nb = evaluatorf1_nb.evaluate(preds_nb)

    val acc_nb = evaluatorAcc_nb.evaluate(preds_nb)
    println("f1 nb: "+f1_nb)
    println("acc nb: "+ acc_nb)

    //---------------------------SVM-------------------------

    val lsvc = new LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)

    val ovr = new OneVsRest().setClassifier(lsvc).setLabelCol("pp_Index").setFeaturesCol("Features").fit(trainingData)

    val predictions_svm = ovr.transform(testData)

    val evaluatorf1_svm = new MulticlassClassificationEvaluator()
      .setLabelCol("pp_Index")
      .setPredictionCol("prediction")
      .setMetricName("f1")

    val f1_svm = evaluatorf1_svm.evaluate(preds_nb)

    val evaluator_svm = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy").setLabelCol("pp_Index")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    // compute the classification error on test data.
    val acc_svm = evaluator_svm.evaluate(predictions_svm)
    println("acc svm: "+ acc_svm)
    println("f1 svm: "+ f1_svm)

    // ------------------------------------------------------ task5 ------------------------------------------------

    val remover_2 = new StopWordsRemover()
      .setStopWords(stopwords.toArray)
      .setInputCol("cleaned_speech")
      .setOutputCol("removed")
    df_for_6 = remover_2.transform(df_for_6)
    //df_for_6.show()

    df_for_6 = df_for_6.filter(size(col("removed")) > 1)
    //df_for_6.show()

    val vectorizer_2 = new CountVectorizer()
      .setInputCol("removed")
      .setOutputCol("raw")
      .setMaxDF(0.5)
      .setVocabSize(2000)
    val vec2 = vectorizer_2.fit(df_for_6)

    df_for_6 = vec2.transform(df_for_6)

    val idf2 = new IDF().setInputCol("raw").setOutputCol("features")
    val idfM2 = idf2.fit(df_for_6)

    df_for_6 = idfM2.transform(df_for_6)

/*
    val assembler = new VectorAssembler().setInputCols(Array("featuresidf")).setOutputCol("vVec")
    df_for_6 = assembler.transform(df_for_6)

    val scaler = new MinMaxScaler().setInputCol("vVec").setOutputCol("features").setMax(1).setMin(-1)
    df_for_6 = scaler.fit(df_for_6).transform(df_for_6)
*/
    val bkm = new BisectingKMeans().setSeed(1).setFeaturesCol("features").setPredictionCol("prediction")
    val model_bkm = bkm.fit(df_for_6)

    df_for_6 = model_bkm.transform(df_for_6)
    //df_for_6.show()

    val evaluator = new ClusteringEvaluator()

    val silhouette = evaluator.evaluate(df_for_6)
    println(s"Silhouette with squared euclidean distance = $silhouette")

    val centers = model_bkm.clusterCenters
    //centers.foreach(println)
    df_for_6 = df_for_6.drop("speech", "features", "vVec")
    df_for_6.select(df_for_6("prediction")).distinct().show()
    //val count = df_for_6.groupBy("prediction").count
    //count.show()
    //val counts = Seq(count)

    df_for_6 = df_for_6.drop("roles", "cleaned_speech", "removed","raw","features")
    //df_for_6.where(df_for_6("prediction") === 0).show(10000, truncate = false)
    //df_for_6.where(df_for_6("prediction") === 0).select(df_for_6("member_name")).distinct().show(1000)
    //df_for_6.where(df_for_6("prediction") === 0).select(df_for_6("political_party")).distinct().show(1000)
    //df_for_6.where(df_for_6("prediction") === 1).select(df_for_6("political_party")).distinct().show(1000)
    //df_for_6.groupBy("prediction").agg(count("political_party")).show(1000)
    //df_for_6.where(df_for_6("prediction") === 1).groupBy("prediction").agg(count("political_party")).show(1000)
    //df_for_6 = df_for_6.withColumn("count", count("political_party").over(
    //  Window.partitionBy("prediction")
    //)).orderBy("political_party")

    //---------------------------members--------------------------------

    //-val df_cl = df_for_6.where(distinct_names("member_name").like("%%")).filter(col("prediction")===(partiesOfInterest:_*))
    val df_cl0 = df_for_6.where(df_for_6("prediction") === 0)
    //df_cl0.show()
    val dfcl0 = df_cl0.groupBy("member_name").agg(count("member_name"))
    dfcl0.show(truncate = false)
    val df_cl1 = df_for_6.where(df_for_6("prediction") === 1)
    val dfcl1 = df_cl1.groupBy("member_name").agg(count("member_name"))
    dfcl1.show(truncate = false)
    val df_cl2 = df_for_6.where(df_for_6("prediction") === 2)
    val dfcl2 = df_cl2.groupBy("member_name").agg(count("member_name"))
    val df_cl3 = df_for_6.where(df_for_6("prediction") === 3)
    val dfcl3 = df_cl3.groupBy("member_name").agg(count("member_name"))
    dfcl3.show(truncate = false)

    //--------------------parties----------------------------

    var df_cl0_ = df_for_6.where(df_for_6("prediction") === 0)
    //df_cl0_.show()
    val dfcl0_ = df_cl0_.groupBy("political_party").agg(count("member_name"))
    dfcl0_.show(truncate = false)
    val df_cl1_ = df_for_6.where(df_for_6("prediction") === 1)
    val dfcl1_ = df_cl1_.groupBy("political_party").agg(count("member_name"))
    dfcl1_.show(truncate = false)
    val df_cl2_ = df_for_6.where(df_for_6("prediction") === 2)
    val dfcl2_ = df_cl2_.groupBy("political_party").agg(count("member_name"))
    val df_cl3_ = df_for_6.where(df_for_6("prediction") === 3)
    val dfcl3_ = df_cl3_.groupBy("political_party").agg(count("member_name"))
    dfcl3_.show(truncate = false)
    /*
    dfcl0.write.mode("append").csv("members_0")
    dfcl1.write.mode("append").csv("members_1")
    dfcl2.write.mode("append").csv("members_2")
    dfcl3.write.mode("append").csv("members_3")

    dfcl0_.write.mode("append").csv("party_0")
    dfcl1_.write.mode("append").csv("party_1")
    dfcl2_.write.mode("append").csv("party_2")
    dfcl3_.write.mode("append").csv("party_3")
*/

  }
}