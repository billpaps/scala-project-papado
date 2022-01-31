import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.LDA

import scala.io.Source
import org.apache.spark.ml.feature.{CountVectorizer, HashingTF, IDF, IDFModel, StopWordsRemover, Tokenizer, VectorAssembler}

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition, Vectors}
import org.apache.spark.sql.functions.{array, asc, col, collect_list, collect_set, concat_ws, count, explode, flatten, lag, last, lower, regexp_replace, size, sort_array, split, to_date, udf, when}
import org.apache.spark.sql.{Column, DataFrame, Row, SparkSession, functions}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.mllib.linalg.{Vectors, Vector => MLLibVector}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector => MLVector}
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.dsl.expressions.StringToAttributeConversionHelper
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types.IntegerType
import shapeless.syntax.std.tuple.unitTupleOps

import scala.collection.mutable

object Task1_3 {

  def get_Names(): List[String] = {
    var num = List("00","01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18")
    return num
  }

  def countNullInCols(columns:Array[String]):Array[Column]={
    columns.map(c=>{
      count(when(col(c).isNull ||
        //col(c)==="" ||
        col(c).contains("NULL") ||
        col(c).contains("null"),c)
      ).alias(c)
    })
  }

  def topTerms(svd: SingularValueDecomposition[RowMatrix, Matrix], numConcepts: Int, numTerms: Int, termId: Array[String])
  : Seq[Seq[(String, Double)]] = {
    val v = svd.V
    val topTerms = new ArrayBuffer[Seq[(String, Double)]]()
    val arr = v.toArray
    for (i <- 0 until numConcepts) {
      val offs = i * v.numRows
      val termWeights = arr.slice(offs, offs + v.numRows).zipWithIndex
      val sorted = termWeights.sortBy(-_._1)

      topTerms += sorted.take(numTerms).map {
        case (score, id) => (termId(id), score)
      }
    }
    topTerms
  }

  def readFile(filename: String): Seq[String] = {
    val bufferedSource = scala.io.Source.fromFile(filename)
    val lines = (for (line <- bufferedSource.getLines()) yield line).toList
    bufferedSource.close
    lines
  }


  def main(args: Array[String]): Unit = {

    val ss = SparkSession.builder().master("local[*]").appName("App").getOrCreate()

    ss.sparkContext.setLogLevel("ERROR")

    import spark.implicits._

    val names = get_Names()
    names.foreach(names => {
      println("Starting file " + names)
      var df = ss.read.parquet("processed.parquet/part-000" + names + "-98b58bde-4095-40d6-abcc-8edc5a0cc0c7-c000.snappy.parquet")

      df = df.withColumn("cleaned_speech",
        concat_ws(",",col("cleaned_speech")))
      df = df.withColumn("cleaned_speech", lower(col("cleaned_speech")))
      df = df.withColumn("cleaned_speech", split(col("cleaned_speech"),",").as("cleaned_speech"))
      df = df.filter(size(df("cleaned_speech")) > 0)
      val stopwords = readFile("stop.txt")

      val remover = new StopWordsRemover()
        .setStopWords(stopwords.toArray)
        .setInputCol("cleaned_speech")
        .setOutputCol("removed")

      df = remover.transform(df)

      // drop rows with empty arrays in column removed
      df = df.filter(size(col("removed")) > 1)
      df.show()
      var tokens_df = df.filter(size(col("removed")) > 1)
      tokens_df.show(truncate = false)

      // ------------------------------------------------------ task1 ------------------------------------------------

      val vectorizer = new CountVectorizer()
        .setInputCol("removed")
        .setOutputCol("raw")
        .setMaxDF(0.2)
        .setVocabSize(8000)

      val vec = vectorizer.fit(tokens_df)
      tokens_df = vec.transform(tokens_df).select("raw")
      tokens_df.show(truncate = false)

      val idf = new IDF().setInputCol("raw").setOutputCol("features")
      val model_idf = idf.fit(tokens_df)
      tokens_df = model_idf.transform(tokens_df)
      tokens_df.show(truncate = false)


      val rdd = tokens_df.select("features").rdd.map {
        row => Vectors.fromML(row.getAs[MLVector]("features"))
      }

      rdd.cache()
      val mat = new RowMatrix(rdd)
      val svd = mat.computeSVD(10, computeU = true)
      print(topTerms(svd, 10, 8, vec.vocabulary).mkString("\n\n"))
      var df_topics = topTerms(svd, 10, 8, vec.vocabulary).toDF("topics")


      //df_topics=df_topics.withColumn("value",removeNumber(col("value"),col("values")))
      //df_topics = df_topics.withColumn("ArrayOfString", stringify(col("ArrayOfString")))
      // df_topics = df_topics.withColumn("topics", concat_ws(",", col("topics")))
      //df_topics = df_topics.withColumn("flatten", flatten(col("topics")))

      df_topics.show(truncate = false)
      // save
      df_topics.write.mode("append").json("topics"+names)

      //------------------------------------------------------Task 3----------------------------------------------------------

      val window = Window.partitionBy(col("member_name"), col("political_party"))

      df = df.withColumn("test", collect_list(("cleaned_speech")).over(window)).groupBy("member_name", "political_party").agg(last("test"))
      df.printSchema()
      df = df.withColumn("flatten", flatten(col("last(test)")))

      df = df.drop(col("last(test)"))
      //df.show()


      val vectorizer2 = new CountVectorizer()
        .setInputCol("flatten")
        .setOutputCol("raw")
        .setMaxDF(0.3)
        .setVocabSize(100)


      val vec2 = vectorizer2.fit(df)
      df = vec2.transform(df)
      val vocabulary = vec2.vocabulary

      val idf2 = new IDF().setInputCol("raw").setOutputCol("features")
      val model_df2 = idf2.fit(df)
      df = model_df2.transform(df)
      //df.show()

      //df = df.select(col("member_name"), explode(col("flatten")).name("expwords")).withColumn("flatten", array("expwords"))
      //val hashdf = udf((vector: String) => vector.indices(0))
      //df = hashingTF.transform(df).withColumn("wordhash", hashdf(col("raw")))


      def keywords(treshold: Double) = udf((features: SparseVector) => {
        (features.indices zip features.values) filter(_._2 < treshold) map {
          case (index, _) => vocabulary(index)
        }
      })

      df = df.select('member_name, 'political_party,
        keywords(4)('features).as("mostUsedWords")
      )
      df = df.filter(size(col("mostUsedWords")) > 1)
      df.show(truncate = false)
      // save
      df = df.withColumn("mostUsedWords",
        concat_ws(",",col("mostUsedWords")))

      //df.write.option("header",true).csv("keywords"+names)

      //df.write.mode("append").json("keywords"+names)


    })

}
}