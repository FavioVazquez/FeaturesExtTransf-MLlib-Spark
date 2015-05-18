import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by Favio on 18/05/15.
 */

object FeaturesMLlibSpark {

 val conf = new SparkConf()
//      .setMaster("local")
    .setMaster("mesos://master.mcbo.mood.com.ve:5050")
    .setAppName("Basic Statistics MLlib")
    .set("spark.executor.memory", "12g")
  val sc = new SparkContext(conf)


//  MLlib - Feature Extraction and Transformation

  /**
   * TF-IDF
   *
   * Term frequency-inverse document frequency (TF-IDF) is a feature
   * vectorization method widely used in text mining to reflect the
   * importance of a term to a document in the corpus. Denote a term by
   * t, a document by d, and the corpus by D. Term frequency TF(t,d) is
   * the number of times that term t appears in document d, while document
   * frequency DF(t,D) is the number of documents that contains term t.
   * If we only use term frequency to measure the importance, it is very
   * easy to over-emphasize terms that appear very often but carry little
   * information about the document, e.g., “a”, “the”, and “of”. If a term
   * appears very often across the corpus, it means it doesn’t carry
   * special information about a particular document. Inverse document
   * frequency is a numerical measure of how much information a term
   * provides:
   *
   *                  IDF(t,D) = log((Math.abs(D)+1)/DF(t,D)+1)
   *
   * Where Math.abs(D)  is the total number of documents in the corpus.
   * Since logarithm is used, if a term appears in all documents, its
   * IDF value becomes 0. Note that a smoothing term is applied to avoid
   * dividing by zero for terms outside the corpus. The TF-IDF measure
   * is simply the product of TF and IDF:
   *
   *                   TFIDF(t,d,D)=TF(t,d)⋅IDF(t,D)
   *
   * There are several variants on the definition of term frequency and
   * document frequency. In MLlib, we separate TF and IDF to make them
   * flexible.
   *
   * Our implementation of term frequency utilizes the hashing trick.
   * A raw feature is mapped into an index (term) by applying a hash
   * function. Then term frequencies are calculated based on the mapped
   * indices. This approach avoids the need to compute a global term-to-index map,
   * which can be expensive for a large corpus, but it suffers from
   * potential hash collisions, where different raw features may become
   * the same term after hashing. To reduce the chance of collision, we
   * can increase the target feature dimension, i.e., the number of buckets
   * of the hash table. The default feature dimension is 2²⁰=1,048,576.
   *
   * TF and IDF are implemented in HashingTF and IDF. HashingTF takes an
   * RDD[Iterable[_]] as the input. Each record could be an iterable of
   * strings or other types.
   *
   */

   // Load documents (one per line).
  val documents: RDD[Seq[String]] = sc.textFile("hdfs://")



  sc.stop()
}
