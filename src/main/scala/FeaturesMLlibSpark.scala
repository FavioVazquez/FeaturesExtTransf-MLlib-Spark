import org.apache.spark.mllib.feature.{IDF, HashingTF}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import globals.Globals

/**
 * Created by Favio on 18/05/15.
 */

object FeaturesMLlibSpark {

  def main(args: Array[String]) {

    val conf = new SparkConf()
            .setMaster("local")
//      .setMaster(Globals.masterSpark)
      .setAppName("Basic Statistics MLlib")
      .set("spark.executor.memory", "6g")
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
    val documents: RDD[Seq[String]] = sc.textFile(Globals.masterHDFS +
      "/try/DonQuijote.txt").map(_.split(" ").toSeq)

    val hashingTF = new HashingTF()
    val tf: RDD[Vector] = hashingTF.transform(documents)

//    tf.take(20).foreach(println)

    /**
     * While applying HashingTF only needs a single pass to the data,
     * applying IDF needs two passes: first to compute the IDF vector
     * and second to scale the term frequencies by IDF.
     */

    tf.cache()
    val idf = new IDF().fit(tf)
    val tfidf: RDD[Vector] = idf.transform(tf)
//    tfidf.take(20).foreach(println)

    tfidf.coalesce(1).saveAsTextFile(Globals.masterHDFS+"/try/hash1")

    /**
     * MLlib’s IDF implementation provides an option for ignoring terms
     * which occur in less than a minimum number of documents. In such
     * cases, the IDF for these terms is set to 0. This feature can be
     * used by passing the minDocFreq value to the IDF constructor.
     *
     */

    val idf1 = new IDF(minDocFreq = 2).fit(tf)
    val tfidf1: RDD[Vector] = idf1.transform(tf)

    tfidf.coalesce(1).saveAsTextFile(Globals.masterHDFS+"/try/hash2")

    sc.stop()
  }
}
