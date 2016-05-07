/**
 * Created by ankitdevani on 5/6/16.
 */
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Row;

public class Main{
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("NetflixAnalysis").setMaster("local[4]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SQLContext sqlContext = new SQLContext(sc);

        // dataframe from the parquet file. You need to give the directory location under which all parquet files are located
        DataFrame movieRatings = sqlContext.read().parquet("/user/user01/data");

        // can convert the dataframe to JavaRDD or RDD using .javaRDD() or .rdd() method
        JavaRDD<Row> rdd = movieRatings.javaRDD();
        movieRatings.printSchema();
        
    }
}

