/**
 * Created by ankitdevani on 5/6/16.
 */

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import scala.Tuple2;

import java.util.Map;


public class Main{
    public static  JavaSparkContext sc;
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("Netflix Analysis").setMaster("local[4]");
        sc = new JavaSparkContext(conf);
        SQLContext sqlContext = new SQLContext(sc);

        // dataframe from the parquet file. You need to give the directory location under which all parquet files are located
        DataFrame movieRatings = sqlContext.read().parquet("/user/user01/data");

        // can convert the dataframe to JavaRDD or RDD using .javaRDD() or .rdd() method
        JavaRDD<Row> rdd = movieRatings.javaRDD();
        movieRatings.printSchema();

        JavaRDD<Rating> ratingsRDD = rdd.map(
                new Function<Row, Rating>() {
                    @Override
                    public Rating call(Row row) throws Exception {
                        return new Rating(row.getInt(1), row.getInt(0), row.getInt(2));
                    }
                }
        );

        /*JavaRDD<Tuple2<Integer, Rating>> ratings = rdd.map(
                new Function<Row, Tuple2<Integer, Rating>>() {
                    public Tuple2<Integer, Rating> call(Row row) throws Exception {

                        Rating rating = new  Rating(row.getInt(1), row.getInt(0), row.getInt(2));
                        return new Tuple2<Integer, Rating>(cacheStamp, rating);
                    }
                }
        );*/

        Map<Integer, String> products = rdd.mapToPair(
                new PairFunction<Row, Integer, String>() {
                    public Tuple2<Integer, String> call(Row row) throws Exception {

                        return new Tuple2<Integer, String>(row.getInt(0), row.getString(5));
                    }
                }
        ).collectAsMap();

        double[] splitWeight = {0.6,0.2, 0.2};

        JavaRDD<Rating>[] splitRDD = ratingsRDD.randomSplit(splitWeight, 0L);
        JavaRDD<Rating> trainingRDD = splitRDD[0].cache();
        JavaRDD<Rating> validationRDD = splitRDD[1].cache();
        JavaRDD<Rating> testingRDD = splitRDD[2].cache();

        long trainingCount = trainingRDD.count();
        long validationCount = validationRDD.count();
        long testingCount = testingRDD.count();

        System.out.println("Training Count: " + trainingCount + "Validation Count: " + validationCount
                                                                                    + "Testing Count: " + testingCount);

        MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(trainingRDD), 20, 10, 0.01);
        RDD<Tuple2<Object, double[]>> features = model.productFeatures();
        System.out.println("Saving model");
        model.save(sc.sc(), "/user/user01/model");
        features.saveAsTextFile("/user/user01/" + "features");
        System.out.println("Loading model");

        model = MatrixFactorizationModel.load(sc.sc(), "/user/user01/model");
    }

}

