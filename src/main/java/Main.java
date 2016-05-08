/**
 * Created by ankitdevani on 5/6/16.
 */

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import scala.Tuple2;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;


public class Main implements Serializable{
    static MatrixFactorizationModel model;
    public static  JavaSparkContext sc;

    Main() {
        File file = new File("/Users/ankitdevani/Downloads/model");
        if(file.exists()) {
            model = MatrixFactorizationModel.load(sc.sc(), "/Users/ankitdevani/Downloads/model");
        }
    }
    public static void main(String[] args) {

        SparkConf conf = new SparkConf().setAppName("Netflix Analysis").setMaster("local[4]").set("spark.executor.memory", "8g");
        sc = new JavaSparkContext(conf);
        SQLContext sqlContext = new SQLContext(sc);

        // dataframe from the parquet file. You need to give the directory location under which all parquet files are located
        DataFrame movieRatings = sqlContext.read().parquet("/Users/ankitdevani/Downloads/parquet_raw_data");

        // can convert the dataframe to JavaRDD or RDD using .javaRDD() or .rdd() method
        JavaRDD<Row> rdd = movieRatings.javaRDD();
        int user_ID = movieRatings.first().getInt(1);
        System.out.println("User_ID" + user_ID);

        JavaRDD<Rating> ratingsRDD = rdd.map(
                new Function<Row, Rating>() {
                    @Override
                    public Rating call(Row row) throws Exception {
                        return new Rating(row.getInt(1), row.getInt(0), row.getInt(2));
                    }
                }
        );

        final Map<Integer, String> productList=rdd.mapToPair(
                new PairFunction<Row, Integer, String>() {
                    @Override
                    public Tuple2<Integer, String> call(Row row) throws Exception {
                        return new Tuple2<Integer, String>(row.getInt(0), row.getString(5));
                    }
                }
        ).distinct().collectAsMap();
        ArrayList<Integer> movieIdList = new ArrayList<Integer>(productList.keySet());

        Main m= new Main();
        if(model == null) {
            double[] splitWeight = {0.6,0.2, 0.2};
            JavaRDD<Rating>[] splitRDD = ratingsRDD.randomSplit(splitWeight, 0L);
            JavaRDD<Rating> trainingRDD = splitRDD[0].cache();
            JavaRDD<Rating> validationRDD = splitRDD[1].cache();
            JavaRDD<Rating> testingRDD = splitRDD[2].cache();
            m.train(trainingRDD);
        }

        List<Rating> recommendations = m.getRecommendations(user_ID, ratingsRDD, movieIdList);

        System.out.println("Recomendation" + recommendations.size());
        System.out.println("List" + recommendations);
        System.out.println("List" + productList.get(recommendations.get(0).product()));
    }

    public  List<Rating> getRecommendations(final int userId, JavaRDD<Rating>
            ratings, final List<Integer> movieList) {
        List<Rating> recommendations;

        //Getting the users ratings
        JavaRDD<Rating> userRatings = ratings.filter(
                new Function<Rating, Boolean>() {
                    @Override
                    public Boolean call(Rating rating) throws Exception {
                        return rating.user() == userId;
                    }
                });

        //Getting the product ID's of the products that user rated
        JavaRDD < Tuple2 < Integer, Integer >> userProducts = userRatings.map(
                new Function<Rating, Tuple2<Integer, Integer>>() {
                    public Tuple2<Integer, Integer> call(Rating r) {
                        return new Tuple2<Integer, Integer>(r.user(), r.product());
                    }
                }
        );

        List<Integer> toRemove =  userProducts.map(
                new Function<Tuple2<Integer,Integer>, Integer>() {
                    @Override
                    public Integer call(Tuple2<Integer, Integer> integerIntegerTuple2) throws Exception {
                        return integerIntegerTuple2._2;
                    }
                }
        ).collect();

        for(Integer movieId : toRemove) {
            movieList.remove(movieId);
        }

        System.out.println("Movie list size " + movieList.size());

        JavaRDD<Integer> candidates = sc.parallelize(movieList);

        JavaRDD<Tuple2<Integer, Integer>> userCandidates = candidates.map(
                new Function<Integer, Tuple2<Integer, Integer>>() {
                    public Tuple2<Integer, Integer> call(Integer integer) throws Exception {
                        return new Tuple2<Integer, Integer>(userId, integer);
                    }
                }
        );
        //Predict recommendations for the given user
        recommendations = model.predict(JavaPairRDD.fromJavaRDD(userCandidates)).collect();

        List<Rating> rec = new ArrayList<Rating>(recommendations);

        //Sorting the recommended products and sort them according to the rating
        Collections.sort(rec, new Comparator<Rating>() {
            public int compare(Rating r1, Rating r2) {
                return r1.rating() < r2.rating() ? -1 : r1.rating() > r2.rating() ? 1 : 0;
            }
        });

        System.out.println("Recommended LIST SIZE" + recommendations.size());

        //get top 50 from the recommended products.
        recommendations = recommendations.subList(0, 50);

        return recommendations;
    }

    public void train(JavaRDD<Rating> trainingRDD) {
        model = ALS.train(JavaRDD.toRDD(trainingRDD), 20, 10, 0.01);
        //RDD<Tuple2<Object, double[]>> features = model.productFeatures();
        System.out.println("Saving model");
        File file = new File("/Users/ankitdevani/Downloads/model");

        if (file.exists()) {
            try {
                delete(file);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        model.save(sc.sc(), "/Users/ankitdevani/Downloads/model");
    }

    public static void delete(File file)
            throws IOException {

        if(file.isDirectory()){

            //directory is empty, then delete it
            if(file.list().length==0){

                file.delete();
                System.out.println("Directory is deleted : "
                        + file.getAbsolutePath());

            }else{

                //list all the directory contents
                String files[] = file.list();

                for (String temp : files) {
                    //construct the file structure
                    File fileDelete = new File(file, temp);

                    //recursive delete
                    delete(fileDelete);
                }

                //check the directory again, if empty then delete it
                if(file.list().length==0){
                    file.delete();
                    System.out.println("Directory is deleted : "
                            + file.getAbsolutePath());
                }
            }

        }else{
            //if file, then delete it
            file.delete();
            System.out.println("File is deleted : " + file.getAbsolutePath());
        }
    }
}

