export SPARK_HOME=/Users/ankitdevani/Downloads/spark-1.6.1
echo $SPARK_HOME
$SPARK_HOME/bin/spark-submit \
  --class Main \
  --master local[4] \
 target/driver-1.0-SNAPSHOT.jar
