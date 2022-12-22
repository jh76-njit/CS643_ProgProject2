import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;

import scala.Tuple2;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

/**
 * Example for Logistic Regression Classifier
 */
public class LogisticRegressionClassifierExample {
	public static void main(String[] args) {
		// configure spark
		SparkConf conf = new SparkConf().setAppName("LogisticRegressionClassifier")
				.setMaster("local[2]").set("spark.executor.memory","2g");
		
		// start a spark context
		SparkContext jsc = new SparkContext(conf);
		
		// provide path to data transformed as [feature vectors]
		String path = "data/mllib/sample_libsvm_data.txt";
		JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc, path).toJavaRDD();
		
		// Split initial RDD into two... [80% training data, 20% testing data].
		JavaRDD[] splits = data.randomSplit(new double[] {0.8, 0.2}, 11L);
		JavaRDD training = splits[0].cache();
		JavaRDD test = splits[1];
		
		// Run training algorithm to build the model.
		LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
				.setNumClasses(10)
				.run(training.rdd());
		
		// Compute raw scores on the test set.
		JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(p ->
		new Tuple2<>(model.predict(((LabeledPoint) p).features()), ((LabeledPoint) p).label()));
		
		// get evaluation metrics
		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
		double accuracy = metrics.accuracy();
		System.out.println("Accuracy = " + accuracy);
		
		// After training, save model to local for prediction in future  
		model.save(jsc, "LogisticRegressionClassifier");
		
		// stop the spark context
		jsc.stop();
		}
	}
