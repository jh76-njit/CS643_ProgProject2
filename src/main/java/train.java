import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.types.*;
import static org.apache.spark.sql.types.DataTypes.*;

public class train {
	private static JavaRDD<LabeledPoint> importRDDdata(JavaSparkContext jsc, String dataFile) {
		JavaRDD<String> data = jsc.textFile(dataFile);
		JavaRDD<String> data2 = data.filter(new Function<String, Boolean>(){public Boolean call(String s) {return !s.matches(".*[a-zA-Z]+.*");}});
		JavaRDD<String> data3 = data2.map(new Function<String, String>() {
			public String call(String line) throws Exception {
				String[] parts = line.split(";");
				for(int i=0; i<parts.length; i++) {
					if(!parts[i].contains("."))
						parts[i] = parts[i] + ".0";
					}
				return String.join(";", parts);
				}
			});
		
		JavaRDD<LabeledPoint> parsedData = data3.map(new Function<String, LabeledPoint>() {
			public LabeledPoint call(String line) throws Exception {
				String[] parts = line.split(";");
				//System.out.println(line);
				return new LabeledPoint(Double.parseDouble(parts[11]),
						Vectors.dense(Double.parseDouble(parts[0]),
								Double.parseDouble(parts[1]),
								Double.parseDouble(parts[2]),
								Double.parseDouble(parts[3]),
								Double.parseDouble(parts[4]),
								Double.parseDouble(parts[5]),
								Double.parseDouble(parts[6]),
								Double.parseDouble(parts[7]),
								Double.parseDouble(parts[8]),
								Double.parseDouble(parts[9]),
								Double.parseDouble(parts[10])));
                    }
                });
		
		return parsedData;
	}
	
	public static void main(String[] args) {
		// configure spark
		SparkConf conf = new SparkConf().setAppName("WineQualityPrediction")
				.setMaster("local[2]").set("spark.executor.memory","2g");
		
		// start a spark context
		JavaSparkContext jsc = new JavaSparkContext(conf);
		
		// provide path to data transformed as [feature vectors]
		String trainingfile = "data/cs643/TrainingDataset.csv";
		String valfile = "data/cs643/ValidationDataset.csv";
		//String trainingfile = "s3://TrainingDataset.csv";
		//String valfile = "s3://ValidationDataset.csv";
		
		JavaRDD<LabeledPoint> datatrain = importRDDdata(jsc, trainingfile);
		JavaRDD<LabeledPoint> dataval = importRDDdata(jsc, valfile);
				
		// Run training algorithm to build the model.
		LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
				.setNumClasses(10)
				.run(datatrain.rdd());
		
		// Compute raw scores on the test set.
		JavaPairRDD<Object, Object> predictionAndLabels = dataval.mapToPair(p ->
		new Tuple2<>(model.predict(((LabeledPoint) p).features()), ((LabeledPoint) p).label()));
		
		// get evaluation metrics
		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
		double accuracy = metrics.accuracy();
		System.out.println("Accuracy = " + accuracy);
		
		// After training, save model to local for prediction in future  
		model.save(jsc.sc(), "LogisticRegressionClassifier");
		
		// stop the spark context
		jsc.stop();
		jsc.close();
		}
}
