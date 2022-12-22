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
import org.apache.spark.mllib.linalg.Matrix;
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

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;


public class predict {
	
	
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
		List<String> outputlines = new ArrayList<String>();
		
		// configure spark
		SparkConf conf = new SparkConf().setAppName("WineQualityPrediction")
				.setMaster("local[2]").set("spark.executor.memory","2g");
		
		// start a spark context
		JavaSparkContext jsc = new JavaSparkContext(conf);
		
		// provide path to data transformed as [feature vectors]
		//String path = "data/cs643/sample_libsvm_data.txt";
		String testfile = "data/cs643/TestDataset.csv";
		String modelFileName = "LogisticRegressionClassifier";
		
		JavaRDD<LabeledPoint> datatest = importRDDdata(jsc, testfile);
		
		LogisticRegressionModel model = LogisticRegressionModel.load(jsc.sc(), modelFileName);
		
		JavaPairRDD<Object, Object> predictionAndLabels = datatest.mapToPair(p ->
		new Tuple2<>(model.predict(((LabeledPoint) p).features()), ((LabeledPoint) p).label()));
		
		// get evaluation metrics
		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
				
		System.out.format("===== Summary Statistics =====\n");
		System.out.format("Weighted F1 score = %f\n", metrics.weightedFMeasure());
		System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
		
		outputlines.add(String.format("===== Summary Statistics =====\n"));
		outputlines.add(String.format("Weighted F1 score = %f\n", metrics.weightedFMeasure()));
		outputlines.add(String.format("Weighted precision = %f\n", metrics.weightedPrecision()));
		
		try {
		      FileWriter fileWriter = new FileWriter("Output.txt");
		      PrintWriter printWriter = new PrintWriter(fileWriter);
		      for(String s : outputlines) {
		    	  printWriter.println(s);
		      }
		      printWriter.close();
		    } catch (IOException e) {
		      System.out.println("An error occurred.");
		      e.printStackTrace();
		    }
		
		jsc.stop();
		jsc.close();
		
	}

}
