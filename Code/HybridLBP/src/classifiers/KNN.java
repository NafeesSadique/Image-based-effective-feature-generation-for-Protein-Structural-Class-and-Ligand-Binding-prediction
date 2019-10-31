package classifiers;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Random;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;

public class KNN {
	
	
	public void run(Instances data,int nearestNeighbour) throws Exception
	{
		IBk knn = new IBk();
		knn.setKNN(nearestNeighbour);
		LinearNNSearch lnn = new LinearNNSearch();
		EuclideanDistance ed = new EuclideanDistance();
		lnn.setDistanceFunction(ed);	
		knn.setNearestNeighbourSearchAlgorithm(lnn);
		
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(knn, data, 3, new Random(1));
		
		
		
		System.out.println(eval.toSummaryString("\nResults\n======\n", true));
		double accuracy = eval.pctCorrect();
		System.out.println("Accuracy = " + accuracy);
		
		
	}

}
