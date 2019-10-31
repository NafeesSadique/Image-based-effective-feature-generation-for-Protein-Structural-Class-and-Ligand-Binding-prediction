package main;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Random;

import classifiers.Adaboost;
import classifiers.KNN;
import classifiers.NaiveBayes;
import classifiers.RandomForest;
import classifiers.SVM;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;
public class WekaClassifier {

	static String fileName = "/home/neaz/Oxygen-workspace/Protein Class Prediction/Rand_MulandSum_LBPCLBP_SMOTE.arff";
	static Instances data;
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
	
		WekaClassifier wc = new WekaClassifier();
		wc.loadDataset();
		wc.runClassifiers("class");

	
	

	}
	
	public void loadDataset() throws Exception
	{
		
		DataSource source = new DataSource(fileName);
		data = source.getDataSet();
		data.setClassIndex(data.numAttributes() -1);
		System.out.println(data.classAttribute());
	}
	
	public void runClassifiers(String fileName) throws Exception
	{
		PrintStream console = System.out;
		File file = new File(fileName+".txt");
		FileOutputStream fos = new FileOutputStream(file);
		PrintStream ps = new PrintStream(fos);
		System.setOut(ps);
		
//		KNN knn = new KNN();
//		knn.run(data, 3);	
//		
//		NaiveBayes nb = new NaiveBayes();
//		nb.run(data);
//		
//		SVM svm = new SVM();
//		svm.run(data);
//		
//		Adaboost ab = new Adaboost();
//		ab.run(data);
//		
		RandomForest rf = new RandomForest();
		rf.predictClass();
		
		System.setOut(console);
	}
	
	public void CSVtoARFF() throws IOException
	{
		// load CSV
	    CSVLoader loader = new CSVLoader();
	    loader.setSource(new File("./R_MULSUB_ULBP_and_LBPCLBP.csv"));
	    Instances data = loader.getDataSet();
	    
	 // save ARFF
	    ArffSaver saver = new ArffSaver();
	    saver.setInstances(data);
	    saver.setFile(new File("R_MULSUB_ULBP_and_LBPCLBP.arff"));
	    saver.writeBatch();

	}
	
	
	

}
