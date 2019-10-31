import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;

public class ClassifyProtein {
	
	static String testFilePath = "input/TestDataset.csv";
	static String outputFilePath = "output/prediction.txt";
	
	public static void main(String[] args) throws Exception {
		
		// Making output directory
		File inputFolder = new File("input");
		if (!inputFolder.exists() || !inputFolder.isDirectory())
			inputFolder.mkdir();
		File outputFolder = new File("output");
		if (!outputFolder.exists() || !outputFolder.isDirectory())
			outputFolder.mkdir();

		ClassifyProtein wc = new ClassifyProtein();
		wc.loadDataset();
		wc.runClassifiers(outputFilePath);
		System.out.println("\n***check prediction(s) in \"" + outputFilePath + "\"file");
		
		System.out.println("Press Enter to exit.......");
		System.in.read();
	}

	public void loadDataset() throws Exception {

		DataSource source = new DataSource(ClassifyProtein.class.getClass().getResourceAsStream("/Hybrid_LBP_SMOTE.arff"));
		Instances data = source.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		System.out.println("Available Classes:" + "\n" + data.classAttribute().toString().replaceAll("@[^,]+,", "\t").replace(",", "\n\t").replace("}", ""));
		System.out.println("\npredicting..." + "(it may take some time)");
	}

	public void runClassifiers(String outputFile) throws Exception {
		File file = new File(outputFile);
		PrintStream ps = new PrintStream(file);

	//	RandomForest
//		RandomForest rf = new RandomForest();
//		String prediction = rf.predictClass(testFilePath);
	
	//	SVM
		SVM svm = new SVM();
		String prediction = svm.predictClass(testFilePath);
		
		ps.println(prediction);
		ps.close();
	}

	public static void CSVtoARFF(String csvFilePath) throws IOException {
		if(csvFilePath.endsWith(".csv")) {
		 // load CSV
			CSVLoader loader = new CSVLoader();
			loader.setSource(new File(testFilePath));
			Instances data = loader.getDataSet();

		 // save ARFF
			ArffSaver saver = new ArffSaver();
			saver.setInstances(data);
			saver.setFile(new File(csvFilePath.replace(".csv", ".arff")));
			saver.writeBatch();
		}
	}
	
	public static void ARFFtoCSV(String arffFilePath) throws IOException {
		
	 // load ARFF
	    ArffLoader loader = new ArffLoader();
	    loader.setSource(new File(arffFilePath));
	    Instances data = loader.getDataSet();
	    
	 // save CSV
	    CSVSaver saver = new CSVSaver();
	    saver.setInstances(data);
	    saver.setFile(new File(arffFilePath.replace(".arff", ".csv")));
	    saver.writeBatch();
	}
}
