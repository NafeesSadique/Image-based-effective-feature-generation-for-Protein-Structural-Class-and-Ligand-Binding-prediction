package classifiers;

import java.io.File;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.output.prediction.PlainText;
import weka.classifiers.misc.InputMappedClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;

public class RandomForest {

	public void run(Instances data) throws Exception
	{
		weka.classifiers.trees.RandomForest rf = new weka.classifiers.trees.RandomForest();
	
		
		Evaluation eval = new Evaluation(data);
		rf.buildClassifier(data);
		System.out.println("test");
		weka.core.SerializationHelper.write("./randomForest.model", rf);
		
		Classifier cls = (Classifier) weka.core.SerializationHelper.read("./randomForest.model");
		
		
		eval.crossValidateModel(cls, data, 3, new Random(1));
		System.out.println(eval.toSummaryString("\n RandomForest Results\n======\n", true));
		double accuracy = eval.pctCorrect();
		System.out.println("Accuracy = " + accuracy);
	}
	
	public void runForClass(Instances data) throws Exception
	{
		weka.classifiers.trees.RandomForest rf = new weka.classifiers.trees.RandomForest();
	
		
//		Evaluation eval = new Evaluation(data);
//		rf.buildClassifier(data);
		
	
		
		DataSource source = new DataSource("R_MULSUB_ULBP_and_LBPCLBP.arff");
		Instances test_data = source.getDataSet();
		test_data.setClassIndex(test_data.numAttributes() -1);
		System.out.println(data.numAttributes());
		System.out.println(test_data.numAttributes());
		System.out.println(test_data.numAttributes());
		System.out.println(data.equalHeaders(test_data));
		//
		//perform your prediction 
		//System.out.println(test_data.numAttributes());
		//System.out.println(data.numAttributes());
		//System.out.println(data.equalHeaders(test_data));
//		for(int i=0;i<test_data.numInstances();i++)
//		{
//			double value=rf.classifyInstance(test_data.instance(i));
//			//get the name of the class value 
//			String prediction=data.classAttribute().value((int)value);
//			System.out.println(
//                    Integer.toString(i)+
//                    ": "+prediction);
//		}
	
		 
		
		//
		
		
	}
	
	public void createModel(Instances data) throws Exception
	{
		weka.classifiers.trees.RandomForest rf = new weka.classifiers.trees.RandomForest();
	
		
		Evaluation eval = new Evaluation(data);
		rf.buildClassifier(data);
		System.out.println("test");
		weka.core.SerializationHelper.write("./randomForest.model", rf);
	}
	
	public void predictClass() throws Exception
	{
		DataSource source1 = new DataSource("/home/neaz/Oxygen-workspace/Protein Class Prediction/Rand_MulandSum_LBPCLBP_SMOTE.arff");
		Instances data = source1.getDataSet();
		data.setClassIndex(data.numAttributes() -1);
		//System.out.println(data.numAttributes());
		weka.classifiers.trees.RandomForest rf = new weka.classifiers.trees.RandomForest();
//	
//		
//		Evaluation eval = new Evaluation(data);
		rf.buildClassifier(data);
		
		//Classifier cls = (Classifier) weka.core.SerializationHelper.read("./randomForestWithoutBOND.model");
		
		
		
		DataSource source = new DataSource("R_MULSUB_ULBP_and_LBPCLBP.arff");
		Instances test_data = source.getDataSet();
		test_data.setClassIndex(test_data.numAttributes() -1);
		
		
		//System.out.println(test_data.numAttributes());
		//System.out.println(test_data.equalHeaders(data));
		//double clsLabel = cls.classifyInstance(test_data.get(0));
        //System.out.println(clsLabel);
//		System.out.println(test_data.numInstances());
//			double value=rf.classifyInstance(data_test.instance(0));
//			//get the name of the class value 
//			String prediction=data.classAttribute().value((int)value);
//			System.out.println(
//                    Integer.toString(0)+
//                    ": "+prediction);
		
		InputMappedClassifier mappedCls = new InputMappedClassifier();
		//rf.buildClassifier(data);
		//Classifier cls = (Classifier) weka.core.SerializationHelper.read("./randomForestWithoutBOND.model");
	    mappedCls.setModelHeader(data);
	    mappedCls.setClassifier(rf);
	    mappedCls.setSuppressMappingReport(true);
	    
	    mappedCls.buildClassifier(data);
	    
//	    
	    Evaluation eval = new Evaluation(data);
	    
	  //  eval.evaluateModelOnce(rf, test_data.instance(0));
	    double value = eval.evaluateModelOnceAndRecordPrediction(mappedCls, test_data.instance(0)); 
//	    double value=mappedCls.classifyInstance(test_data.instance(0));
	    //System.out.println(value);
		String prediction=data.classAttribute().value((int)value);
		System.out.println(
                Integer.toString(0)+
                ": "+prediction);
	    
	  

		
		
	}
	
	
}
