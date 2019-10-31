package classifiers;

import java.util.Random;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.core.Instances;

public class SVM {
	
	public void run(Instances data) throws Exception
	{
		SMO svm = new SMO();
		svm.setC(1);
		
		Logistic l = new Logistic();
		l.setMaxIts(-1);
		svm.setCalibrator(l);
		
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(svm, data, 3, new Random(1));
		System.out.println(eval.toSummaryString("\n SVM Results\n======\n", true));
		double accuracy = eval.pctCorrect();
		System.out.println("Accuracy = " + accuracy);
		
		
		
	}

}
