package classifiers;

import java.util.Random;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class Adaboost {
	
	public void run(Instances data) throws Exception
	{
		AdaBoostM1 ab = new AdaBoostM1();
		J48 j48 = new J48();
		ab.setClassifier(j48);
		
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(ab, data, 3, new Random(1));
		System.out.println(eval.toSummaryString("\n AbaBoost Results\n======\n", true));
		double accuracy = eval.pctCorrect();
		System.out.println("Accuracy = " + accuracy);
		
	}

}
