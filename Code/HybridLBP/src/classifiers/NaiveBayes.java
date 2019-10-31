package classifiers;
import java.util.Random;

import weka.classifiers.*;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

public class NaiveBayes {
	
	public void run(Instances data) throws Exception
	{
		weka.classifiers.bayes.NaiveBayes nb = new weka.classifiers.bayes.NaiveBayes();
		nb.setBatchSize("100");
		
		
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(nb, data, 3, new Random(1));
		System.out.println(eval.toSummaryString("\n Naive Baysian Results\n======\n", true));
		double accuracy = eval.pctCorrect();
		System.out.println("Accuracy = " + accuracy);
		
		
		
	}

}
