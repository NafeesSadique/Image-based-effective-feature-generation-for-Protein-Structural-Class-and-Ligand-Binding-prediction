
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.misc.InputMappedClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class SVM {

	public String predictClass(String testFilePath) throws Exception {
		/*
		 * give the Train dataset path
		 */
		DataSource trainDataset = new DataSource(ClassifyProtein.class.getClass().getResourceAsStream("/Hybrid_LBP_SMOTE.arff"));
		Instances train_data = trainDataset.getDataSet();
		train_data.setClassIndex(train_data.numAttributes() - 1);
		weka.classifiers.functions.SMO svm = new weka.classifiers.functions.SMO();
		svm.setC(1);
		
		Logistic l = new Logistic();
		l.setMaxIts(-1);
		svm.setCalibrator(l);

		// Classifier cls = (Classifier)
		// weka.core.SerializationHelper.read("./randomForestWithoutBOND.model");

		/*
		 * give the single instance dataset path
		 */
		DataSource testDataset = new DataSource(testFilePath);
		Instances test_data = testDataset.getDataSet();
		test_data.setClassIndex(test_data.numAttributes() - 1);


		InputMappedClassifier mappedCls = new InputMappedClassifier();
		mappedCls.setModelHeader(train_data);
		mappedCls.setClassifier(svm);
		mappedCls.setSuppressMappingReport(true);
		mappedCls.buildClassifier(train_data);

		Evaluation eval = new Evaluation(train_data);
		String prediction = "";
		for(int i=0; i<test_data.numInstances(); i++) {
			double value = eval.evaluateModelOnceAndRecordPrediction(mappedCls, test_data.instance(i));
			String show = (i+1) + "  -->  " + train_data.classAttribute().value((int) value);
			prediction += show + "\n";
			System.out.println(show);
		}
		return prediction;
	}

}
