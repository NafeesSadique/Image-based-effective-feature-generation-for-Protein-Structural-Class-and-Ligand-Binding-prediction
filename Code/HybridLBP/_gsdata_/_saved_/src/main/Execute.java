package main;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Scanner;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import prepare_dataset.ClusteredUnderSampling;
import prepare_dataset.RandomUndersampling;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class Execute {
	
//	read
	public static int proteinFeature = 736;		//for Protein-Ligand Binding dataset only
	public static int ligandFeature = 677;		//for Protein-Ligand Binding dataset only
	
	static String proteinClassInputDirectory = "input/Class Prediction (Protein PDB)";
	
	static String bindingInputDirectory = "input/Binding Prediction (Protein PDB + Ligand PDB)";
		
//	write
	static String proteinDatasetPath = "output/Class Prediction/Hybrid_LBP_protein_dataset.csv";
	static String proteinFeaturePath = "output/Binding Prediction/ProteinFeature.csv";
	static String ligandFeaturePath = "output/Binding Prediction/LigandFeature.csv";
	static String mergedDatasetPath = "output/Binding Prediction/HybridLBP_(merged).csv";
	static String randomSampleDatasetPath = "output/Binding Prediction/HybridLBP_(random).csv";
	static String clusterSampleDatasetPath = "output/Binding Prediction/HybridLBP_(cluster).csv";
	
//	reduce check
	static String allData = "";
	static String img = "images/Temporary.jpg";
	
	public static void main(String[] args) throws Exception {
		initialize();
		
	 // for Protein Class Prediction
		generateProteinFeaturesForClassPrediction(proteinClassInputDirectory, proteinDatasetPath);
		
	 // for Protein-Ligand Binding Prediction
		generateProteinLigandFeaturesForBindingPrediction(bindingInputDirectory, proteinFeaturePath, ligandFeaturePath);
		
	 //	for making dataset for Protein-Ligand Binding
//		makeRandomUndersampledDataset(proteinFeaturePath, ligandFeaturePath, randomSampleDatasetPath);
//		makeClusteredUndersampledDataset(proteinFeaturePath, ligandFeaturePath, mergedDatasetPath, clusterSampleDatasetPath);
		
		System.out.println("\nAll features are extracted into \"output\" folder");
		
		System.out.println("Press Enter to exit.......");
		System.in.read();
	}

	public static void initialize() {
		// Loading opencv library
		nu.pattern.OpenCV.loadShared();
		System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);

		// Making output directory
		File outputFolder = new File("output/Class Prediction");
		if (!outputFolder.exists() || !outputFolder.isDirectory())
			outputFolder.mkdirs();
		outputFolder = new File("output/Binding Prediction");
		if (!outputFolder.exists() || !outputFolder.isDirectory())
			outputFolder.mkdirs();

	}

	public static void generateProteinFeaturesForClassPrediction(String proteinSrc, String des) throws FileNotFoundException {
		//initialize reduced dataset info
		initializeRreduceDatasetList();
		
		System.out.println("Generating Protein Features For Class Prediction...");
		String header = "basicLbp_0,basicLbp_1,basicLbp_2,basicLbp_3,basicLbp_4,basicLbp_5,basicLbp_6,basicLbp_7,basicLbp_8,basicLbp_9,basicLbp_10,basicLbp_11,basicLbp_12,basicLbp_13,basicLbp_14,basicLbp_15,basicLbp_16,basicLbp_17,basicLbp_18,basicLbp_19,basicLbp_20,basicLbp_21,basicLbp_22,basicLbp_23,basicLbp_24,basicLbp_25,basicLbp_26,basicLbp_27,basicLbp_28,basicLbp_29,basicLbp_30,basicLbp_31,basicLbp_32,basicLbp_33,basicLbp_34,basicLbp_35,basicLbp_36,basicLbp_37,basicLbp_38,basicLbp_39,basicLbp_40,basicLbp_41,basicLbp_42,basicLbp_43,basicLbp_44,basicLbp_45,basicLbp_46,basicLbp_47,basicLbp_48,basicLbp_49,basicLbp_50,basicLbp_51,basicLbp_52,basicLbp_53,basicLbp_54,basicLbp_55,basicLbp_56,basicLbp_57,basicLbp_58,basicLbp_59,basicLbp_60,basicLbp_61,basicLbp_62,basicLbp_63,basicLbp_64,basicLbp_65,basicLbp_66,basicLbp_67,basicLbp_68,basicLbp_69,basicLbp_70,basicLbp_71,basicLbp_72,basicLbp_73,basicLbp_74,basicLbp_75,basicLbp_76,basicLbp_77,basicLbp_78,basicLbp_79,basicLbp_80,basicLbp_81,basicLbp_82,basicLbp_83,basicLbp_84,basicLbp_85,basicLbp_86,basicLbp_87,basicLbp_88,basicLbp_89,basicLbp_90,basicLbp_91,basicLbp_92,basicLbp_93,basicLbp_94,basicLbp_95,basicLbp_96,basicLbp_97,basicLbp_98,basicLbp_99,basicLbp_100,basicLbp_101,basicLbp_102,basicLbp_103,basicLbp_104,basicLbp_105,basicLbp_106,basicLbp_107,basicLbp_108,basicLbp_109,basicLbp_110,basicLbp_111,basicLbp_112,basicLbp_113,basicLbp_114,basicLbp_115,basicLbp_116,basicLbp_117,basicLbp_118,basicLbp_119,basicLbp_120,basicLbp_121,basicLbp_122,basicLbp_123,basicLbp_124,basicLbp_125,basicLbp_126,basicLbp_127,basicLbp_128,basicLbp_129,basicLbp_130,basicLbp_131,basicLbp_132,basicLbp_133,basicLbp_134,basicLbp_135,basicLbp_136,basicLbp_137,basicLbp_138,basicLbp_139,basicLbp_140,basicLbp_141,basicLbp_142,basicLbp_143,basicLbp_144,basicLbp_145,basicLbp_146,basicLbp_147,basicLbp_148,basicLbp_149,basicLbp_150,basicLbp_151,basicLbp_152,basicLbp_153,basicLbp_154,basicLbp_155,basicLbp_156,basicLbp_157,basicLbp_158,basicLbp_159,basicLbp_160,basicLbp_161,basicLbp_162,basicLbp_163,basicLbp_164,basicLbp_165,basicLbp_166,basicLbp_167,basicLbp_168,basicLbp_169,basicLbp_170,basicLbp_171,basicLbp_172,basicLbp_173,basicLbp_174,basicLbp_175,basicLbp_176,basicLbp_177,basicLbp_178,basicLbp_179,basicLbp_180,basicLbp_181,basicLbp_182,basicLbp_183,basicLbp_184,basicLbp_185,basicLbp_186,basicLbp_187,basicLbp_188,basicLbp_189,basicLbp_190,basicLbp_191,basicLbp_192,basicLbp_193,basicLbp_194,basicLbp_195,basicLbp_196,basicLbp_197,basicLbp_198,basicLbp_199,basicLbp_200,basicLbp_201,basicLbp_202,basicLbp_203,basicLbp_204,basicLbp_205,basicLbp_206,basicLbp_207,basicLbp_208,basicLbp_209,basicLbp_210,basicLbp_211,basicLbp_212,basicLbp_213,basicLbp_214,basicLbp_215,basicLbp_216,basicLbp_217,basicLbp_218,basicLbp_219,basicLbp_220,basicLbp_221,basicLbp_222,basicLbp_223,basicLbp_224,basicLbp_225,basicLbp_226,basicLbp_227,basicLbp_228,basicLbp_229,basicLbp_230,basicLbp_231,basicLbp_232,basicLbp_233,basicLbp_234,basicLbp_235,basicLbp_236,basicLbp_237,basicLbp_238,basicLbp_239,basicLbp_240,basicLbp_241,basicLbp_242,basicLbp_243,basicLbp_244,basicLbp_245,basicLbp_246,basicLbp_247,basicLbp_248,basicLbp_249,basicLbp_250,basicLbp_251,basicLbp_252,basicLbp_253,basicLbp_254,basicLbp_255,gaborLbp_0,gaborLbp_1,gaborLbp_2,gaborLbp_3,gaborLbp_4,gaborLbp_5,gaborLbp_6,gaborLbp_7,gaborLbp_8,gaborLbp_9,gaborLbp_10,gaborLbp_11,gaborLbp_12,gaborLbp_13,gaborLbp_14,gaborLbp_15,gaborLbp_16,gaborLbp_17,gaborLbp_18,gaborLbp_19,gaborLbp_20,gaborLbp_21,gaborLbp_22,gaborLbp_23,gaborLbp_24,gaborLbp_25,gaborLbp_26,gaborLbp_27,gaborLbp_28,gaborLbp_29,gaborLbp_30,gaborLbp_31,gaborLbp_32,gaborLbp_33,gaborLbp_34,gaborLbp_35,gaborLbp_36,gaborLbp_37,gaborLbp_38,gaborLbp_39,gaborLbp_40,gaborLbp_41,gaborLbp_42,gaborLbp_43,gaborLbp_44,gaborLbp_45,gaborLbp_46,gaborLbp_47,gaborLbp_48,gaborLbp_49,gaborLbp_50,gaborLbp_51,gaborLbp_52,gaborLbp_53,gaborLbp_54,gaborLbp_55,gaborLbp_56,gaborLbp_57,gaborLbp_58,gaborLbp_59,gaborLbp_60,gaborLbp_61,gaborLbp_62,gaborLbp_63,gaborLbp_64,gaborLbp_65,gaborLbp_66,gaborLbp_67,gaborLbp_68,gaborLbp_69,gaborLbp_70,gaborLbp_71,gaborLbp_72,gaborLbp_73,gaborLbp_74,gaborLbp_75,gaborLbp_76,gaborLbp_77,gaborLbp_78,gaborLbp_79,gaborLbp_80,gaborLbp_81,gaborLbp_82,gaborLbp_83,gaborLbp_84,gaborLbp_85,gaborLbp_86,gaborLbp_87,gaborLbp_88,gaborLbp_89,gaborLbp_90,gaborLbp_91,gaborLbp_92,gaborLbp_93,gaborLbp_94,gaborLbp_95,gaborLbp_96,gaborLbp_97,gaborLbp_98,gaborLbp_99,gaborLbp_100,gaborLbp_101,gaborLbp_102,gaborLbp_103,gaborLbp_104,gaborLbp_105,gaborLbp_106,gaborLbp_107,gaborLbp_108,gaborLbp_109,gaborLbp_110,gaborLbp_111,gaborLbp_112,gaborLbp_113,gaborLbp_114,gaborLbp_115,gaborLbp_116,gaborLbp_117,gaborLbp_118,gaborLbp_119,gaborLbp_120,gaborLbp_121,gaborLbp_122,gaborLbp_123,gaborLbp_124,gaborLbp_125,gaborLbp_126,gaborLbp_127,gaborLbp_128,gaborLbp_129,gaborLbp_130,gaborLbp_131,gaborLbp_132,gaborLbp_133,gaborLbp_134,gaborLbp_135,gaborLbp_136,gaborLbp_137,gaborLbp_138,gaborLbp_139,gaborLbp_140,gaborLbp_141,gaborLbp_142,gaborLbp_143,gaborLbp_144,gaborLbp_145,gaborLbp_146,gaborLbp_147,gaborLbp_148,gaborLbp_149,gaborLbp_150,gaborLbp_151,gaborLbp_152,gaborLbp_153,gaborLbp_154,gaborLbp_155,gaborLbp_156,gaborLbp_157,gaborLbp_158,gaborLbp_159,gaborLbp_160,gaborLbp_161,gaborLbp_162,gaborLbp_163,gaborLbp_164,gaborLbp_165,gaborLbp_166,gaborLbp_167,gaborLbp_168,gaborLbp_169,gaborLbp_170,gaborLbp_171,gaborLbp_172,gaborLbp_173,gaborLbp_174,gaborLbp_175,gaborLbp_176,gaborLbp_177,gaborLbp_178,gaborLbp_179,gaborLbp_180,gaborLbp_181,gaborLbp_182,gaborLbp_183,gaborLbp_184,gaborLbp_185,gaborLbp_186,gaborLbp_187,gaborLbp_188,gaborLbp_189,gaborLbp_190,gaborLbp_191,gaborLbp_192,gaborLbp_193,gaborLbp_194,gaborLbp_195,gaborLbp_196,gaborLbp_197,gaborLbp_198,gaborLbp_199,gaborLbp_200,gaborLbp_201,gaborLbp_202,gaborLbp_203,gaborLbp_204,gaborLbp_205,gaborLbp_206,gaborLbp_207,gaborLbp_208,gaborLbp_209,gaborLbp_210,gaborLbp_211,gaborLbp_212,gaborLbp_213,gaborLbp_214,gaborLbp_215,gaborLbp_216,gaborLbp_217,gaborLbp_218,gaborLbp_219,gaborLbp_220,gaborLbp_221,gaborLbp_222,gaborLbp_223,gaborLbp_224,gaborLbp_225,gaborLbp_226,gaborLbp_227,gaborLbp_228,gaborLbp_229,gaborLbp_230,gaborLbp_231,gaborLbp_232,gaborLbp_233,gaborLbp_234,gaborLbp_235,gaborLbp_236,gaborLbp_237,gaborLbp_238,gaborLbp_239,gaborLbp_240,gaborLbp_241,gaborLbp_242,gaborLbp_243,gaborLbp_244,gaborLbp_245,gaborLbp_246,gaborLbp_247,gaborLbp_248,gaborLbp_249,gaborLbp_250,gaborLbp_251,gaborLbp_252,gaborLbp_253,gaborLbp_254,gaborLbp_255,N_count(%), C_count(%), O_count(%), S_count(%), H_count(%), N1+_count(%), O1+_count(%), O1-_count(%), X_count(%), D_count(%), atom_1, atom_2, atom_3, atom_4, atom_5, atom_6, atom_7, atom_8, atom_9, atom_10, atom_11, atom_12, atom_13, atom_14, atom_15, atom_16, atom_17, atom_18, atom_19, atom_20, atom_21, atom_22, atom_23, atom_24, atom_25, atom_26, atom_27, atom_28, atom_29, atom_30, atom_31, atom_32, atom_33, atom_34, atom_35, atom_36, atom_37, atom_38, atom_39, atom_40, atom_41, atom_42, atom_43, atom_44, atom_45, atom_46, atom_47, atom_48, atom_49, atom_50, atom_51, atom_52, atom_53, atom_54, atom_55, atom_56, atom_57, atom_58, atom_59, atom_60, atom_61, atom_62, atom_63, atom_64, atom_65, atom_66, atom_67, atom_68, atom_69, atom_70, atom_71, atom_72, atom_73, atom_74, atom_75, atom_76, atom_77, atom_78, atom_79, atom_80, atom_81, atom_82, atom_83, atom_84, atom_85, atom_86, atom_87, atom_88, atom_89, atom_90, atom_91, atom_92, atom_93, atom_94, atom_95, atom_96, atom_97, atom_98, atom_99, atom_100, C~N_P(%), C~O_P(%), C~H_P(%), H~N_P(%), O~H_P(%), O~N_P(%),mulHist_0,mulHist_1,mulHist_2,mulHist_3,mulHist_4,mulHist_5,mulHist_6,mulHist_7,mulHist_8,mulHist_9,mulHist_10,mulHist_11,mulHist_12,mulHist_13,mulHist_14,mulHist_15,mulHist_16,mulHist_17,mulHist_18,mulHist_19,mulHist_20,mulHist_21,mulHist_22,mulHist_23,mulHist_24,mulHist_25,mulHist_26,mulHist_27,mulHist_28,mulHist_29,mulHist_30,mulHist_31,mulHist_32,mulHist_33,mulHist_34,mulHist_35,mulHist_36,mulHist_37,mulHist_38,mulHist_39,mulHist_40,mulHist_41,mulHist_42,mulHist_43,mulHist_44,mulHist_45,mulHist_46,mulHist_47,mulHist_48,mulHist_49,mulHist_50,mulHist_51,mulHist_52,mulHist_53,mulHist_54,mulHist_55,mulHist_56,mulHist_57,mulHist_58,subHist_0,subHist_1,subHist_2,subHist_3,subHist_4,subHist_5,subHist_6,subHist_7,subHist_8,subHist_9,subHist_10,subHist_11,subHist_12,subHist_13,subHist_14,subHist_15,subHist_16,subHist_17,subHist_18,subHist_19,subHist_20,subHist_21,subHist_22,subHist_23,subHist_24,subHist_25,subHist_26,subHist_27,subHist_28,subHist_29,subHist_30,subHist_31,subHist_32,subHist_33,subHist_34,subHist_35,subHist_36,subHist_37,subHist_38,subHist_39,subHist_40,subHist_41,subHist_42,subHist_43,subHist_44,subHist_45,subHist_46,subHist_47,subHist_48,subHist_49,subHist_50,subHist_51,subHist_52,subHist_53,subHist_54,subHist_55,subHist_56,subHist_57,subHist_58,Class";
		PrintWriter outPClass = new PrintWriter(des);
		outPClass.println(header);
		File inputFolder = new File(proteinSrc);
		if (!inputFolder.exists() || !inputFolder.isDirectory())
			inputFolder.mkdirs();
		ImageCreatorForCLassPrediction ic = null;
		File[] listOfFiles = inputFolder.listFiles();
		for (int i = 0; i < listOfFiles.length; i++) {
			System.out.print("Parsing file "+ (i+1) +" : " + listOfFiles[i].getName());
			
			//show all images
//			img = "images/"+listOfFiles[i].getName()+".jpg";
			
			//for reduced check
			if(!checkIfReduced(listOfFiles[i])) {
				System.out.println("\tNot in Reduced dataset...(skipping)");
				continue;
			}
			// End of reduced check
			

			// converting to image
			ic = new ImageCreatorForCLassPrediction();
			ic.runFeatureExtraction(listOfFiles[i]);

			// feature generation
			R_MULSUB_ULBP_and_LBPCLBP featureGeneration = new R_MULSUB_ULBP_and_LBPCLBP();
			if (listOfFiles[i].getName().endsWith(".ent") || listOfFiles[i].getName().endsWith(".pdb")
					|| listOfFiles[i].getName().endsWith(".txt"))
				outPClass.println(featureGeneration.run(new File(img), listOfFiles[i]));
			System.out.println("\t(Done)");
		}
		outPClass.close();

	}
	
	private static void initializeRreduceDatasetList() {
		File reduced = new File("Reduced Dataset Info.csv");
		try {
			Scanner scan = new Scanner(reduced);
			allData =  scan.nextLine();
			scan.close();
			System.out.println("Reduced data will be used");
		} catch (FileNotFoundException e) {
			System.out.println("All data will be used");
//			e.printStackTrace();
		}
		
	}

	private static boolean checkIfReduced(File inputFile) {
		if(allData.equals("") || allData.contains(inputFile.getName().replace(".ent", "")))
			return true;
		return false;
		
	}

	public static void generateProteinLigandFeaturesForBindingPrediction(String bindingInputDirectory, String pDes, String lDes) throws FileNotFoundException {
		System.out.println("\nGenerating Protein & Ligand Features For Binding Prediction...");
		PrintWriter outP = new PrintWriter(pDes);
		PrintWriter outL = new PrintWriter(lDes);
		String header = "P_basicLbp_0,P_basicLbp_1,P_basicLbp_2,P_basicLbp_3,P_basicLbp_4,P_basicLbp_5,P_basicLbp_6,P_basicLbp_7,P_basicLbp_8,P_basicLbp_9,P_basicLbp_10,P_basicLbp_11,P_basicLbp_12,P_basicLbp_13,P_basicLbp_14,P_basicLbp_15,P_basicLbp_16,P_basicLbp_17,P_basicLbp_18,P_basicLbp_19,P_basicLbp_20,P_basicLbp_21,P_basicLbp_22,P_basicLbp_23,P_basicLbp_24,P_basicLbp_25,P_basicLbp_26,P_basicLbp_27,P_basicLbp_28,P_basicLbp_29,P_basicLbp_30,P_basicLbp_31,P_basicLbp_32,P_basicLbp_33,P_basicLbp_34,P_basicLbp_35,P_basicLbp_36,P_basicLbp_37,P_basicLbp_38,P_basicLbp_39,P_basicLbp_40,P_basicLbp_41,P_basicLbp_42,P_basicLbp_43,P_basicLbp_44,P_basicLbp_45,P_basicLbp_46,P_basicLbp_47,P_basicLbp_48,P_basicLbp_49,P_basicLbp_50,P_basicLbp_51,P_basicLbp_52,P_basicLbp_53,P_basicLbp_54,P_basicLbp_55,P_basicLbp_56,P_basicLbp_57,P_basicLbp_58,P_basicLbp_59,P_basicLbp_60,P_basicLbp_61,P_basicLbp_62,P_basicLbp_63,P_basicLbp_64,P_basicLbp_65,P_basicLbp_66,P_basicLbp_67,P_basicLbp_68,P_basicLbp_69,P_basicLbp_70,P_basicLbp_71,P_basicLbp_72,P_basicLbp_73,P_basicLbp_74,P_basicLbp_75,P_basicLbp_76,P_basicLbp_77,P_basicLbp_78,P_basicLbp_79,P_basicLbp_80,P_basicLbp_81,P_basicLbp_82,P_basicLbp_83,P_basicLbp_84,P_basicLbp_85,P_basicLbp_86,P_basicLbp_87,P_basicLbp_88,P_basicLbp_89,P_basicLbp_90,P_basicLbp_91,P_basicLbp_92,P_basicLbp_93,P_basicLbp_94,P_basicLbp_95,P_basicLbp_96,P_basicLbp_97,P_basicLbp_98,P_basicLbp_99,P_basicLbp_100,P_basicLbp_101,P_basicLbp_102,P_basicLbp_103,P_basicLbp_104,P_basicLbp_105,P_basicLbp_106,P_basicLbp_107,P_basicLbp_108,P_basicLbp_109,P_basicLbp_110,P_basicLbp_111,P_basicLbp_112,P_basicLbp_113,P_basicLbp_114,P_basicLbp_115,P_basicLbp_116,P_basicLbp_117,P_basicLbp_118,P_basicLbp_119,P_basicLbp_120,P_basicLbp_121,P_basicLbp_122,P_basicLbp_123,P_basicLbp_124,P_basicLbp_125,P_basicLbp_126,P_basicLbp_127,P_basicLbp_128,P_basicLbp_129,P_basicLbp_130,P_basicLbp_131,P_basicLbp_132,P_basicLbp_133,P_basicLbp_134,P_basicLbp_135,P_basicLbp_136,P_basicLbp_137,P_basicLbp_138,P_basicLbp_139,P_basicLbp_140,P_basicLbp_141,P_basicLbp_142,P_basicLbp_143,P_basicLbp_144,P_basicLbp_145,P_basicLbp_146,P_basicLbp_147,P_basicLbp_148,P_basicLbp_149,P_basicLbp_150,P_basicLbp_151,P_basicLbp_152,P_basicLbp_153,P_basicLbp_154,P_basicLbp_155,P_basicLbp_156,P_basicLbp_157,P_basicLbp_158,P_basicLbp_159,P_basicLbp_160,P_basicLbp_161,P_basicLbp_162,P_basicLbp_163,P_basicLbp_164,P_basicLbp_165,P_basicLbp_166,P_basicLbp_167,P_basicLbp_168,P_basicLbp_169,P_basicLbp_170,P_basicLbp_171,P_basicLbp_172,P_basicLbp_173,P_basicLbp_174,P_basicLbp_175,P_basicLbp_176,P_basicLbp_177,P_basicLbp_178,P_basicLbp_179,P_basicLbp_180,P_basicLbp_181,P_basicLbp_182,P_basicLbp_183,P_basicLbp_184,P_basicLbp_185,P_basicLbp_186,P_basicLbp_187,P_basicLbp_188,P_basicLbp_189,P_basicLbp_190,P_basicLbp_191,P_basicLbp_192,P_basicLbp_193,P_basicLbp_194,P_basicLbp_195,P_basicLbp_196,P_basicLbp_197,P_basicLbp_198,P_basicLbp_199,P_basicLbp_200,P_basicLbp_201,P_basicLbp_202,P_basicLbp_203,P_basicLbp_204,P_basicLbp_205,P_basicLbp_206,P_basicLbp_207,P_basicLbp_208,P_basicLbp_209,P_basicLbp_210,P_basicLbp_211,P_basicLbp_212,P_basicLbp_213,P_basicLbp_214,P_basicLbp_215,P_basicLbp_216,P_basicLbp_217,P_basicLbp_218,P_basicLbp_219,P_basicLbp_220,P_basicLbp_221,P_basicLbp_222,P_basicLbp_223,P_basicLbp_224,P_basicLbp_225,P_basicLbp_226,P_basicLbp_227,P_basicLbp_228,P_basicLbp_229,P_basicLbp_230,P_basicLbp_231,P_basicLbp_232,P_basicLbp_233,P_basicLbp_234,P_basicLbp_235,P_basicLbp_236,P_basicLbp_237,P_basicLbp_238,P_basicLbp_239,P_basicLbp_240,P_basicLbp_241,P_basicLbp_242,P_basicLbp_243,P_basicLbp_244,P_basicLbp_245,P_basicLbp_246,P_basicLbp_247,P_basicLbp_248,P_basicLbp_249,P_basicLbp_250,P_basicLbp_251,P_basicLbp_252,P_basicLbp_253,P_basicLbp_254,P_basicLbp_255,P_gaborLbp_0,P_gaborLbp_1,P_gaborLbp_2,P_gaborLbp_3,P_gaborLbp_4,P_gaborLbp_5,P_gaborLbp_6,P_gaborLbp_7,P_gaborLbp_8,P_gaborLbp_9,P_gaborLbp_10,P_gaborLbp_11,P_gaborLbp_12,P_gaborLbp_13,P_gaborLbp_14,P_gaborLbp_15,P_gaborLbp_16,P_gaborLbp_17,P_gaborLbp_18,P_gaborLbp_19,P_gaborLbp_20,P_gaborLbp_21,P_gaborLbp_22,P_gaborLbp_23,P_gaborLbp_24,P_gaborLbp_25,P_gaborLbp_26,P_gaborLbp_27,P_gaborLbp_28,P_gaborLbp_29,P_gaborLbp_30,P_gaborLbp_31,P_gaborLbp_32,P_gaborLbp_33,P_gaborLbp_34,P_gaborLbp_35,P_gaborLbp_36,P_gaborLbp_37,P_gaborLbp_38,P_gaborLbp_39,P_gaborLbp_40,P_gaborLbp_41,P_gaborLbp_42,P_gaborLbp_43,P_gaborLbp_44,P_gaborLbp_45,P_gaborLbp_46,P_gaborLbp_47,P_gaborLbp_48,P_gaborLbp_49,P_gaborLbp_50,P_gaborLbp_51,P_gaborLbp_52,P_gaborLbp_53,P_gaborLbp_54,P_gaborLbp_55,P_gaborLbp_56,P_gaborLbp_57,P_gaborLbp_58,P_gaborLbp_59,P_gaborLbp_60,P_gaborLbp_61,P_gaborLbp_62,P_gaborLbp_63,P_gaborLbp_64,P_gaborLbp_65,P_gaborLbp_66,P_gaborLbp_67,P_gaborLbp_68,P_gaborLbp_69,P_gaborLbp_70,P_gaborLbp_71,P_gaborLbp_72,P_gaborLbp_73,P_gaborLbp_74,P_gaborLbp_75,P_gaborLbp_76,P_gaborLbp_77,P_gaborLbp_78,P_gaborLbp_79,P_gaborLbp_80,P_gaborLbp_81,P_gaborLbp_82,P_gaborLbp_83,P_gaborLbp_84,P_gaborLbp_85,P_gaborLbp_86,P_gaborLbp_87,P_gaborLbp_88,P_gaborLbp_89,P_gaborLbp_90,P_gaborLbp_91,P_gaborLbp_92,P_gaborLbp_93,P_gaborLbp_94,P_gaborLbp_95,P_gaborLbp_96,P_gaborLbp_97,P_gaborLbp_98,P_gaborLbp_99,P_gaborLbp_100,P_gaborLbp_101,P_gaborLbp_102,P_gaborLbp_103,P_gaborLbp_104,P_gaborLbp_105,P_gaborLbp_106,P_gaborLbp_107,P_gaborLbp_108,P_gaborLbp_109,P_gaborLbp_110,P_gaborLbp_111,P_gaborLbp_112,P_gaborLbp_113,P_gaborLbp_114,P_gaborLbp_115,P_gaborLbp_116,P_gaborLbp_117,P_gaborLbp_118,P_gaborLbp_119,P_gaborLbp_120,P_gaborLbp_121,P_gaborLbp_122,P_gaborLbp_123,P_gaborLbp_124,P_gaborLbp_125,P_gaborLbp_126,P_gaborLbp_127,P_gaborLbp_128,P_gaborLbp_129,P_gaborLbp_130,P_gaborLbp_131,P_gaborLbp_132,P_gaborLbp_133,P_gaborLbp_134,P_gaborLbp_135,P_gaborLbp_136,P_gaborLbp_137,P_gaborLbp_138,P_gaborLbp_139,P_gaborLbp_140,P_gaborLbp_141,P_gaborLbp_142,P_gaborLbp_143,P_gaborLbp_144,P_gaborLbp_145,P_gaborLbp_146,P_gaborLbp_147,P_gaborLbp_148,P_gaborLbp_149,P_gaborLbp_150,P_gaborLbp_151,P_gaborLbp_152,P_gaborLbp_153,P_gaborLbp_154,P_gaborLbp_155,P_gaborLbp_156,P_gaborLbp_157,P_gaborLbp_158,P_gaborLbp_159,P_gaborLbp_160,P_gaborLbp_161,P_gaborLbp_162,P_gaborLbp_163,P_gaborLbp_164,P_gaborLbp_165,P_gaborLbp_166,P_gaborLbp_167,P_gaborLbp_168,P_gaborLbp_169,P_gaborLbp_170,P_gaborLbp_171,P_gaborLbp_172,P_gaborLbp_173,P_gaborLbp_174,P_gaborLbp_175,P_gaborLbp_176,P_gaborLbp_177,P_gaborLbp_178,P_gaborLbp_179,P_gaborLbp_180,P_gaborLbp_181,P_gaborLbp_182,P_gaborLbp_183,P_gaborLbp_184,P_gaborLbp_185,P_gaborLbp_186,P_gaborLbp_187,P_gaborLbp_188,P_gaborLbp_189,P_gaborLbp_190,P_gaborLbp_191,P_gaborLbp_192,P_gaborLbp_193,P_gaborLbp_194,P_gaborLbp_195,P_gaborLbp_196,P_gaborLbp_197,P_gaborLbp_198,P_gaborLbp_199,P_gaborLbp_200,P_gaborLbp_201,P_gaborLbp_202,P_gaborLbp_203,P_gaborLbp_204,P_gaborLbp_205,P_gaborLbp_206,P_gaborLbp_207,P_gaborLbp_208,P_gaborLbp_209,P_gaborLbp_210,P_gaborLbp_211,P_gaborLbp_212,P_gaborLbp_213,P_gaborLbp_214,P_gaborLbp_215,P_gaborLbp_216,P_gaborLbp_217,P_gaborLbp_218,P_gaborLbp_219,P_gaborLbp_220,P_gaborLbp_221,P_gaborLbp_222,P_gaborLbp_223,P_gaborLbp_224,P_gaborLbp_225,P_gaborLbp_226,P_gaborLbp_227,P_gaborLbp_228,P_gaborLbp_229,P_gaborLbp_230,P_gaborLbp_231,P_gaborLbp_232,P_gaborLbp_233,P_gaborLbp_234,P_gaborLbp_235,P_gaborLbp_236,P_gaborLbp_237,P_gaborLbp_238,P_gaborLbp_239,P_gaborLbp_240,P_gaborLbp_241,P_gaborLbp_242,P_gaborLbp_243,P_gaborLbp_244,P_gaborLbp_245,P_gaborLbp_246,P_gaborLbp_247,P_gaborLbp_248,P_gaborLbp_249,P_gaborLbp_250,P_gaborLbp_251,P_gaborLbp_252,P_gaborLbp_253,P_gaborLbp_254,P_gaborLbp_255,P_C_count(%), P_N_count(%), P_O_count(%), P_N1+_count(%), P_N+1_count(%), P_atom_1, P_atom_2, P_atom_3, P_atom_4, P_atom_5, P_atom_6, P_atom_7, P_atom_8, P_atom_9, P_atom_10, P_atom_11, P_atom_12, P_atom_13, P_atom_14, P_atom_15, P_atom_16, P_atom_17, P_atom_18, P_atom_19, P_atom_20, P_atom_21, P_atom_22, P_atom_23, P_atom_24, P_atom_25, P_atom_26, P_atom_27, P_atom_28, P_atom_29, P_atom_30, P_atom_31, P_atom_32, P_atom_33, P_atom_34, P_atom_35, P_atom_36, P_atom_37, P_atom_38, P_atom_39, P_atom_40, P_atom_41, P_atom_42, P_atom_43, P_atom_44, P_atom_45, P_atom_46, P_atom_47, P_atom_48, P_atom_49, P_atom_50, P_atom_51, P_atom_52, P_atom_53, P_atom_54, P_atom_55, P_atom_56, P_atom_57, P_atom_58, P_atom_59, P_atom_60, P_atom_61, P_atom_62, P_atom_63, P_atom_64, P_atom_65, P_atom_66, P_atom_67, P_atom_68, P_atom_69, P_atom_70, P_atom_71, P_atom_72, P_atom_73, P_atom_74, P_atom_75, P_atom_76, P_atom_77, P_atom_78, P_atom_79, P_atom_80, P_atom_81, P_atom_82, P_atom_83, P_atom_84, P_atom_85, P_atom_86, P_atom_87, P_atom_88, P_atom_89, P_atom_90, P_atom_91, P_atom_92, P_atom_93, P_atom_94, P_atom_95, P_atom_96, P_atom_97, P_atom_98, P_atom_99, P_atom_100, P_C~N,P_mulHist_0,P_mulHist_1,P_mulHist_2,P_mulHist_3,P_mulHist_4,P_mulHist_5,P_mulHist_6,P_mulHist_7,P_mulHist_8,P_mulHist_9,P_mulHist_10,P_mulHist_11,P_mulHist_12,P_mulHist_13,P_mulHist_14,P_mulHist_15,P_mulHist_16,P_mulHist_17,P_mulHist_18,P_mulHist_19,P_mulHist_20,P_mulHist_21,P_mulHist_22,P_mulHist_23,P_mulHist_24,P_mulHist_25,P_mulHist_26,P_mulHist_27,P_mulHist_28,P_mulHist_29,P_mulHist_30,P_mulHist_31,P_mulHist_32,P_mulHist_33,P_mulHist_34,P_mulHist_35,P_mulHist_36,P_mulHist_37,P_mulHist_38,P_mulHist_39,P_mulHist_40,P_mulHist_41,P_mulHist_42,P_mulHist_43,P_mulHist_44,P_mulHist_45,P_mulHist_46,P_mulHist_47,P_mulHist_48,P_mulHist_49,P_mulHist_50,P_mulHist_51,P_mulHist_52,P_mulHist_53,P_mulHist_54,P_mulHist_55,P_mulHist_56,P_mulHist_57,P_mulHist_58,P_subHist_0,P_subHist_1,P_subHist_2,P_subHist_3,P_subHist_4,P_subHist_5,P_subHist_6,P_subHist_7,P_subHist_8,P_subHist_9,P_subHist_10,P_subHist_11,P_subHist_12,P_subHist_13,P_subHist_14,P_subHist_15,P_subHist_16,P_subHist_17,P_subHist_18,P_subHist_19,P_subHist_20,P_subHist_21,P_subHist_22,P_subHist_23,P_subHist_24,P_subHist_25,P_subHist_26,P_subHist_27,P_subHist_28,P_subHist_29,P_subHist_30,P_subHist_31,P_subHist_32,P_subHist_33,P_subHist_34,P_subHist_35,P_subHist_36,P_subHist_37,P_subHist_38,P_subHist_39,P_subHist_40,P_subHist_41,P_subHist_42,P_subHist_43,P_subHist_44,P_subHist_45,P_subHist_46,P_subHist_47,P_subHist_48,P_subHist_49,P_subHist_50,P_subHist_51,P_subHist_52,P_subHist_53,P_subHist_54,P_subHist_55,P_subHist_56,P_subHist_57,P_subHist_58,pro_name";
		outP.println(header);
		header = "L_basicLbp_0,L_basicLbp_1,L_basicLbp_2,L_basicLbp_3,L_basicLbp_4,L_basicLbp_5,L_basicLbp_6,L_basicLbp_7,L_basicLbp_8,L_basicLbp_9,L_basicLbp_10,L_basicLbp_11,L_basicLbp_12,L_basicLbp_13,L_basicLbp_14,L_basicLbp_15,L_basicLbp_16,L_basicLbp_17,L_basicLbp_18,L_basicLbp_19,L_basicLbp_20,L_basicLbp_21,L_basicLbp_22,L_basicLbp_23,L_basicLbp_24,L_basicLbp_25,L_basicLbp_26,L_basicLbp_27,L_basicLbp_28,L_basicLbp_29,L_basicLbp_30,L_basicLbp_31,L_basicLbp_32,L_basicLbp_33,L_basicLbp_34,L_basicLbp_35,L_basicLbp_36,L_basicLbp_37,L_basicLbp_38,L_basicLbp_39,L_basicLbp_40,L_basicLbp_41,L_basicLbp_42,L_basicLbp_43,L_basicLbp_44,L_basicLbp_45,L_basicLbp_46,L_basicLbp_47,L_basicLbp_48,L_basicLbp_49,L_basicLbp_50,L_basicLbp_51,L_basicLbp_52,L_basicLbp_53,L_basicLbp_54,L_basicLbp_55,L_basicLbp_56,L_basicLbp_57,L_basicLbp_58,L_basicLbp_59,L_basicLbp_60,L_basicLbp_61,L_basicLbp_62,L_basicLbp_63,L_basicLbp_64,L_basicLbp_65,L_basicLbp_66,L_basicLbp_67,L_basicLbp_68,L_basicLbp_69,L_basicLbp_70,L_basicLbp_71,L_basicLbp_72,L_basicLbp_73,L_basicLbp_74,L_basicLbp_75,L_basicLbp_76,L_basicLbp_77,L_basicLbp_78,L_basicLbp_79,L_basicLbp_80,L_basicLbp_81,L_basicLbp_82,L_basicLbp_83,L_basicLbp_84,L_basicLbp_85,L_basicLbp_86,L_basicLbp_87,L_basicLbp_88,L_basicLbp_89,L_basicLbp_90,L_basicLbp_91,L_basicLbp_92,L_basicLbp_93,L_basicLbp_94,L_basicLbp_95,L_basicLbp_96,L_basicLbp_97,L_basicLbp_98,L_basicLbp_99,L_basicLbp_100,L_basicLbp_101,L_basicLbp_102,L_basicLbp_103,L_basicLbp_104,L_basicLbp_105,L_basicLbp_106,L_basicLbp_107,L_basicLbp_108,L_basicLbp_109,L_basicLbp_110,L_basicLbp_111,L_basicLbp_112,L_basicLbp_113,L_basicLbp_114,L_basicLbp_115,L_basicLbp_116,L_basicLbp_117,L_basicLbp_118,L_basicLbp_119,L_basicLbp_120,L_basicLbp_121,L_basicLbp_122,L_basicLbp_123,L_basicLbp_124,L_basicLbp_125,L_basicLbp_126,L_basicLbp_127,L_basicLbp_128,L_basicLbp_129,L_basicLbp_130,L_basicLbp_131,L_basicLbp_132,L_basicLbp_133,L_basicLbp_134,L_basicLbp_135,L_basicLbp_136,L_basicLbp_137,L_basicLbp_138,L_basicLbp_139,L_basicLbp_140,L_basicLbp_141,L_basicLbp_142,L_basicLbp_143,L_basicLbp_144,L_basicLbp_145,L_basicLbp_146,L_basicLbp_147,L_basicLbp_148,L_basicLbp_149,L_basicLbp_150,L_basicLbp_151,L_basicLbp_152,L_basicLbp_153,L_basicLbp_154,L_basicLbp_155,L_basicLbp_156,L_basicLbp_157,L_basicLbp_158,L_basicLbp_159,L_basicLbp_160,L_basicLbp_161,L_basicLbp_162,L_basicLbp_163,L_basicLbp_164,L_basicLbp_165,L_basicLbp_166,L_basicLbp_167,L_basicLbp_168,L_basicLbp_169,L_basicLbp_170,L_basicLbp_171,L_basicLbp_172,L_basicLbp_173,L_basicLbp_174,L_basicLbp_175,L_basicLbp_176,L_basicLbp_177,L_basicLbp_178,L_basicLbp_179,L_basicLbp_180,L_basicLbp_181,L_basicLbp_182,L_basicLbp_183,L_basicLbp_184,L_basicLbp_185,L_basicLbp_186,L_basicLbp_187,L_basicLbp_188,L_basicLbp_189,L_basicLbp_190,L_basicLbp_191,L_basicLbp_192,L_basicLbp_193,L_basicLbp_194,L_basicLbp_195,L_basicLbp_196,L_basicLbp_197,L_basicLbp_198,L_basicLbp_199,L_basicLbp_200,L_basicLbp_201,L_basicLbp_202,L_basicLbp_203,L_basicLbp_204,L_basicLbp_205,L_basicLbp_206,L_basicLbp_207,L_basicLbp_208,L_basicLbp_209,L_basicLbp_210,L_basicLbp_211,L_basicLbp_212,L_basicLbp_213,L_basicLbp_214,L_basicLbp_215,L_basicLbp_216,L_basicLbp_217,L_basicLbp_218,L_basicLbp_219,L_basicLbp_220,L_basicLbp_221,L_basicLbp_222,L_basicLbp_223,L_basicLbp_224,L_basicLbp_225,L_basicLbp_226,L_basicLbp_227,L_basicLbp_228,L_basicLbp_229,L_basicLbp_230,L_basicLbp_231,L_basicLbp_232,L_basicLbp_233,L_basicLbp_234,L_basicLbp_235,L_basicLbp_236,L_basicLbp_237,L_basicLbp_238,L_basicLbp_239,L_basicLbp_240,L_basicLbp_241,L_basicLbp_242,L_basicLbp_243,L_basicLbp_244,L_basicLbp_245,L_basicLbp_246,L_basicLbp_247,L_basicLbp_248,L_basicLbp_249,L_basicLbp_250,L_basicLbp_251,L_basicLbp_252,L_basicLbp_253,L_basicLbp_254,L_basicLbp_255,L_gaborLbp_0,L_gaborLbp_1,L_gaborLbp_2,L_gaborLbp_3,L_gaborLbp_4,L_gaborLbp_5,L_gaborLbp_6,L_gaborLbp_7,L_gaborLbp_8,L_gaborLbp_9,L_gaborLbp_10,L_gaborLbp_11,L_gaborLbp_12,L_gaborLbp_13,L_gaborLbp_14,L_gaborLbp_15,L_gaborLbp_16,L_gaborLbp_17,L_gaborLbp_18,L_gaborLbp_19,L_gaborLbp_20,L_gaborLbp_21,L_gaborLbp_22,L_gaborLbp_23,L_gaborLbp_24,L_gaborLbp_25,L_gaborLbp_26,L_gaborLbp_27,L_gaborLbp_28,L_gaborLbp_29,L_gaborLbp_30,L_gaborLbp_31,L_gaborLbp_32,L_gaborLbp_33,L_gaborLbp_34,L_gaborLbp_35,L_gaborLbp_36,L_gaborLbp_37,L_gaborLbp_38,L_gaborLbp_39,L_gaborLbp_40,L_gaborLbp_41,L_gaborLbp_42,L_gaborLbp_43,L_gaborLbp_44,L_gaborLbp_45,L_gaborLbp_46,L_gaborLbp_47,L_gaborLbp_48,L_gaborLbp_49,L_gaborLbp_50,L_gaborLbp_51,L_gaborLbp_52,L_gaborLbp_53,L_gaborLbp_54,L_gaborLbp_55,L_gaborLbp_56,L_gaborLbp_57,L_gaborLbp_58,L_gaborLbp_59,L_gaborLbp_60,L_gaborLbp_61,L_gaborLbp_62,L_gaborLbp_63,L_gaborLbp_64,L_gaborLbp_65,L_gaborLbp_66,L_gaborLbp_67,L_gaborLbp_68,L_gaborLbp_69,L_gaborLbp_70,L_gaborLbp_71,L_gaborLbp_72,L_gaborLbp_73,L_gaborLbp_74,L_gaborLbp_75,L_gaborLbp_76,L_gaborLbp_77,L_gaborLbp_78,L_gaborLbp_79,L_gaborLbp_80,L_gaborLbp_81,L_gaborLbp_82,L_gaborLbp_83,L_gaborLbp_84,L_gaborLbp_85,L_gaborLbp_86,L_gaborLbp_87,L_gaborLbp_88,L_gaborLbp_89,L_gaborLbp_90,L_gaborLbp_91,L_gaborLbp_92,L_gaborLbp_93,L_gaborLbp_94,L_gaborLbp_95,L_gaborLbp_96,L_gaborLbp_97,L_gaborLbp_98,L_gaborLbp_99,L_gaborLbp_100,L_gaborLbp_101,L_gaborLbp_102,L_gaborLbp_103,L_gaborLbp_104,L_gaborLbp_105,L_gaborLbp_106,L_gaborLbp_107,L_gaborLbp_108,L_gaborLbp_109,L_gaborLbp_110,L_gaborLbp_111,L_gaborLbp_112,L_gaborLbp_113,L_gaborLbp_114,L_gaborLbp_115,L_gaborLbp_116,L_gaborLbp_117,L_gaborLbp_118,L_gaborLbp_119,L_gaborLbp_120,L_gaborLbp_121,L_gaborLbp_122,L_gaborLbp_123,L_gaborLbp_124,L_gaborLbp_125,L_gaborLbp_126,L_gaborLbp_127,L_gaborLbp_128,L_gaborLbp_129,L_gaborLbp_130,L_gaborLbp_131,L_gaborLbp_132,L_gaborLbp_133,L_gaborLbp_134,L_gaborLbp_135,L_gaborLbp_136,L_gaborLbp_137,L_gaborLbp_138,L_gaborLbp_139,L_gaborLbp_140,L_gaborLbp_141,L_gaborLbp_142,L_gaborLbp_143,L_gaborLbp_144,L_gaborLbp_145,L_gaborLbp_146,L_gaborLbp_147,L_gaborLbp_148,L_gaborLbp_149,L_gaborLbp_150,L_gaborLbp_151,L_gaborLbp_152,L_gaborLbp_153,L_gaborLbp_154,L_gaborLbp_155,L_gaborLbp_156,L_gaborLbp_157,L_gaborLbp_158,L_gaborLbp_159,L_gaborLbp_160,L_gaborLbp_161,L_gaborLbp_162,L_gaborLbp_163,L_gaborLbp_164,L_gaborLbp_165,L_gaborLbp_166,L_gaborLbp_167,L_gaborLbp_168,L_gaborLbp_169,L_gaborLbp_170,L_gaborLbp_171,L_gaborLbp_172,L_gaborLbp_173,L_gaborLbp_174,L_gaborLbp_175,L_gaborLbp_176,L_gaborLbp_177,L_gaborLbp_178,L_gaborLbp_179,L_gaborLbp_180,L_gaborLbp_181,L_gaborLbp_182,L_gaborLbp_183,L_gaborLbp_184,L_gaborLbp_185,L_gaborLbp_186,L_gaborLbp_187,L_gaborLbp_188,L_gaborLbp_189,L_gaborLbp_190,L_gaborLbp_191,L_gaborLbp_192,L_gaborLbp_193,L_gaborLbp_194,L_gaborLbp_195,L_gaborLbp_196,L_gaborLbp_197,L_gaborLbp_198,L_gaborLbp_199,L_gaborLbp_200,L_gaborLbp_201,L_gaborLbp_202,L_gaborLbp_203,L_gaborLbp_204,L_gaborLbp_205,L_gaborLbp_206,L_gaborLbp_207,L_gaborLbp_208,L_gaborLbp_209,L_gaborLbp_210,L_gaborLbp_211,L_gaborLbp_212,L_gaborLbp_213,L_gaborLbp_214,L_gaborLbp_215,L_gaborLbp_216,L_gaborLbp_217,L_gaborLbp_218,L_gaborLbp_219,L_gaborLbp_220,L_gaborLbp_221,L_gaborLbp_222,L_gaborLbp_223,L_gaborLbp_224,L_gaborLbp_225,L_gaborLbp_226,L_gaborLbp_227,L_gaborLbp_228,L_gaborLbp_229,L_gaborLbp_230,L_gaborLbp_231,L_gaborLbp_232,L_gaborLbp_233,L_gaborLbp_234,L_gaborLbp_235,L_gaborLbp_236,L_gaborLbp_237,L_gaborLbp_238,L_gaborLbp_239,L_gaborLbp_240,L_gaborLbp_241,L_gaborLbp_242,L_gaborLbp_243,L_gaborLbp_244,L_gaborLbp_245,L_gaborLbp_246,L_gaborLbp_247,L_gaborLbp_248,L_gaborLbp_249,L_gaborLbp_250,L_gaborLbp_251,L_gaborLbp_252,L_gaborLbp_253,L_gaborLbp_254,L_gaborLbp_255,L_C_count(%), L_N_count(%), L_CL_count(%), L_O_count(%), L_F_count(%), L_BR_count(%), L_S_count(%), L_P_count(%), L_I_count(%), L_SE_count(%), L_BE_count(%), L_CU_count(%), L_O1_count-(%), L_FE_count(%), L_V_count(%), L_ZN_count(%), L_AS_count(%), L_HG_count(%), L_MO_count(%), L_W_count(%), L_atom_1, L_atom_2, L_atom_3, L_atom_4, L_atom_5, L_atom_6, L_atom_7, L_atom_8, L_atom_9, L_atom_10, L_atom_11, L_atom_12, L_atom_13, L_atom_14, L_atom_15, L_atom_16, L_atom_17, L_atom_18, L_atom_19, L_atom_20, L_atom_21, L_atom_22, L_atom_23, L_atom_24, L_C~N(%), L_C~O(%), L_N~O(%),L_mulHist_0,L_mulHist_1,L_mulHist_2,L_mulHist_3,L_mulHist_4,L_mulHist_5,L_mulHist_6,L_mulHist_7,L_mulHist_8,L_mulHist_9,L_mulHist_10,L_mulHist_11,L_mulHist_12,L_mulHist_13,L_mulHist_14,L_mulHist_15,L_mulHist_16,L_mulHist_17,L_mulHist_18,L_mulHist_19,L_mulHist_20,L_mulHist_21,L_mulHist_22,L_mulHist_23,L_mulHist_24,L_mulHist_25,L_mulHist_26,L_mulHist_27,L_mulHist_28,L_mulHist_29,L_mulHist_30,L_mulHist_31,L_mulHist_32,L_mulHist_33,L_mulHist_34,L_mulHist_35,L_mulHist_36,L_mulHist_37,L_mulHist_38,L_mulHist_39,L_mulHist_40,L_mulHist_41,L_mulHist_42,L_mulHist_43,L_mulHist_44,L_mulHist_45,L_mulHist_46,L_mulHist_47,L_mulHist_48,L_mulHist_49,L_mulHist_50,L_mulHist_51,L_mulHist_52,L_mulHist_53,L_mulHist_54,L_mulHist_55,L_mulHist_56,L_mulHist_57,L_mulHist_58,L_subHist_0,L_subHist_1,L_subHist_2,L_subHist_3,L_subHist_4,L_subHist_5,L_subHist_6,L_subHist_7,L_subHist_8,L_subHist_9,L_subHist_10,L_subHist_11,L_subHist_12,L_subHist_13,L_subHist_14,L_subHist_15,L_subHist_16,L_subHist_17,L_subHist_18,L_subHist_19,L_subHist_20,L_subHist_21,L_subHist_22,L_subHist_23,L_subHist_24,L_subHist_25,L_subHist_26,L_subHist_27,L_subHist_28,L_subHist_29,L_subHist_30,L_subHist_31,L_subHist_32,L_subHist_33,L_subHist_34,L_subHist_35,L_subHist_36,L_subHist_37,L_subHist_38,L_subHist_39,L_subHist_40,L_subHist_41,L_subHist_42,L_subHist_43,L_subHist_44,L_subHist_45,L_subHist_46,L_subHist_47,L_subHist_48,L_subHist_49,L_subHist_50,L_subHist_51,L_subHist_52,L_subHist_53,L_subHist_54,L_subHist_55,L_subHist_56,L_subHist_57,L_subHist_58,lig_name";
		outL.println(header);
		File inputFolder = new File(bindingInputDirectory);
		if (!inputFolder.exists() || !inputFolder.isDirectory())
			inputFolder.mkdirs();
		ImageCreatorForBindingPrediction im = null;
		File[] listOfFiles = inputFolder.listFiles();
		for (int i = 0; i < listOfFiles.length; i++) {
			System.out.print("Parsing file " + (i+1) + " : " + listOfFiles[i].getName());

			//show all images
//			img = "images/"+listOfFiles[i].getName()+".jpg";
			
			//converting to image
			im = new ImageCreatorForBindingPrediction();
			im.runFeatureExtraction(listOfFiles[i]);

			// feature generation
			R_MULSUB_ULBP_and_LBPCLBP featureGeneration = new R_MULSUB_ULBP_and_LBPCLBP();
			if (listOfFiles[i].getName().endsWith("_pro_cg.pdb"))
				outP.println(featureGeneration.run(new File(img), listOfFiles[i]));
			else if (listOfFiles[i].getName().endsWith("_lig_cg.pdb"))
				outL.println(featureGeneration.run(new File(img), listOfFiles[i]));
			System.out.println("\t(Done)");
		}
		outP.close();
		outL.close();

//		Merging Protein & Ligand
		makeMergedDataset(proteinFeaturePath, ligandFeaturePath, mergedDatasetPath);
		
	}
	
	public static void makeMergedDataset(String proteinPath, String ligandPath, String des) {
		File proteinData = new File(proteinPath);
		File ligandData = new File(ligandPath);
		System.out.println("\nMerging Protein & Ligand features in one csv file...");
		FileOutputStream fout = null;
		try {
			fout = new FileOutputStream(new File(des));
			Scanner scan1 = new Scanner(proteinData);
			Scanner scan2 = new Scanner(ligandData);
			// header
			String proteinHeader = scan1.nextLine();
			String ligandHeader = scan2.nextLine();
			scan2.close();
			String header = proteinHeader.replaceAll(",[^,]+$", ",") 
					+ ligandHeader.replaceAll(",[^,]+$", ",") 
					+ "Bond"
					+ "\n";
			fout.write(header.getBytes());
			while (scan1.hasNext()) {
				String proteinInstance = scan1.nextLine();
				String[] token1 = proteinInstance.split(",");
				scan2 = new Scanner(ligandData);
				String ligandInstance = scan2.nextLine();	//read header to advance to instance
				while(scan2.hasNext()) {
					ligandInstance = scan2.nextLine();
					String[] token2 = ligandInstance.split(",");
					if (token1[token1.length - 1].trim().equals(token2[token2.length - 1].trim())) {
						System.out.println("\t" + token1[token1.length - 1] + ", " + token2[token2.length - 1] + " -> yes");
						String instance = proteinInstance.replaceAll(",[^,]+$", ",")
								+ ligandInstance.replaceAll(",[^,]+$", ",") 
								+ "yes"
								+ "\n";
						fout.write(instance.getBytes());
						break;
					}
				}
				scan2.close();
			}
			scan1.close();
			fout.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}

	public static void makeRandomUndersampledDataset(String proteinPath, String ligandPath, String des) {
		File proteinData = new File(proteinPath);
		File ligandData = new File(ligandPath);
		if(!proteinData.isFile() || !ligandData.isFile()) {
			System.out.println("\n***Error: Generate Protein and Ligand features first.....");
			return;
		}
		RandomUndersampling merge = new RandomUndersampling();
		merge.makeRandomUndersample(proteinData, ligandData, des); // random negative pairs
		
	}

	public static void makeClusteredUndersampledDataset(String proteinPath, String ligandPath, String mergedSrc, String des) throws Exception {
		File proteinData = new File(proteinPath);
		File ligandData = new File(ligandPath);
		File mergedData = new File(mergedSrc);
		if(!proteinData.isFile() || !ligandData.isFile() || !mergedData.isFile()) {
			System.out.println("\n***Error: Generate Protein and Ligand features first.....");
			return;
		}
		String tempClusterFile = "output/Binding Prediction/Temporary File Cluster.csv";
		System.out.print("\nMaking clusters...");
		CSVLoader loader = new CSVLoader();
        loader.setSource(mergedData);
        Instances data = loader.getDataSet();
//        PrintStream err=null;
//        System.setErr(err);	//hack to avoid some error messages

        // Create the KMeans object.
        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setNumClusters(10);
        kmeans.setMaxIterations(500);
        kmeans.setPreserveInstancesOrder(true);
        
        // Perform K-Means clustering.
        try {  
            kmeans.buildClusterer(data);
        } catch (Exception ex) {
            System.err.println("Unable to buld Clusterer: " + ex.getMessage());
            ex.printStackTrace();
        }
        
//      get Assignments:
		int[] assignments = kmeans.getAssignments();
		
		PrintWriter out = new PrintWriter(tempClusterFile);
		Scanner scan = new Scanner(mergedData);
		String line = scan.nextLine();
		out.println(line);
		int i=0;
		while(scan.hasNext()) {
			line = scan.nextLine();
			out.println(line + ",cluster" + assignments[i]);
			i++;
		}
		scan.close();
		out.close();
		System.out.println("\t(Done)");
		
		new ClusteredUnderSampling(proteinPath, ligandPath, tempClusterFile, des, kmeans.getNumClusters());
		new File(tempClusterFile).delete();
	}

}
