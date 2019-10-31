package main;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import atom_bond_feature.AtomBondFeatureForProteinClassPredicton;
import atom_bond_feature.LigandAtomBondFeatureForBindingPrediction;
import atom_bond_feature.ProteinAtomBondFeatureForBindingPrediction;
import filter.GaborFilter;
import localBinaryPattern.CreateLBP;
import randomPattern.RandomPattern;

public class R_MULSUB_ULBP_and_LBPCLBP {

	StringBuilder stringBuilder = new StringBuilder();
	int totalDirectory = 0;
	int fileNotFound = 0;
	String scopeid = "";
	public String classNumber; 
	public String type; // protein or ligand

	public String run(File imagePath, File pdbFile) throws FileNotFoundException {
		String instance = null;
		if (imagePath.exists()) {
			// System.out.println("file exists");

			// getClassValue(fileName);

			Mat image = Imgcodecs.imread(imagePath.getPath());
			Size size = new Size(128, 128);
			Mat resizedImage = new Mat();
			Imgproc.resize(image, resizedImage, size);
			/*
			 * Zero Padding Basic LBP ; Uses Real Image
			 */
			CreateLBP lbp = new CreateLBP();
			String basicLbp = lbp.processForHistogram_BasicLBP(image);
			/*
			 * Zero Padding Gabor Filtered Basic LBP ; Uses Real Image
			 */
			GaborFilter gf = new GaborFilter();
			String gaborLbp = lbp.processForHistogram_BasicLBP_forGabor(gf.gabor(image));
			/*
			 * Zero Padding Resized images for Multiplication and Substraction
			 */
			RandomPattern rp = new RandomPattern();
			String mulHist = rp.create_Multiplication(resizedImage);
			String subHist = rp.create_Substraction(resizedImage);

			String atomBond = null;
			if(pdbFile.getName().endsWith("_pro_cg.pdb")) {
				atomBond = ProteinAtomBondFeatureForBindingPrediction.runFeatureExtraction(pdbFile);
				instance = basicLbp + "-" + gaborLbp + "-" + atomBond + "-" + mulHist + "-" + subHist + "-" + pdbFile.getName().replaceAll("_pro_cg.pdb", "");
			}
			else if(pdbFile.getName().endsWith("_lig_cg.pdb")) {
				atomBond= LigandAtomBondFeatureForBindingPrediction.runFeatureExtraction(pdbFile);
				instance = basicLbp + "-" + gaborLbp + "-" + atomBond + "-" + mulHist + "-" + subHist + "-" + pdbFile.getName().replaceAll("_lig_cg.pdb", "");
			}
			else if(pdbFile.getName().endsWith(".ent") || pdbFile.getName().endsWith(".pdb") || pdbFile.getName().endsWith(".txt")) {
				atomBond = AtomBondFeatureForProteinClassPredicton.runFeatureExtraction(pdbFile);
				
			//	Get sid
				Pattern pattern = Pattern.compile("~(.*)$");
				Matcher matcher = pattern.matcher(atomBond);
				String sid = "";
				if (matcher.find())
				{
				    sid = matcher.group(1);
				}
				
			//	Finding Class Name
				Scanner scan = new Scanner(Execute.class.getClass().getResourceAsStream("/Scope-sid to Class info.csv"));
				String token[] = null,className = "?";
				while(scan.hasNext()) {
					token = scan.nextLine().split(",");
					if(token[0].equals(sid)) {
						className = token[1];
						break;
					}
				}
				scan.close();
				
			//  Or
//				InputStreamReader isr = new InputStreamReader(Execute.class.getClass().getResourceAsStream("/Scope-sid to Class info.csv"));
//				BufferedReader br = new BufferedReader(isr);
//				String line;
//				while ((line = br.readLine()) != null) {
//				    System.out.println(line);
//				}
//				br.close();
				
				instance = basicLbp + "-" + gaborLbp + "-" + atomBond.replaceAll("~.*$", "") + "-" + mulHist + "-" + subHist + "-" + className.replace("-", "~");
			}

			instance = instance.replace("-", ",").replace("~", "-");
			
//			System.out.print("\n\t" + (basicLbp.length()-basicLbp.replace("-", "").length()+1));
//			System.out.print("\t" + (gaborLbp.length()-gaborLbp.replace("-", "").length()+1));
//			System.out.print("\t" + (atomBond.length()-atomBond.replace(",", "").length()+1));
//			System.out.print("\t" + (mulHist.length()-mulHist.replace("-", "").length()+1));
//			System.out.print("\t" + (subHist.length()-subHist.replace("-", "").length()+1));
//			System.out.print("\t1");
//			System.out.println("\t--> " + (instance.length()-instance.replace(",", "").length()+1));
//			System.out.println(basicLbp);
//			System.out.println(gaborLbp);
//			System.out.println(atomBond);
//			System.out.println(mulHist);
//			System.out.println(subHist);
		}

		else {
			System.out.println("file not found");
			fileNotFound++;
		}
		System.gc();
		return instance;

	}

	public void getClassValue(String fname) {
		String fileName = fname;
		String[] parts = fileName.split("_");

		int cn = Integer.parseInt(parts[0].trim());
		System.out.println(cn);
		this.classNumber = "" + cn;
		this.type = parts[1].trim();
	}

	public String getSccs(String fname) {
		String fileName = fname;
		String[] parts = fileName.split("sccs:");
		String sccs = parts[0].trim();
		String[] sccsParts = sccs.split("\\.");
		return sccsParts[0];
	}

}
