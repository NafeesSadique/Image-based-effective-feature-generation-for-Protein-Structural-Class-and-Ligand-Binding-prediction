package main;


import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.StringTokenizer;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;

public class ImageCreatorForBindingPrediction {
	//variable
	int maxCaCount = 20000;
	double xyz[][];
	String sccs = "scCs";
	File outDir = new File("images");

	public String runFeatureExtraction(File pdbFormatFile) {
		
		xyz = new double[3][maxCaCount];
		if (!pdbFormatFile.getName().endsWith(".pdb")) {
			return "";
		}
		int seqNo = 0;
		try {
			Scanner sc = new Scanner(pdbFormatFile);
			while (sc.hasNext()) {
				String line = sc.nextLine();
				line = line.replace("-", " -").replaceAll(" -$","-"); // for separating attached values
				StringTokenizer strTok;
				strTok = new StringTokenizer(line, " ");
				int numOfTokens = strTok.countTokens();
				String tokens[] = new String[numOfTokens];
				int tokenid = 0;
				while (strTok.hasMoreTokens()) {
					tokens[tokenid++] = strTok.nextToken();
				}

				// *********************for train protein pdb*********************
				if (pdbFormatFile.getName().endsWith("_pro_cg.pdb") && tokens[0].startsWith("ATOM")) {
					if (tokens[2].equalsIgnoreCase("ca")) {
						int lastIndex = numOfTokens - 1;
						int occurance = tokens[lastIndex - 1].length() - tokens[lastIndex - 1].replace(".", "").length();
						if (occurance == 2)
							lastIndex++;
						xyz[0][seqNo] = Double.parseDouble(tokens[lastIndex - 5]);
						xyz[1][seqNo] = Double.parseDouble(tokens[lastIndex - 4]);
						xyz[2][seqNo] = Double.parseDouble(tokens[lastIndex - 3]);
						seqNo++;
					}
				}
				// *********************end train protein pdb*********************
				

				// *********************for train ligand pdb*********************
				else if (pdbFormatFile.getName().endsWith("_lig_cg.pdb") && tokens[0].startsWith("HETATM")) {
					// if (tokens[numOfTokens-1].equalsIgnoreCase("c")) {
					int lastIndex = numOfTokens - 1;
					int occurance = tokens[lastIndex - 1].length() - tokens[lastIndex - 1].replace(".", "").length();
					if (occurance == 2)
						lastIndex++;
					xyz[0][seqNo] = Double.parseDouble(tokens[lastIndex - 5]);
					xyz[1][seqNo] = Double.parseDouble(tokens[lastIndex - 4]);
					xyz[2][seqNo] = Double.parseDouble(tokens[lastIndex - 3]);
					seqNo++;
					// }
				}
				// *********************end train ligand pdb*********************

			}
			sc.close();
			int numOfCaAtom = seqNo;

			return runFeatureExtraction(xyz[0], xyz[1], xyz[2], numOfCaAtom);
		} catch (FileNotFoundException e) {
			// logger.fatal(e.getMessage(),e);
			return "Error at line: "+seqNo;
		} catch (NumberFormatException e) {
			// logger.fatal(e.getMessage(),e);
			return "Error at line: "+seqNo;
		} catch (Exception e) {
			return "Error at line: "+seqNo;
		}
	}

	public String runFeatureExtraction(double x[], double y[], double z[], int numOfCAatom) {
		try {
			double[][] calphadistmat = new double[numOfCAatom][numOfCAatom];
			double maxDistance = -1;
			double minDistance = 100000000;
			int n = 0;
			double totalDistData[] = new double[numOfCAatom * numOfCAatom];
			for (int j = 0; j < numOfCAatom; j++) {
				for (int k = 0; k < numOfCAatom; k++) {
					double dist = Math.sqrt((x[j] - x[k]) * (x[j] - x[k]) + (y[j] - y[k]) * (y[j] - y[k]) + (z[j] - z[k]) * (z[j] - z[k]));
					maxDistance = Math.max(maxDistance, dist);
					minDistance = Math.min(minDistance, dist);
					totalDistData[n++] = dist;
				}
			}
			// System.out.print("Maximum Distance: ");
			// System.out.println(maxDistance);
			int noQuantLevel = 255;
//			int camatq1maxval = (int) (maxDistance * 2);
			for (int j = 0; j < numOfCAatom; j++) {
				for (int k = 0; k < numOfCAatom; k++) {
					int valq2 = (int) ((totalDistData[j * numOfCAatom + k] - minDistance) * noQuantLevel / (maxDistance - minDistance));
					calphadistmat[j][k] = valq2;
				}
			}

			//Opencv
			Mat matrix = new Mat(numOfCAatom, numOfCAatom, CvType.CV_8UC1, new Scalar(0));

			for (int i = 0; i < numOfCAatom; i++) {
				for (int j = 0; j < numOfCAatom; j++) {
					matrix.put(i, j, calphadistmat[i][j]);
				}
			}
			// System.out.println(matrix.dump());
			if (!outDir.exists() || !outDir.isDirectory()) {
				outDir.mkdir();
			}

			if (!matrix.empty()) {
				Imgcodecs.imwrite(Execute.img, matrix);
//				System.out.println("conversion successfull");
			} else {
				System.out.println("\t\t\t\t\tconversion unsuccessfull");
				return "Error in image write";
			}

			// printCalphamatImage(calphadistmat, numOfCAatom);
			return "Matrix: " + calphadistmat.toString();

		} catch (Exception e) {
			// logger.fatal(e.getMessage(),e);
			return "Error in Matrix";
		}
	}

}
