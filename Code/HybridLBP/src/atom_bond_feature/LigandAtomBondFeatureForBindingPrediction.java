package atom_bond_feature;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.StringTokenizer;

public class LigandAtomBondFeatureForBindingPrediction {
	static String[] atomArray = { "C", "N", "CL", "O", "F", "BR", "S", "P", "I", "SE", "BE", "CU", "O1", "FE", "V", "ZN", "AS", "HG", "MO", "W" };
	static int[] chemicalNumber = { 12, 14, 35, 16, 19, 80, 32, 31, 127, 79, 9, 64, 16, 56, 51, 65, 75, 201, 96, 184 };
	static int sequenceLength = 24;
	static double bondTheshold = 31.85660798;

	// for info
	static String atoms = "";
	static int[] globalAtomCount = new int[atomArray.length];
	static int[] maxGlobalAtomCount = new int[atomArray.length];
	static int uniqueAtomCount = 0;
	static int maxLineCount = 0;
	static String largestData;

	public static String runFeatureExtraction(File pdbFile) {
		String outString = "";
		int lineCount = 0;

		// for counting atoms
		int countArray[] = new int[atomArray.length];
		Arrays.fill(countArray, 0);

		// for atom's position
		int positionArray[] = new int[sequenceLength];
		Arrays.fill(positionArray, 0);

		// for atom bond count
		@SuppressWarnings("unchecked")
		ArrayList<double[]>[] importantAtoms = new ArrayList[3]; // C,N,O
		importantAtoms[0] = new ArrayList<double[]>(); // C
		importantAtoms[1] = new ArrayList<double[]>(); // N
		importantAtoms[2] = new ArrayList<double[]>(); // O
		long[] bond = new long[3]; // C~N, C~O, N~O
		Arrays.fill(bond, 0);
		double[] minDistance = new double[3]; // C~N, C~O, N~O
		Arrays.fill(minDistance, Double.POSITIVE_INFINITY);
		double[] maxDistance = new double[3]; // C~N, C~O, N~O
		Arrays.fill(maxDistance, Double.MIN_VALUE);

		try {
			Scanner sc = new Scanner(pdbFile);
			while (sc.hasNext()) {
				String line = sc.nextLine();
				line = line.replace("-", " -").replaceAll(" -$","-"); // for separating attached values
				StringTokenizer strTok = null;
				strTok = new StringTokenizer(line, " ");
				int numOfTokens = strTok.countTokens();
				String tokens[] = new String[numOfTokens];
				int tokenid = 0;
				while (strTok.hasMoreTokens()) {
					tokens[tokenid++] = strTok.nextToken();
				}
				
				String str = tokens[numOfTokens - 1];

				// get all atoms name
				if (!atoms.contains(str)) {
					atoms += str + ", ";
					uniqueAtomCount++;
				}

				// count atoms
				for (int i = 0; i < atomArray.length; i++) {
					if (str.equalsIgnoreCase(atomArray[i])) {
						countArray[i]++;
						globalAtomCount[i]++;
						if (lineCount < sequenceLength)
							positionArray[lineCount] = chemicalNumber[i];
						if (countArray[i] > maxGlobalAtomCount[i])
							maxGlobalAtomCount[i] = countArray[i];
					}
				}

				// atom bonding
				double[] xyz = new double[3];
				int lastIndex = numOfTokens - 1;
				int occurance = tokens[lastIndex - 1].length() - tokens[lastIndex - 1].replace(".", "").length();
				if (occurance == 2)
					lastIndex++;
				if (str.equalsIgnoreCase("C")) {
					xyz[0] = Double.parseDouble(tokens[lastIndex - 5]);
					xyz[1] = Double.parseDouble(tokens[lastIndex - 4]);
					xyz[2] = Double.parseDouble(tokens[lastIndex - 3]);
					importantAtoms[0].add(xyz);
				} else if (str.equalsIgnoreCase("N")) {
					xyz[0] = Double.parseDouble(tokens[lastIndex - 5]);
					xyz[1] = Double.parseDouble(tokens[lastIndex - 4]);
					xyz[2] = Double.parseDouble(tokens[lastIndex - 3]);
					importantAtoms[1].add(xyz);
				} else if (str.equalsIgnoreCase("O")) {
					xyz[0] = Double.parseDouble(tokens[lastIndex - 5]);
					xyz[1] = Double.parseDouble(tokens[lastIndex - 4]);
					xyz[2] = Double.parseDouble(tokens[lastIndex - 3]);
					importantAtoms[2].add(xyz);
				}
				lineCount++;
			}
			sc.close();
			if (lineCount > maxLineCount) {
				maxLineCount = lineCount;
				largestData = pdbFile.getName();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		// storing atom's count
		for (int i = 0; i < atomArray.length; i++) {
			// outString += countArray[i] +", ";
			outString += ((double) countArray[i] / lineCount) * 100 + ", ";
		}

		// storing atom's position
		for (int i = 0; i < positionArray.length; i++) {
			outString += positionArray[i] + ", ";
		}

		// C~N
		for (int i = 0; i < importantAtoms[0].size(); i++) {
			for (int j = 0; j < importantAtoms[1].size(); j++) {
				double[] C_xyz = importantAtoms[0].get(i);
				double[] N_xyz = importantAtoms[1].get(j);
				double distance = Math.sqrt(Math.pow((C_xyz[0] - N_xyz[0]), 2) + Math.pow((C_xyz[1] - N_xyz[1]), 2)
						+ Math.pow((C_xyz[2] - N_xyz[2]), 2));
				if (distance <= bondTheshold)
					bond[0]++;
				if (distance > maxDistance[0])
					maxDistance[0] = distance;
				if (distance < minDistance[0])
					minDistance[0] = distance;
			}
		}
		// C~O
		for (int i = 0; i < importantAtoms[0].size(); i++) {
			for (int j = 0; j < importantAtoms[2].size(); j++) {
				double[] C_xyz = importantAtoms[0].get(i);
				double[] O_xyz = importantAtoms[2].get(j);
				double distance = Math.sqrt(Math.pow((C_xyz[0] - O_xyz[0]), 2) + Math.pow((C_xyz[1] - O_xyz[1]), 2)
						+ Math.pow((C_xyz[2] - O_xyz[2]), 2));
				if (distance <= bondTheshold)
					bond[1]++;
				if (distance > maxDistance[1])
					maxDistance[1] = distance;
				if (distance < minDistance[1])
					minDistance[1] = distance;
			}
		}
		// N~O
		for (int i = 0; i < importantAtoms[1].size(); i++) {
			for (int j = 0; j < importantAtoms[2].size(); j++) {
				double[] N_xyz = importantAtoms[1].get(i);
				double[] O_xyz = importantAtoms[2].get(j);
				double distance = Math.sqrt(Math.pow((N_xyz[0] - O_xyz[0]), 2) + Math.pow((N_xyz[1] - O_xyz[1]), 2)
						+ Math.pow((N_xyz[2] - O_xyz[2]), 2));
				if (distance <= bondTheshold)
					bond[2]++;
				if (distance > maxDistance[2])
					maxDistance[2] = distance;
				if (distance < minDistance[2])
					minDistance[2] = distance;
			}
		}

		// storing atom's bond count
		for (int i = 0; i < bond.length; i++) {
			// outString += bond[i] + ", ";

			long sum = bond[0] + bond[1] + bond[2];
			if (sum != 0)
				outString += ((double) bond[i] / sum) * 100 + ", ";
			else
				outString += 0 + ", ";

//			if (minDistance[i] == Double.POSITIVE_INFINITY)
//				outString += "-" + ",";
//			else
//				outString += minDistance[i] + ",";
//
//			if (maxDistance[i] == Double.MIN_VALUE)
//				outString += "-" + ",";
//			else
//				outString += maxDistance[i] + ",";
		}
//		outString += pdbFile.getName().replace("_lig_cg.pdb", "");
		return outString.replaceAll(",\\s*$", "");
	}

}
