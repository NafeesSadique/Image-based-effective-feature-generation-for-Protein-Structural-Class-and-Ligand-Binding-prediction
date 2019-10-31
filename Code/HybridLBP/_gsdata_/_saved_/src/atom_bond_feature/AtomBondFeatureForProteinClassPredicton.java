package atom_bond_feature;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.StringTokenizer;

public class AtomBondFeatureForProteinClassPredicton {
	static String[] atomArray = { "N", "C", "O", "S", "H", "N1+", "O1+", "O1-", "X", "D" };
	static int[] chemicalNumber = { 12, 14, 16, 14, 1, 14, 16, 16, 131, 162 };
	static int sequenceLength = 100;
	static double bondTheshold = 90.43011059;

	// for info
	static String atoms = "";
	static int[] globalAtomCount = new int[atomArray.length];
	static int[] maxGlobalAtomCount = new int[atomArray.length];
	static int uniqueAtomCount = 0;
	static int maxLineCount = 0;
	static String largestData;

	public static String runFeatureExtraction(File pdbFile) {
		String outString = "";
		String sid = "";
		int lineCount = 0;

		// for counting atoms
		int countArray[] = new int[atomArray.length];
		Arrays.fill(countArray, 0);

		// for atom's position
		int positionArray[] = new int[sequenceLength];
		Arrays.fill(positionArray, 0);

		// for atom bond count
		@SuppressWarnings("unchecked")
		ArrayList<double[]>[] importantAtoms = new ArrayList[4]; // C,N,O,H
		importantAtoms[0] = new ArrayList<double[]>(); // C
		importantAtoms[1] = new ArrayList<double[]>(); // N
		importantAtoms[2] = new ArrayList<double[]>(); // O
		importantAtoms[3] = new ArrayList<double[]>(); // H
		long[] bond = new long[6]; // C~N, C~O, C~H, H~N, O~H, O~N
		Arrays.fill(bond, 0);
		double[] minDistance = new double[6]; // C~N, C~O, C~H, H~N, O~H, O~N
		Arrays.fill(minDistance, Double.POSITIVE_INFINITY);
		double[] maxDistance = new double[6]; // C~N, C~O, C~H, H~N, O~H, O~N
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

				// getting sid
				if (numOfTokens == 6) {																//change it
					if (tokens[0].equalsIgnoreCase("REMARK")) {
						if (tokens[4].equalsIgnoreCase("-sid:")) {
							sid = tokens[5].toString();
						}
					}
				}

				else if (tokens[0].toLowerCase().equals("atom")) {							//change it
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
					} else if (str.equalsIgnoreCase("H")) {
						xyz[0] = Double.parseDouble(tokens[lastIndex - 5]);
						xyz[1] = Double.parseDouble(tokens[lastIndex - 4]);
						xyz[2] = Double.parseDouble(tokens[lastIndex - 3]);
						importantAtoms[3].add(xyz);
					}
					lineCount++;
				}
			}
			sc.close();
			if (lineCount > maxLineCount) {
				maxLineCount = lineCount;
				largestData = pdbFile.getName();
			}
		} catch (Exception e) {
			System.out.println(lineCount);
			e.printStackTrace();
		}

		// storing atom's count
		for (int i = 0; i < atomArray.length; i++) {
			// outString += countArray[i] + ", ";
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
		
		// C~H
		for (int i = 0; i < importantAtoms[0].size(); i++) {
			for (int j = 0; j < importantAtoms[3].size(); j++) {
				double[] C_xyz = importantAtoms[0].get(i);
				double[] H_xyz = importantAtoms[3].get(j);
				double distance = Math.sqrt(Math.pow((C_xyz[0] - H_xyz[0]), 2) + Math.pow((C_xyz[1] - H_xyz[1]), 2)
						+ Math.pow((C_xyz[2] - H_xyz[2]), 2));
				if (distance <= bondTheshold)
					bond[2]++;
				if (distance > maxDistance[2])
					maxDistance[2] = distance;
				if (distance < minDistance[2])
					minDistance[2] = distance;
			}
		}
		
		// H~N
		for (int i = 0; i < importantAtoms[3].size(); i++) {
			for (int j = 0; j < importantAtoms[1].size(); j++) {
				double[] H_xyz = importantAtoms[3].get(i);
				double[] N_xyz = importantAtoms[1].get(j);
				double distance = Math.sqrt(Math.pow((H_xyz[0] - N_xyz[0]), 2) + Math.pow((H_xyz[1] - N_xyz[1]), 2)
						+ Math.pow((H_xyz[2] - N_xyz[2]), 2));
				if (distance <= bondTheshold)
					bond[3]++;
				if (distance > maxDistance[3])
					maxDistance[3] = distance;
				if (distance < minDistance[3])
					minDistance[3] = distance;
			}
		}
		
		// O~H
		for (int i = 0; i < importantAtoms[2].size(); i++) {
			for (int j = 0; j < importantAtoms[3].size(); j++) {
				double[] O_xyz = importantAtoms[2].get(i);
				double[] H_xyz = importantAtoms[3].get(j);
				double distance = Math.sqrt(Math.pow((O_xyz[0] - H_xyz[0]), 2) + Math.pow((O_xyz[1] - H_xyz[1]), 2)
						+ Math.pow((O_xyz[2] - H_xyz[2]), 2));
				if (distance <= bondTheshold)
					bond[4]++;
				if (distance > maxDistance[4])
					maxDistance[4] = distance;
				if (distance < minDistance[4])
					minDistance[4] = distance;
			}
		}
		
		// O~N
		for (int i = 0; i < importantAtoms[2].size(); i++) {
			for (int j = 0; j < importantAtoms[1].size(); j++) {
				double[] O_xyz = importantAtoms[2].get(i);
				double[] N_xyz = importantAtoms[1].get(j);
				double distance = Math.sqrt(Math.pow((O_xyz[0] - N_xyz[0]), 2) + Math.pow((O_xyz[1] - N_xyz[1]), 2)
						+ Math.pow((O_xyz[2] - N_xyz[2]), 2));
				if (distance <= bondTheshold)
					bond[5]++;
				if (distance > maxDistance[5])
					maxDistance[5] = distance;
				if (distance < minDistance[5])
					minDistance[5] = distance;
			}
		}

		// storing atom's bond count
		long sum = bond[0] + bond[1] + bond[2] + bond[3] + bond[4] + bond[5];
		for (int i = 0; i < bond.length; i++) {
			// outString += bond[i] + ", ";
			
			//Atom pair bond percentage in PDB
			if (sum != 0)
				outString += ((double) bond[i] / sum) * 100 + ", ";
			else
				outString += 0 + ", ";

//			//minimum distance of a pair
//			if (minDistance[i] == Double.POSITIVE_INFINITY)
//				outString += "-" + ", ";
//			else
//				outString += minDistance[i] + ", ";
//
//			//maximum distance of pair
//			if (maxDistance[i] == Double.MIN_VALUE)
//				outString += "-" + ", ";
//			else
//				outString += maxDistance[i] + ", ";
		}
		
		//sid as class
//		outString += sid;

		return outString.replaceAll(",\\s*$", "~" + sid);
	}

}
