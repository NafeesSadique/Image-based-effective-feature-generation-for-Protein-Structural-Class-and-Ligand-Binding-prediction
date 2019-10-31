
/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package prepare_dataset;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;

public class RandomUndersampling {

	private static Scanner scan1, scan2;

	public void makeRandomUndersample(File proteinData, File ligandData, String des) {
		FileOutputStream fout = null;
		Random rand = new Random();
		System.out.println("\nRandom Undersampling...");
		int lineCount = 0, minCount = 0;
		try {
		//	get minimum number of pairs
			scan1 = new Scanner(proteinData);
			scan1.nextLine();
			while(scan1.hasNext()) {
				scan1.nextLine();
				lineCount++;
			}
			minCount = lineCount;
			lineCount=0;
			scan1.close();
			scan2 = new Scanner(ligandData);
			scan2.nextLine();
			while(scan2.hasNext()) {
				scan2.nextLine();
				lineCount++;
			}
			if(lineCount < minCount)
				minCount = lineCount;
			scan2.close();
			
			fout = new FileOutputStream(new File(des));
			scan1 = new Scanner(proteinData);
			scan2 = new Scanner(ligandData);
			// header
			String proteinHeader = scan1.nextLine();
			String ligandHeader = scan2.nextLine();
			scan2.close();
			String header = proteinHeader.replaceAll("[^,]+$", "") + ligandHeader.replaceAll(",[^,]+$", "") + ",Bond\n";
			fout.write(header.getBytes());
			int pCount = 0;
			while (scan1.hasNext()) {
				pCount++;
				String proteinInstance = scan1.nextLine();
				String[] token1 = proteinInstance.split(",");
				scan2 = new Scanner(ligandData);
				scan2.nextLine(); // for header reading and advancing to instance
				int randCount = rand.nextInt(minCount) + 1;
				while (pCount == randCount)
					randCount = rand.nextInt(minCount) + 1;
				int lCount = 0;
				int dataCount = 0;
				while (scan2.hasNext()) {
					lCount++;
					String ligandInstance = scan2.nextLine();
					String[] token2 = ligandInstance.split(",");
					if (token1[token1.length - 1].trim().equals(token2[token2.length - 1].trim())) {
						System.out.println("\t" + token1[token1.length - 1] + ", " + token2[token2.length - 1] + " -> yes");
						String instance = proteinInstance.replaceAll("[^,]+$", "")
								+ ligandInstance.replaceAll(",[^,]+$", "") + ",yes" + "\n";
						fout.write(instance.getBytes());
						dataCount++;
					} else if (lCount == randCount) {
						System.out.println("\t" + token1[token1.length - 1] + ", " + token2[token2.length - 1] + " -> no");
						String instance = proteinInstance.replaceAll("[^,]+$", "")
								+ ligandInstance.replaceAll(",[^,]+$", "") + ",no" + "\n";
						fout.write(instance.getBytes());
						dataCount++;
					}
					if(dataCount==2)
						break;
				}
				scan2.close();
				if(dataCount<2) {
					System.out.println("\n***Warning: Data is imbalanced.\n\tMake sure \"input/Binding Prediction (Protein PDB + Ligand PDB)\" has same name of each pair of Protein & Ligand.....");
//					System.out.println("pCount: "+pCount+"\nlCount"+lCount+"\nrandCount"+randCount);
					scan1.close();
					fout.close();
					return;
				}
			}
			scan1.close();
			fout.close();
		} catch (IOException e) {
			System.out.println("\n***Error: Feature file is missing...");
			e.printStackTrace();
		}
		
	}
}
