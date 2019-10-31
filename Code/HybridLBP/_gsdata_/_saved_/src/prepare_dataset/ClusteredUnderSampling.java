package prepare_dataset;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

import main.Execute;

public class ClusteredUnderSampling {
	
	// read files
	static File proteinData = null;
	static File ligandData = null;
	static File clusturedYesData = null;
	
	// write files
	static File clusturedNoData = null;
	static File finalData = null;

	// data info
	static int positiveInstanceNumber;

	static int clusterNumber;
	static int[] clusterCount = null;
	static double[][] clusterCenter = null;
	static double[] maxDistance = null;

	private static Scanner scan1, scan2;
	

	public ClusteredUnderSampling(String proteinPath, String ligandPath, String clusterSrc, String des, int numCluster) {
		clusterNumber = numCluster;
		clusterCount = new int[clusterNumber];
		clusterCenter = new double[clusterNumber][Execute.proteinFeature + Execute.ligandFeature];
		maxDistance = new double[clusterNumber];
		// read files
		proteinData = new File(proteinPath);
		ligandData = new File(ligandPath);
		clusturedYesData = new File(clusterSrc);
		
		// write files
		clusturedNoData = new File("output/Binding Prediction/Temporary File Negative Cluster.csv");
		finalData = new File(des);
		
		try {
			setClusterInfo();
			
			System.out.println("\nTotal Positive Instances: " + positiveInstanceNumber + "\n");
			System.out.println("Positive Cluster Counts:(" + clusterNumber + ")"
					+ "\n" + Arrays.toString(clusterCount).replace("[", "").replace("]","")+"\n");			
//			System.out.println("Maximum Distance in each cluster: (" + clusterNumber + ")"
//					+ "\n" + Arrays.toString(maxDistance).replace("[", "").replace("]","")+"\n");
			
			if(makeNoDataset()) {
				appendDataset();
				System.out.println("Done Undersampling.......");
			}
			else {
				System.out.println("Can't create Clustered Undersampling"
						+ "\n\t(Very few instances are available for " + clusterNumber + " clusters)");
			}
			clusturedNoData.delete();
		} catch (Exception e) {
			System.out.println("\n***Error: Temporary file is missing...");
			e.printStackTrace();
		}
	}

	private static void setClusterInfo() throws Exception {
		// setting cluster count
		scan1 = new Scanner(clusturedYesData);
		scan1.nextLine();		// for reading header and advance to instance
		int instanceCount = 0;
		while (scan1.hasNext()) {
			instanceCount++;
			String instance = scan1.nextLine();
			String str[] = instance.split(",");
			for (int clusterNo = 0; clusterNo < clusterNumber; clusterNo++) {
				if (str[str.length - 1].equals("cluster" + clusterNo)) {
					clusterCount[clusterNo]++;
					for (int feature = 0; feature < (Execute.proteinFeature + Execute.ligandFeature); feature++) {
						clusterCenter[clusterNo][feature] += Double.parseDouble(str[feature]);	// adding up clusters
					}
				}
			}
		}
		scan1.close();
		positiveInstanceNumber = instanceCount;
		
		// setting cluster center
		for (int clusterNo = 0; clusterNo < clusterNumber; clusterNo++) {
			for (int feature = 0; feature < (Execute.proteinFeature + Execute.ligandFeature); feature++) {
				clusterCenter[clusterNo][feature] /= clusterCount[clusterNo];	// average as centers
			}
		}
		
		// setting max distance
		scan1 = new Scanner(clusturedYesData);
		scan1.nextLine();		// for reading header and advance to instance
		while (scan1.hasNext()) {
			String instance = scan1.nextLine();
			String str[] = instance.split(",");
			for (int clusterNo = 0; clusterNo < clusterNumber; clusterNo++) {
				if (str[str.length - 1].equals("cluster" + clusterNo)) {
					double distance = 0.0;
					for (int feature = 0; feature < (Execute.proteinFeature + Execute.ligandFeature); feature++) {
						distance += Math.pow(clusterCenter[clusterNo][feature] - Double.parseDouble(str[feature]), 2);
					}
					distance = Math.sqrt(distance);
					if(distance > maxDistance[clusterNo]) {
						maxDistance[clusterNo] = distance;
					}
				}
			}
		}
		scan1.close();
	}
	
	private static boolean makeNoDataset() throws Exception {
		ArrayList<Integer> checked = new ArrayList<>();
		Random rand = new Random();
		System.out.println("Cluster Based Undersampling..."
				+ "\n(this might take some time)");
		FileOutputStream fout = new FileOutputStream(clusturedNoData);
		
		scan1 = new Scanner(clusturedYesData);
		String clusterHeader = scan1.nextLine()+"\n";
		scan1.close();
		fout.write(clusterHeader.getBytes());
		
		int[] tempClusterCount = new int[clusterNumber];
		Arrays.fill(tempClusterCount, 0);
		
		int pNum = 1;
		int lNum = 0;
		int possible = positiveInstanceNumber*positiveInstanceNumber;
		int test = 0;
		
		while(true) {
			test++;
		// check if completed
			boolean isCompleted = true; 
			for(int i=0; i < clusterNumber; i++) {
				if(tempClusterCount[i] < clusterCount[i]) {
					isCompleted = false;
				}
			}
			if(isCompleted) {
				fout.close();
				return true;
			}
			else if(test > possible) {
				fout.close();
				return false;
			}
					
			
			/*
			 * if instance number is low, random pair can't ensure to check every possible pair
			 * So sequential samples are safer
			 */
			
			if(positiveInstanceNumber < 1000) {
			//	sequential samples
				lNum++;
				if(lNum > positiveInstanceNumber) {
					lNum = 1;
					pNum++;
					if(pNum > positiveInstanceNumber) {
						fout.close();
						return false;
					}
				}
			}
			else {
			// random samples
				pNum = rand.nextInt(positiveInstanceNumber) + 1;
				lNum = rand.nextInt(positiveInstanceNumber) + 1;
				int pair = Integer.valueOf(String.valueOf(pNum)+String.valueOf(lNum));
				//checking if already checked
				while( checked.contains(pair) || pNum==lNum ) {
					pNum = rand.nextInt(positiveInstanceNumber) + 1;
					lNum = rand.nextInt(positiveInstanceNumber) + 1;
					pair = Integer.valueOf(String.valueOf(pNum)+String.valueOf(lNum));
				}
				checked.add(pair);
			}
			
			
			
		// find protein
			String proteinInstance = null;
			String[] token1 = null;
			int pCount = 0;
			scan1 = new Scanner(proteinData);
			scan1.nextLine(); // read header to advance to instance
			while (scan1.hasNext()) {
				proteinInstance = scan1.nextLine();
				pCount++;
				if(pCount==pNum)
					break;
			}
			token1 = proteinInstance.split(",");
			scan1.close();
			
		// find ligand
			String ligandInstance = null;
			String[] token2 = null;
			int lCount = 0;
			scan2 = new Scanner(ligandData);
			scan2.nextLine(); // read header to advance to instance
			while (scan2.hasNext()) {
				ligandInstance = scan2.nextLine();
				lCount++;
				if(lCount==lNum)
					break;
			}
			token2 = ligandInstance.split(",");
			scan2.close();
			
			if(token1[token1.length-1].trim().equals(token2[token2.length-1].trim()))		// check if same Protein~Ligand number
				continue;
			
		//  finding closest cluster
			int closestCluster = -1;
			double minDistance = Double.MAX_VALUE;
			for(int clusterNo=0; clusterNo < clusterNumber; clusterNo++) {
			// checking if in cluster
				double distance = 0.0;
				// for protein distance
				for(int i=0; i<Execute.proteinFeature; i++) {
					distance += Math.pow( (clusterCenter[clusterNo][i] - Double.parseDouble(token1[i])) , 2);
				}
				// for ligand distance
				for(int i=0; i<Execute.ligandFeature; i++) {
					distance += Math.pow( (clusterCenter[clusterNo][Execute.proteinFeature+i] - Double.parseDouble(token2[i])) , 2);
				}
				distance = Math.sqrt(distance);
				
				if ( (distance <= maxDistance[clusterNo]) && (distance <= minDistance) ) {
					minDistance = distance;
					closestCluster = clusterNo;
				}
			}
			
		//  store negative-cluster
			if( (closestCluster >= 0) && (tempClusterCount[closestCluster] < clusterCount[closestCluster])) {
				System.out.println("\t" + token1[token1.length - 1] + ", " + token2[token2.length - 1] + " -> no -> cluster" + closestCluster);
				String instance = proteinInstance.replaceAll(",[^,]+$", ",")
						+ ligandInstance.replaceAll(",[^,]+$", ",")
						+ "no,"
						+ "cluster" + closestCluster + "\n";
				fout.write(instance.getBytes());
				tempClusterCount[closestCluster]++;
			}
			else {
//				System.out.println("\t" + token1[token1.length - 1] + ", " + token2[token2.length - 1] + " -> outside");
			}
		}
	}
	
	
	private static void appendDataset() throws Exception {
		FileOutputStream fout = new FileOutputStream(finalData);
		
		// writing yes data
		scan1 = new Scanner(clusturedYesData);
		String clusterHeader = scan1.nextLine() + "\n";
		fout.write(clusterHeader.getBytes());
		while (scan1.hasNext()) {
			String instance = scan1.nextLine().replaceAll(",[^,]+$", "") + "\n";
			fout.write(instance.getBytes());
		}
		scan1.close();
		
		// writing no data
		scan2 = new Scanner(clusturedNoData);
		scan2.nextLine();	// reading header and advance to instance
		while (scan2.hasNext()) {
			String instance = scan2.nextLine().replaceAll(",[^,]+$", "") + "\n";
			fout.write(instance.getBytes());
		}
		scan2.close();
		
		fout.close();
		
	}
}
