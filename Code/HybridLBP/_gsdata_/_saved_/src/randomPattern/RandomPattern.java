package randomPattern;


import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Range;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import localBinaryPattern.CreateLBP;
import localBinaryPattern.LBPreturnValue;

public class RandomPattern {
	
	File folder = new File("/home/neaz/Oxygen-workspace/tesr files/resized Images 128x128");
	File[] listOfFiles = folder.listFiles();
	StringBuilder stringBuilder = new StringBuilder();
	int totalDirectory=0;
	int fileNotFound=0;
	String scopeid="";
	public void makeDataset() throws FileNotFoundException
	{
		stringBuilder.append(generateRowNames());
		stringBuilder.append("\n");
		System.out.println(stringBuilder.toString());
		System.out.println("Main folder name : " + folder.getName() );
		System.out.println(""+listOfFiles.length);
		int totalFiles = listOfFiles.length;
		int totalInstance=0;
		
		//going inside each folder   
		 for (int i = 0; i < totalFiles; i++) 
		 {
			 totalInstance++;
			
				//taking first file of the folder
				   File fin=new File(listOfFiles[i].getPath());
				   System.out.println(fin.getName());
				   
					if(fin.exists())
					{
						System.out.println("file exists");
						String fileName = fin.getName();
						scopeid=getSccs(fileName);
						Mat image = Imgcodecs.imread(fin.getPath());
						String histogram = createV2(image);
						stringBuilder.append(histogram+","+scopeid);
    					stringBuilder.append("\n");
					}
	
					else
					{
						System.out.println("file not found");
						fileNotFound++;
					}
					System.gc();
		}
		String finalInstances = stringBuilder.toString();
		
		PrintWriter out = new PrintWriter("Random_Substraction_LBPUniform.csv");
		
		out.println(finalInstances);
		out.close();
		
		System.out.println("\ntotal instances : "+ totalInstance+"\ntotal files : "+ totalFiles +"\nfile not found : "+fileNotFound);

	}
	private String generateRowNames() {
    	String rowNames="";
    	for(int i=0;i<256;i++)
    	{
    		
    			rowNames=rowNames+"r"+i+",";
    		
    	}
    	
    	rowNames=rowNames+"class";
    	return rowNames;
	}
	
	   public static String getSccs(String fname)
		{
			String fileName = fname;
			String[] parts = fileName.split("sccs:");
			String sccs = parts[0].trim();
			String[] sccsParts = sccs.split("\\.");
			 return sccsParts[0];
//			 fold = sccsParts[1]; 
//			 superfamily = sccsParts[2];
//			 family = sccsParts[3];
		}
	   
		public String calculateGreyHistogram(Mat image)
		{
			 //splitting the frames in multiple images
		    java.util.List<Mat> images = new ArrayList<Mat>();
		    Core.split(image, images);
		    
			//set the number of bins at 256
		    MatOfInt histSize = new MatOfInt(256);
		    
		    /*
		     * for uniform LBP , set the size to 59
		     */
		    //MatOfInt histSize = new MatOfInt(59);
		    
		    // only one channel
		    MatOfInt channels = new MatOfInt(0);
		    
		    //set of ranges
		    //MatOfFloat histRange = new MatOfFloat(0,256);
		    
		    //set of ranges for uniform LBP
		    MatOfFloat histRange = new MatOfFloat(0,256);
		    
		   
		    //Compute the histogram s for B component(from BGR)
		    Mat hist_b=new Mat();
		    
		   //histogram for gray image
		    Imgproc.calcHist(images.subList(0, 1), channels, new Mat(), hist_b, histSize, histRange);
		    
		    String histogram="";
		    
		    for(int i =0 ;i<256;i++)
			{
				double[] value = hist_b.get(i, 0);
				
				if(i==255)
				{
					histogram+=(int)value[0];
				}
				else
				{
					histogram+=(int)value[0]+"-";
				}
					
			}
		    

			return histogram;
			
		}
	   
	public void create(Mat image)
	{
		
		Mat transformedImage = image.clone();
		//Mat multipliedImage = image.clone();
		Imgproc.cvtColor(image, transformedImage, Imgproc.COLOR_RGB2GRAY);
		//int[][] matrix = new int[image.rows()][image.cols()];
		for(int i=0,a=image.rows()-1;i<image.rows();i++,a--)
		{
			for(int j=0,b=image.cols()-1;j<image.cols();j++,b--)
			{
				double[] valueOne = image.get(i, j);
				double[] valueTwo = image.get(a, b);
				double first_blue = valueOne[0];
				double second_blue = valueTwo[0];
				double first_green = valueOne[0];
				double second_green = valueTwo[0];
				double first_red = valueOne[0];
				double second_red = valueTwo[0];
				int firstPoint=0;
				int secondPoint =0;
				if(first_blue>second_blue)
				{
					firstPoint++;
					
				}
				else
				{
					secondPoint++;
					
				}
				if(first_green>second_green)
				{
					firstPoint++;
				}
				else
				{
					secondPoint++;
				}
				if(first_red>second_red)
				{
					firstPoint++;
				}
				else
				{
					secondPoint++;
				}
				
				if(firstPoint>secondPoint)
				{
//					matrix[i][j]=0;
//					matrix[a][b]=1;
					
					transformedImage.put(i, j, 0);
					transformedImage.put(a, b, 1);
				}
				else
				{
					
//					matrix[i][j]=1;
//					matrix[a][b]=0;
					
					transformedImage.put(i, j, 1);
					transformedImage.put(a, b, 0);
				}
				
			}
		}
		
		Imgproc.Canny(transformedImage, transformedImage, 2, 4);
		
		
		/*
		 * 6 bit binaray 
		 */
		
//		Mat patternImage = new Mat(transformedImage.rows(),transformedImage.cols(),CvType.CV_8UC1);
//		for(int i=0;i<image.rows();i++)
//		{
//			for(int j=0;j<image.cols();j++)
//			{
//				String binary="";
//				if(j<=image.cols()-6)
//				{
//					binary=""+matrix[i][j]+matrix[i][j+1]+matrix[i][j+2]+matrix[i][j+3]+matrix[i][j+4]+matrix[5][j+5];
//					int decimalValue = Integer.parseInt(binary,2);
//					System.out.println(binary);
//					patternImage.put(i, j, decimalValue);
//				}
//				if(j==image.cols()-5)
//				{
//					binary=""+matrix[i][j]+matrix[i][j+1]+matrix[i][j+2]+matrix[i][j+3]+matrix[i][j+4];
//					int decimalValue = Integer.parseInt(binary,2);
//					patternImage.put(i, j, decimalValue);
//					
//				}
//				if(j==image.cols()-4)
//				{
//					binary=""+matrix[i][j]+matrix[i][j+1]+matrix[i][j+2]+matrix[i][j+3];
//					int decimalValue = Integer.parseInt(binary,2);
//					patternImage.put(i, j, decimalValue);
//					
//				}
//				if(j==image.cols()-3)
//				{
//					binary=""+matrix[i][j]+matrix[i][j+1]+matrix[i][j+2];
//					int decimalValue = Integer.parseInt(binary,2);
//					patternImage.put(i, j, decimalValue);
//					
//				}
//				if(j==image.cols()-2)
//				{
//					binary=""+matrix[i][j]+matrix[i][j+1];
//					int decimalValue = Integer.parseInt(binary,2);
//					patternImage.put(i, j, decimalValue);
//					
//				}
//				if(j==image.cols()-1)
//				{
//					patternImage.put(i, j, matrix[i][j]);
//					
//				}
//				
//			}
//		}
		
		
//		for(int i=0;i<image.rows();i++)
//		{
//			for(int j=0;j<image.cols();j++)
//			{
//				double value[] = transformedImage.get(i,j);
//				double value2[] = image.get(i, j);
//				double result[] = new double[3];
//				result[0]=value[0]*value2[0];
//				result[1]=value[0]*value2[1];
//				result[2]=value[0]*value2[2];
//				
//				multipliedImage.put(i, j,result);
//			}
//		}
		
//		Mat distanceMatrix = new Mat(transformedImage.rows(),transformedImage.cols(),CvType.CV_8UC1);
//		
		
		//Imgproc.Canny(multipliedImage, multipliedImage, 3, 6);
		
		//Highgui.imwrite("Random_TopDown/protein_canny_halfRow.jpg", transformedImage);
	}
	
	public String createV2(Mat image)
	{
		Mat newImage = new Mat(image.rows(),image.cols(),CvType.CV_32FC1);
		for(int i=1;i<image.rows()-1;i++)
		{
			
			for(int j=1;j<image.cols()-2;j++)
			{
				
				//System.out.println("matrix ["+(i)+"]["+(j)+"]");
				Mat slice = new Mat(image, new Range(i-1,i+2), new Range(j-1,j+2));
				Mat slice2 = new Mat(image, new Range(i-1,i+2), new Range(j,j+3));
				
				Mat value = createSustraction(slice, slice2);
				//Mat value = createmultiplication(slice);
				
//				CreateLBP lbp = new CreateLBP();
//				LBPreturnValue lbpResult = lbp.createLBP(value);
//				System.out.println(value.type());
				
				/*
				 * For multiplication
				 */
//				for(int k=i-1,a=0;k<i+2;k++,a++)
//				{
//					
//					for(int l=j-1,b=0;l<j+2;l++,b++)
//					{
//						
//						newImage.put(k, l,value.get(a, b));
//						
//						//System.out.println(newImage.get(k, j).toString());
//					}
//				}
				
				//newImage.put(i, j, lbpResult.getDecimalValue());
				
//				CreateLBP lbp = new CreateLBP();
//				LBPreturnValue returnValue = lbp.createLBP(newImage);
//				newImage.put(i, j, returnValue.getDecimalValue());
				
				/*
				 * For Substraction
				 */
				for(int a=i-1,row=0;a<i+2;a++,row++)
				{ 
				
					for(int b=j-1,col=0;b<j+2;b++,col++)
					{
						double[] checkZero = value.get(row, col);
						if(checkZero[0]<0)
						{
							newImage.put(a, b, 0);
						}
						else
						{
							newImage.put(a, b, value.get(row, col));
						}
						
					}
				}
				
				
//				for(int a=i-1;a>i+2;a++)
//				{
//					for(int b=j-1;j>j+2;j++)
//					{
//						newImage.put(a, b, value.get(a	, b));
//					}
//				}

				
			}
		}
//		CreateLBP lbp = new CreateLBP();
//		String histogram = lbp.processForHistogram(newImage);
		String histogram = "";
		Imgcodecs.imwrite("Random_Multiplication/protein_substraction.jpg", newImage);
		//return newImage;
		return histogram;
	}
	
	public String create_Substraction(Mat image)
	{
		Mat newImage = new Mat(image.rows(),image.cols(),CvType.CV_32FC1);
		for(int i=1;i<image.rows()-1;i++)
		{
			
			for(int j=1;j<image.cols()-2;j++)
			{
				
				//System.out.println("matrix ["+(i)+"]["+(j)+"]");
				Mat slice = new Mat(image, new Range(i-1,i+2), new Range(j-1,j+2));
				Mat slice2 = new Mat(image, new Range(i-1,i+2), new Range(j,j+3));
				
				Mat value = createSustraction(slice, slice2);
				//Mat value = createmultiplication(slice);
				
//				CreateLBP lbp = new CreateLBP();
//				LBPreturnValue lbpResult = lbp.createLBP(value);
//				System.out.println(value.type());
				
				/*
				 * For multiplication
				 */
//				for(int k=i-1,a=0;k<i+2;k++,a++)
//				{
//					
//					for(int l=j-1,b=0;l<j+2;l++,b++)
//					{
//						
//						newImage.put(k, l,value.get(a, b));
//						
//						//System.out.println(newImage.get(k, j).toString());
//					}
//				}
				
				//newImage.put(i, j, lbpResult.getDecimalValue());
				
//				CreateLBP lbp = new CreateLBP();
//				LBPreturnValue returnValue = lbp.createLBP(newImage);
//				newImage.put(i, j, returnValue.getDecimalValue());
				
				/*
				 * For Substraction
				 */
				for(int a=i-1,row=0;a<i+2;a++,row++)
				{ 
				
					for(int b=j-1,col=0;b<j+2;b++,col++)
					{
						double[] checkZero = value.get(row, col);
						if(checkZero[0]<0)
						{
							newImage.put(a, b, 0);
						}
						else
						{
							newImage.put(a, b, value.get(row, col));
						}
						
					}
				}
				
				
//				for(int a=i-1;a>i+2;a++)
//				{
//					for(int b=j-1;j>j+2;j++)
//					{
//						newImage.put(a, b, value.get(a	, b));
//					}
//				}

				
			}
		}
		CreateLBP lbp = new CreateLBP();
		String histogram = lbp.processForHistogram_UniformLBPNN(newImage);
//		String histogram = "";
//		Highgui.imwrite("Random_Multiplication/protein_substraction.jpg", newImage);
//		//return newImage;
		return histogram;
	}
	
	public String create_Multiplication(Mat image)
	{
		Mat newImage = new Mat(image.rows(),image.cols(),CvType.CV_32FC1);
		for(int i=1;i<image.rows()-1;i++)
		{
			
			for(int j=1;j<image.cols()-2;j++)
			{
				
				//System.out.println("matrix ["+(i)+"]["+(j)+"]");
				Mat slice = new Mat(image, new Range(i-1,i+2), new Range(j-1,j+2));
				Mat slice2 = new Mat(image, new Range(i-1,i+2), new Range(j,j+3));
				
				
				Mat value = createmultiplication(slice);
				

				
				/*
				 * For multiplication
				 */
				for(int k=i-1,a=0;k<i+2;k++,a++)
				{
					
					for(int l=j-1,b=0;l<j+2;l++,b++)
					{
						
						newImage.put(k, l,value.get(a, b));
						
						//System.out.println(newImage.get(k, j).toString());
					}
				}
				


				
			}
		}
		CreateLBP lbp = new CreateLBP();
		String histogram = lbp.processForHistogram_UniformLBPNN(newImage);
//		String histogram = "";
//		Highgui.imwrite("Random_Multiplication/protein_substraction.jpg", newImage);
//		//return newImage;
		return histogram;
	}
	
	private Mat createmultiplication(Mat image) {
		Mat slice = new Mat(3, 3, CvType.CV_32FC1);
		Mat resultslice = slice.clone();
		for(int i=0;i<3;i++)
		{
			for(int j=0;j<3;j++)
			{
				slice.put(i, j, image.get(i, j));
			}
		}
		Mat mat1 = new Mat(1, 3, CvType.CV_32FC1);
		mat1.put(0, 0, slice.get(0, 0));
		mat1.put(0, 1, slice.get(0, 1));
		mat1.put(0, 2, slice.get(0, 2));
		Mat mat2 = new Mat(1, 3, CvType.CV_32FC1);
		mat2.put(0, 0, slice.get(1, 0));
		mat2.put(0, 1, slice.get(1, 1));
		mat2.put(0, 2, slice.get(1, 2));
		Mat mat3 = new Mat(1, 3, CvType.CV_32FC1);
		mat3.put(0, 0, slice.get(2, 0));
		mat3.put(0, 1, slice.get(2, 1));
		mat3.put(0, 2, slice.get(2, 2));
		
		Mat matResult1 = new Mat(1, 3, slice.type());
		Mat matResult2 = new Mat(1, 3, slice.type());
		Mat matResult3 = new Mat(1, 3, slice.type());
		
		Core.gemm(mat1,slice, 1, new Mat(), 0, matResult1);
		Core.gemm(mat2,slice, 1, new Mat(), 0, matResult2);
		Core.gemm( mat3,slice, 1, new Mat(), 0, matResult3);
		//Core.multiply(mat1,slice, matResult1);
		
	
//		Core.normalize(matResult1, matResult1);
//		Core.normalize(matResult2, matResult2);
//		Core.normalize(matResult3, matResult3);
		
		for(int i=0;i<1;i++)
		{
			for(int j=0;j<3;j++)
			{
				resultslice.put(i, j, matResult1.get(i, j));
			}
		}
		for(int i=1;i<2;i++)
		{
			for(int j=0;j<3;j++)
			{
				
				resultslice.put(i, j, matResult2.get(0, j));
			}
		}
		for(int i=2;i<3;i++)
		{
			for(int j=0;j<3;j++)
			{
				resultslice.put(i, j, matResult3.get(0, j));
			}
		}
		
		for(int i=0;i<3;i++)
		{
			for(int j=0;j<3;j++)
			{
				double[] value = resultslice.get(i, j);
				value[0]=(int)value[0]/100;
				if(value[0]>255)
				{	
					value[0]=255;
					resultslice.put(i, j, value);
				}
				else
				{
					resultslice.put(i, j, value);
				}

				
			}
		}
		
		//System.out.println(resultslice.dump());
		
		
		
		return resultslice;
	}
	
	public Mat createSustraction(Mat image1,Mat image2)
	{
		Mat slice = new Mat(3, 3, CvType.CV_32FC1);
		Mat slice2 = new Mat(3, 3, CvType.CV_32FC1);
		Mat resultslice = slice.clone();
		for(int i=0;i<3;i++)
		{
			for(int j=0;j<3;j++)
			{
				slice.put(i, j, image1.get(i, j));
			}
		}
		for(int i=0;i<3;i++)
		{
			for(int j=0;j<3;j++)
			{
				slice2.put(i, j, image2.get(i, j));
			}
		}
	
		
		Mat matResult1 = new Mat(3, 3, slice.type());
	
		
		Core.subtract(slice	, slice2, matResult1);
		//System.out.println(matResult1.dump());
		return matResult1;
	}

	
	
	
	
	public Mat substrationV2(Mat image)
	{
		Mat transformedImage = image.clone();
		//Mat multipliedImage = image.clone();
		Imgproc.cvtColor(image, transformedImage, Imgproc.COLOR_RGB2GRAY);
		//int[][] matrix = new int[image.rows()][image.cols()];
		for(int i=0,a=image.rows()-1;i<image.rows()/2;i++,a--)
		{
			for(int j=0,b=image.cols()-1;j<image.cols();j++,b--)
			{
				double[] valueOne = image.get(i, j);
				double[] valueTwo = image.get(a, b);
				double first_blue = valueOne[0];
				double second_blue = valueTwo[0];
				double first_green = valueOne[0];
				double second_green = valueTwo[0];
				double first_red = valueOne[0];
				double second_red = valueTwo[0];
				int firstPoint=0;
				int secondPoint =0;
				if(first_blue>second_blue)
				{
					firstPoint++;
					
				}
				else
				{
					secondPoint++;
					
				}
				if(first_green>second_green)
				{
					firstPoint++;
				}
				else
				{
					secondPoint++;
				}
				if(first_red>second_red)
				{
					firstPoint++;
				}
				else
				{
					secondPoint++;
				}
				
				if(firstPoint>secondPoint)
				{
//					matrix[i][j]=0;
//					matrix[a][b]=1;
					
					transformedImage.put(i, j, first_blue);
					transformedImage.put(a, b, first_blue-second_blue);
				}
				else
				{
					
//					matrix[i][j]=1;
//					matrix[a][b]=0;
					
					transformedImage.put(i, j, second_blue-first_blue);
					transformedImage.put(a, b, second_blue);
				}
				
			}
		}
		
		//Imgproc.Canny(transformedImage, transformedImage, 2, 4);
		
		Imgcodecs.imwrite("Random_Substraction/protein_Random_Substraction_Half.jpg", transformedImage);
		return transformedImage;
	}
	
	

}
