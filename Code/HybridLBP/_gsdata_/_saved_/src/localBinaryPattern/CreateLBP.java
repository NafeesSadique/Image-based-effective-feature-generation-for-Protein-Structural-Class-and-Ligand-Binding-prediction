package localBinaryPattern;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Range;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;



public class CreateLBP {
	/*
	 * Real Image
	 */
	//File folder = new File("/home/neaz/imageOutput with sccs");
	/*
	 * 128x128 Image
	 */
	
	File folder = new File("/home/neaz/Oxygen-workspace/tesr files/resized Images 128x128");
	
	File[] listOfFiles = folder.listFiles();
	StringBuilder stringBuilder = new StringBuilder();
	int totalDirectory=0;
	int fileNotFound=0;
	String scopeid="";
	
	public void datasetLDP() throws FileNotFoundException
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
						String LDPvalue = process(image);
						stringBuilder.append(LDPvalue+","+scopeid);
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
		
		PrintWriter out = new PrintWriter("LDP.csv");
		
		out.println(finalInstances);
		out.close();
		
		System.out.println("\ntotal instances : "+ totalInstance+"\ntotal files : "+ totalFiles +"\nfile not found : "+fileNotFound);


	}
	
	
	
	public String process(Mat image)
	{
		String LDPvalue ="";
		if(image.empty())
		{
			System.out.println("Empty");
			
		}
		else
		{
			Imgproc.cvtColor(image, image, Imgproc.COLOR_RGB2GRAY);
			
			
			
			for(int i=3;i<128;i+=3)
			{
				
				for(int j=3;j<128;j+=3)
				{
					Mat slice = new Mat(image, new Range(i-3,i), new Range(j-3,j));
					
					LBPreturnValue value = createLBP(slice);
					
					if(i==126 && j==126)
					{
						LDPvalue+=value;
					}
					else
					{
						LDPvalue+=value+"-";
					}
					
					
				}
			}
			
			
			
			
		}
		return LDPvalue;
		
		
	}
	
	/*
	 * 
	 * LBP with grey histogram 
	 * 
	 */
	
	
	public void datasetLBPHistogram() throws FileNotFoundException
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
						String LDPhistogram = processForHistogram_UniformLBPNN(image);
						//String LDPhistogram = processForHistogramRandomPattern(image);
						stringBuilder.append(LDPhistogram+","+scopeid);
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
		
		PrintWriter out = new PrintWriter("./uniformDatasets/ResizedImage_UniformLBP_V3.csv");
		
		out.println(finalInstances);
		out.close();
		
		System.out.println("\ntotal instances : "+ totalInstance+"\ntotal files : "+ totalFiles +"\nfile not found : "+fileNotFound);


	}
	
	public String processForHistogram_UniformLBPNN(Mat image)
	{
		String histogram ="";
		if(image.empty())
		{
			System.out.println("Image is Empty");
			
		}
		else
		{
			//need rgb to GREY
			//Imgproc.cvtColor(image, image, Imgproc.COLOR_RGB2GRAY);
			//Imgproc.Canny(image, image, 300, 600, 5, true); 
			//Mat LBPimage = image.clone();
			Mat LBPimage = new Mat(image.rows(),image.cols(),image.type());
			// New image to put zero around
			Mat testImage = new Mat(image.rows()+2,image.cols()+2,image.type());
			
			//Putting zero in 0th row of the testImage
			for(int i=0;i<1;i++)
			{
				for(int j=0;j<testImage.cols();j++)
				{
					testImage.put(i, j, 0);
				}
			}
			
			//Putting zero in the last row of the testImage
			for(int i=testImage.rows()-1;i<testImage.rows();i++)
			{
				for(int j=0;j<testImage.cols();j++)
				{
					testImage.put(i, j, 0);
				}
			}
			
			//Putting zero in 0th column of the testImage
			for(int i=0;i<testImage.rows();i++)
			{
				for(int j=0;j<1;j++)
				{
					testImage.put(i, j, 0);
				}
			}
			
			//Putting zero in the last row of the testImage
			for(int i=0;i<testImage.rows();i++)
			{
				for(int j=testImage.cols()-1;j<testImage.cols();j++)
				{
					testImage.put(i, j, 0);
				}
			}
			
			//putting rest of the values
			for(int i=1;i<testImage.rows()-1;i++)
			{
				for(int j=1;j<testImage.cols()-1;j++)
				{
					double[] value = image.get(i-1, j-1);
					
					testImage.put(i, j, value[0]);
				}
			}
			
			//System.out.println(testImage.dump());
			//Highgui.imwrite("NN.jpg", testImage);
			
			/*
			 * LBP implementation NN
			 */
			
			for(int i=1;i<testImage.rows()-1;i++)
			{
				
				for(int j=1;j<testImage.cols()-1;j++)
				{
					
					//System.out.println("matrix ["+(i)+"]["+(j)+"]");
					Mat slice = new Mat(testImage, new Range(i-1,i+2), new Range(j-1,j+2));
					
					LBPreturnValue value = createLBPClockwise(slice);
					
					
					//LBPimage.put(i, j, value.getDecimalValue());
					LBPimage.put(i, j, value.getUniformValue());

					
				}
			}
			
		
			
//			Highgui.imwrite("halfImage.jpg", halfImage);
			
			
			histogram = calculateGreyHistogram_Uniform(LBPimage);
			
			//System.out.println(histogram);
			
			
		}
		
	
		return histogram;
		
		
	}
	
	public String processForHistogram_BasicLBP_forGabor(Mat image)
	{
		String histogram ="";
		if(image.empty())
		{
			System.out.println("Image is Empty");
			
		}
		else
		{
			
			
			Mat LBPimage = new Mat(image.rows(),image.cols(),image.type());
			// New image to put zero around
			Mat testImage = new Mat(image.rows()+2,image.cols()+2,image.type());
			
			//Putting zero in 0th row of the testImage
			for(int i=0;i<1;i++)
			{
				for(int j=0;j<testImage.cols();j++)
				{
					testImage.put(i, j, 0);
				}
			}
			
			//Putting zero in the last row of the testImage
			for(int i=testImage.rows()-1;i<testImage.rows();i++)
			{
				for(int j=0;j<testImage.cols();j++)
				{
					testImage.put(i, j, 0);
				}
			}
			
			//Putting zero in 0th column of the testImage
			for(int i=0;i<testImage.rows();i++)
			{
				for(int j=0;j<1;j++)
				{
					testImage.put(i, j, 0);
				}
			}
			
			//Putting zero in the last row of the testImage
			for(int i=0;i<testImage.rows();i++)
			{
				for(int j=testImage.cols()-1;j<testImage.cols();j++)
				{
					testImage.put(i, j, 0);
				}
			}
			
			//putting rest of the values
			for(int i=1;i<testImage.rows()-1;i++)
			{
				for(int j=1;j<testImage.cols()-1;j++)
				{
					double[] value = image.get(i-1, j-1);
					
					testImage.put(i, j, value[0]);
				}
			}
			
			//System.out.println(testImage.dump());
			//Highgui.imwrite("NN.jpg", testImage);
			
			/*
			 * LBP implementation NN
			 */
			
			for(int i=1;i<testImage.rows()-1;i++)
			{
				
				for(int j=1;j<testImage.cols()-1;j++)
				{
					
					//System.out.println("matrix ["+(i)+"]["+(j)+"]");
					Mat slice = new Mat(testImage, new Range(i-1,i+2), new Range(j-1,j+2));
					
					LBPreturnValue value = createLBPClockwise(slice);
					
					
					//LBPimage.put(i, j, value.getDecimalValue());
					LBPimage.put(i, j, value.getDecimalValue());

					
				}
			}
			
		
			
//			Highgui.imwrite("halfImage.jpg", halfImage);
			
			
			histogram = calculateGreyHistogram(LBPimage);
			
			//System.out.println(histogram);
			
			
		}
		
	
		return histogram;
		
		
		
		
	}
	
	public String processForHistogram_BasicLBP(Mat image)
	{
		String histogram ="";
		if(image.empty())
		{
			System.out.println("Image is Empty");
			
		}
		else
		{
			//need rgb to GREY
			Imgproc.cvtColor(image, image, Imgproc.COLOR_RGB2GRAY);
			
			Mat LBPimage = new Mat(image.rows(),image.cols(),image.type());
			// New image to put zero around
			Mat testImage = new Mat(image.rows()+2,image.cols()+2,image.type());
			
			//Putting zero in 0th row of the testImage
			for(int i=0;i<1;i++)
			{
				for(int j=0;j<testImage.cols();j++)
				{
					testImage.put(i, j, 0);
				}
			}
			
			//Putting zero in the last row of the testImage
			for(int i=testImage.rows()-1;i<testImage.rows();i++)
			{
				for(int j=0;j<testImage.cols();j++)
				{
					testImage.put(i, j, 0);
				}
			}
			
			//Putting zero in 0th column of the testImage
			for(int i=0;i<testImage.rows();i++)
			{
				for(int j=0;j<1;j++)
				{
					testImage.put(i, j, 0);
				}
			}
			
			//Putting zero in the last row of the testImage
			for(int i=0;i<testImage.rows();i++)
			{
				for(int j=testImage.cols()-1;j<testImage.cols();j++)
				{
					testImage.put(i, j, 0);
				}
			}
			
			//putting rest of the values
			for(int i=1;i<testImage.rows()-1;i++)
			{
				for(int j=1;j<testImage.cols()-1;j++)
				{
					double[] value = image.get(i-1, j-1);
					
					testImage.put(i, j, value[0]);
				}
			}
			
			//System.out.println(testImage.dump());
			//Highgui.imwrite("NN.jpg", testImage);
			
			/*
			 * LBP implementation NN
			 */
			
			for(int i=1;i<testImage.rows()-1;i++)
			{
				
				for(int j=1;j<testImage.cols()-1;j++)
				{
					
					//System.out.println("matrix ["+(i)+"]["+(j)+"]");
					Mat slice = new Mat(testImage, new Range(i-1,i+2), new Range(j-1,j+2));
					
					LBPreturnValue value = createLBPClockwise(slice);
					
					
					//LBPimage.put(i, j, value.getDecimalValue());
					LBPimage.put(i, j, value.getDecimalValue());

					
				}
			}
			
		
			
//			Highgui.imwrite("halfImage.jpg", halfImage);
			
			
			histogram = calculateGreyHistogram(LBPimage);
			
			//System.out.println(histogram);
			
			
		}
		
	
		return histogram;
		
		
		
		
	}
	
	public String processForHistogramRandomPattern(Mat image)
	{
		String histogram ="";
		if(image.empty())
		{
			System.out.println("Empty");
			
		}
		else
		{
			Imgproc.cvtColor(image, image, Imgproc.COLOR_RGB2GRAY);
			Mat transformedImage = image.clone();
			
			//Imgproc.Canny(image, image, 300, 600, 5, true); 
			//Mat LBPimage = image.clone();
			Mat LBPimage = new Mat(image.rows(),image.cols(),image.type());
			
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
						transformedImage.put(i, j, 0);
						transformedImage.put(a, b, 255);
					}
					else
					{
						transformedImage.put(i, j, 255);
						transformedImage.put(a, b, 0);
					}
					
				}
			}
			
			
			/*
			 * LBP implimentation V2
			 */
//			for(int i=3;i<image.rows();i+=3)
//			{
//				
//				for(int j=3;j<image.cols();j+=3)
//				{
//					System.out.println("matrix ["+(i-3)+"]["+(j-3)+"]");
//					Mat slice = new Mat(image, new Range(i-3,i), new Range(j-3,j));
//					
//					LBPreturnValue value = createLBP(slice);
//					
//					for(int k=i-3,a=0;k<i;k++,a++)
//					{
//						for(int l=j-3,b=0;l<j;l++,b++)
//						{
//							
//							LBPimage.put(k, l, value.matrix[a][b]);
//						}
//					}
//					
//					LBPimage.put(((i-3)+i)/2,((j-3)+j)/2, value.getDecimalValue());
//
//					
//					
//				}
//			}
			
			/*
			 * Crossing 2 images
			 */
//			for(int i=0;i<image.rows();i++)
//			{
//				for(int j=0;j<image.cols();j++)
//				{
//					double value[] = image.get(i,j);
//					double value2[] = LBPimage.get(i, j);
//					
//					image.put(i, j,value[0]*value2[0]);
//				}
//			}
			//Highgui.imwrite("canny_lbp_kaido.jpg", image);
			
			/*
			 * LBP implementation V3
			 */
			
			
			Imgproc.Canny(transformedImage, transformedImage, 3, 6);
			//Highgui.imwrite("protein_RandomPattern_Canny.jpg", transformedImage);
			for(int i=1;i<image.rows()-1;i++)
			{
				
				for(int j=1;j<image.cols()-1;j++)
				{
					
					//System.out.println("matrix ["+(i)+"]["+(j)+"]");
					Mat slice = new Mat(transformedImage, new Range(i-1,i+2), new Range(j-1,j+2));
					
					LBPreturnValue value = createLBPClockwise(slice);
					
					
					LBPimage.put(i, j, value.getUniformValue());

					
				}
			}
			Imgcodecs.imwrite("protein_Uniform_LBP.jpg", LBPimage);
			histogram = calculateGreyHistogram(LBPimage);
			System.out.println(histogram);
			
			
		}
		
	
		return histogram;
		
		
	}
	
	public LBPreturnValue createLBP(Mat image)
	{
		//System.out.println(image.rows()+"\t"+image.cols());
		LBPreturnValue rv = new LBPreturnValue();
		double[] value= image.get(1, 1);
		double centerValue =value[0];
		String binary="";
		String[] uniform = {"00000000",
				"10000000","01000000","00100000","00010000","00001000","00000100","00000010","00000001",
				"11000000","01100000","00110000","00011000","00001100","00000110","00000011","10000001",
				"11100000","01110000","00111000","00011100","00001110","00000111","10000011","11000001",
				"11110000","01111000","00111100","00011110","00001111","10000111","11000011","11100001",
				"11111000","01111100","00111110","00011111","10001111","11000111","11100011","11110001",
				"11111100","01111110","00111111","10011111","11001111","11100111","11110011","11111001",
				"11111110","01111111","10111111","11011111","11101111","11110111","11111011","11111101",
				"11111111"};
		double[][] binaryPixelValues= new double[3][3];
		double pixelValue =0;
		/*
		 * for [1,2]
		 */
		value = image.get(1, 2);
		
		pixelValue = value[0];
		
		if(centerValue>pixelValue)
		{
			binary+="1";
			binaryPixelValues[1][2]=1;
		
		}
		else
		{
			binary+="0";
			binaryPixelValues[1][2]=0;
		
		}
		/*
		 * for [2][0] to [2][2]
		 */
		for(int i=2;i<3;i++)
		{
			for(int j=0;j<3;j++)
			{
				if(i==1 && j==1)
				{
					//this is the center value
				}
				else
				{
					value = image.get(i, j);
					
					pixelValue = value[0];
					
					if(centerValue>pixelValue)
					{
						binary+="1";
						binaryPixelValues[i][j]=1;
				
					}
					else
					{
						binary+="0";
						binaryPixelValues[i][j]=0;
				
					}
					
					
				}
			}
			
		}
		
		/* 
		 * for [1][2] to[0][2]
		 */
		for(int i=1;i>=0;i--)
		{
			for(int j=2;j<3;j++)
			{
				if(i==1 && j==1)
				{
					//this is the center value
				}
				else
				{
					value = image.get(i, j);
					
					pixelValue = value[0];
					
					if(centerValue>pixelValue)
					{
						binary+="1";
						binaryPixelValues[i][j]=1;
				
					}
					else
					{
						binary+="0";
						binaryPixelValues[i][j]=0;
					
					}
					
					
				}
			}
			
		}
		
		/*
		 * for [0][1] to [0][0]
		 */
		for(int i=0;i==0;i++)
		{
			for(int j=1;j>=0;j--)
			{
				if(i==1 && j==1)
				{
					//this is the center value
				}
				else
				{
					value = image.get(i, j);
					
					pixelValue = value[0];
					
					if(centerValue>pixelValue)
					{
						binary+="1";
						binaryPixelValues[i][j]=1;
						
					}
					else
					{
						binary+="0";
						binaryPixelValues[i][j]=0;
						
					}
					
					
				}
			}
			
		}
		
		//String reverseBinary = new StringBuilder(binary).reverse().toString();
		int decimalValue = Integer.parseInt(binary,2);
		int binaryValue = Integer.parseInt(binary);
//		int flag=0;
//		for(int i=0;i<58;i++)
//		{
//			if(binary.equals(uniform[i]))
//			{
//				rv.setUniformValue(i);
//				flag++;
//			}
//		}
//		if(flag==0)
//		{
//			rv.setUniformValue(58);
//		}
//		
		rv.setDecimalValue(decimalValue);
		rv.setMatrix(binaryPixelValues);
		return rv;
	}
	
	public LBPreturnValue createLBPClockwise(Mat image)
	{
		//System.out.println(image.rows()+"\t"+image.cols());
		LBPreturnValue rv = new LBPreturnValue();
		double[] value= image.get(1, 1);
		double centerValue =value[0];
		String binary="";
		String[] uniform = {"00000000",
				"10000000","01000000","00100000","00010000","00001000","00000100","00000010","00000001",
				"11000000","01100000","00110000","00011000","00001100","00000110","00000011","10000001",
				"11100000","01110000","00111000","00011100","00001110","00000111","10000011","11000001",
				"11110000","01111000","00111100","00011110","00001111","10000111","11000011","11100001",
				"11111000","01111100","00111110","00011111","10001111","11000111","11100011","11110001",
				"11111100","01111110","00111111","10011111","11001111","11100111","11110011","11111001",
				"11111110","01111111","10111111","11011111","11101111","11110111","11111011","11111101",
				"11111111"};
		double[][] binaryPixelValues= new double[3][3];
		double pixelValue =0;
		
		/*
		 * for[0,0] to [2,0]
		 */
		
		for(int i=0;i<3;i++)
		{
			for(int j=0;j<1;j++)
			{
				
				
					value = image.get(i, j);
					
					pixelValue = value[0];
					
					if(centerValue>pixelValue)
					{
						binary+="1";
						binaryPixelValues[i][j]=1;
				
					}
					else
					{
						binary+="0";
						binaryPixelValues[i][j]=0;
				
					}
			}
			
		}
		
		/*
		 * for[2,1] to [2,2]
		 */
		
		for(int i=2;i<3;i++)
		{
			for(int j=1;j<3;j++)
			{
				
				
					value = image.get(i, j);
					
					pixelValue = value[0];
					
					if(centerValue>pixelValue)
					{
						binary+="1";
						binaryPixelValues[i][j]=1;
				
					}
					else
					{
						binary+="0";
						binaryPixelValues[i][j]=0;
				
					}
			}
			
		}
		
		/*
		 * for [1,2] to [0,2]
		 */
		
		for(int i=1;i>=0;i--)
		{
			for(int j=2;j<3;j++)
			{
				
					value = image.get(i, j);
					
					pixelValue = value[0];
					
					if(centerValue>pixelValue)
					{
						binary+="1";
						binaryPixelValues[i][j]=1;
				
					}
					else
					{
						binary+="0";
						binaryPixelValues[i][j]=0;
					
					}
					
					
				
			}
			
		}
		
		
		/*
		 * for [0,1]
		 */
		value = image.get(0, 1);
		
		pixelValue = value[0];
		
		if(centerValue>pixelValue)
		{
			binary+="1";
			binaryPixelValues[1][2]=1;
		
		}
		else
		{
			binary+="0";
			binaryPixelValues[1][2]=0;
		
		}
		
		
		//String reverseBinary = new StringBuilder(binary).reverse().toString();
		int decimalValue = Integer.parseInt(binary,2);
		int binaryValue = Integer.parseInt(binary);
		int flag=0;
		for(int i=0;i<58;i++)
		{
			if(binary.equals(uniform[i]))
			{
				rv.setUniformValue(i);
				flag++;
			}
		}
		if(flag==0)
		{
			rv.setUniformValue(58);
		}
		
		rv.setDecimalValue(decimalValue);
		rv.setMatrix(binaryPixelValues);
		return rv;
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
	   // MatOfInt histSize = new MatOfInt(59);
	    
	    // only one channel
	    MatOfInt channels = new MatOfInt(0);
	    
	    //set of ranges
	    MatOfFloat histRange = new MatOfFloat(0,256);
	    
	    //set of ranges for uniform LBP
	    //MatOfFloat histRange = new MatOfFloat(0,59);
	    
	   
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
				
				histogram+=(int)(value[0]);
			}
			else
			{
				histogram+=(int)value[0]+"-";
			}
				
		}
	    
	    //for uniform LBP
//	    for(int i =0 ;i<59;i++)
//		{
//			double[] value = hist_b.get(i, 0);
//			
//			if(i==58)
//			{
//				histogram+=(int)value[0];
//			}
//			else
//			{
//				histogram+=(int)value[0]+"-";
//			}
//				
//		}
		return histogram;
		
	}
	
	
	public String calculateGreyHistogram_Uniform(Mat image)
	{
		 //splitting the frames in multiple images
	    java.util.List<Mat> images = new ArrayList<Mat>();
	    Core.split(image, images);
	    
		
	    
	    /*
	     * for uniform LBP , set the size to 59
	     */
	    MatOfInt histSize = new MatOfInt(59);
	    
	    // only one channel
	    MatOfInt channels = new MatOfInt(0);
	    
	  
	    
	    //set of ranges for uniform LBP
	    MatOfFloat histRange = new MatOfFloat(0,59);
	    
	   
	    //Compute the histogram s for B component(from BGR)
	    Mat hist_b=new Mat();
	    
	   //histogram for gray image
	    Imgproc.calcHist(images.subList(0, 1), channels, new Mat(), hist_b, histSize, histRange);
	    
	    String histogram="";
	    

	    
	    //for uniform LBP
	    for(int i =0 ;i<59;i++)
		{
			double[] value = hist_b.get(i, 0);
			
			if(i==58)
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
	
	public static String generateRowNames()
    {
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
//		 fold = sccsParts[1]; 
//		 superfamily = sccsParts[2];
//		 family = sccsParts[3];
	}
}
