package WaveltTest;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import wavelet.HaarWaveletTransform;

public class wavTest {
	public Mat image;
	public int rows;
	public int columns;
	public double[][] blueMatrix ;
	public double[][] greenMatrix;
	public double[][] redMatrix;
	
	
	
	public wavTest(Mat image)
	{
		this.image = image.clone();
		rows=image.rows();
		columns=image.cols();
		
	}
	
	public void createBlueMatrix()
	{
		blueMatrix = new double[rows][columns];
		for(int i=0;i<rows;i++)
		{
			for(int j=0;j<columns;j++)
			{
				double[] value=image.get(i, j);
				blueMatrix[i][j]=value[0];
			}
		}
	}
	
	public void createGreenMatrix()
	{
		greenMatrix = new double[rows][columns];
		for(int i=0;i<rows;i++)
		{
			for(int j=0;j<columns;j++)
			{
				double[] value=image.get(i, j);
				greenMatrix[i][j]=value[0];
			}
		}
	}
	
	public void createRedMatrix()
	{
		redMatrix = new double[rows][columns];
		for(int i=0;i<rows;i++)
		{
			for(int j=0;j<columns;j++)
			{
				double[] value=image.get(i, j);
				redMatrix[i][j]=value[0];
			}
		}
	}
	
	public Mat conversion()
	{
		createBlueMatrix();
		createGreenMatrix();
		createRedMatrix();
		
		HaarWaveletTransform hwtb = new HaarWaveletTransform();
		double[][] blueHaarMatrix=hwtb.doHaar2DFWTransform(blueMatrix, 2);
		HaarWaveletTransform hwtg = new HaarWaveletTransform();
		double[][] greenHaarMatrix=hwtg.doHaar2DFWTransform(greenMatrix, 2);
		HaarWaveletTransform hwtr = new HaarWaveletTransform();
		double[][] redHaarMatrix=hwtr.doHaar2DFWTransform(redMatrix, 2);
		
		Mat newImage = new Mat(new Size(rows,columns),CvType.CV_8UC3);
		
		for(int i=0;i<rows;i++)
		{
			for(int j=0;j<columns;j++)
			{
				double b=blueHaarMatrix[i][j];
				double g=greenHaarMatrix[i][j];
				double r=redHaarMatrix[i][j];
				newImage.put(i, j, new double[]{b,g,r});
			}
		}
		return newImage;
		
	}

}
