package filter;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class GaborFilter {
	public Mat gabor(Mat image)
	{
		int kernel_size = 3;
        double sig = 5, th = 0, lm = 10, gm = 0.02, ps = 0;
       // System.out.println(image.type());
        Mat mat1 = new Mat(image.rows(), image.cols(), CvType.CV_8UC1);
     // Making the source black and white.
      //  Imgproc.cvtColor(image, mat1, Imgproc.COLOR_RGB2GRAY);
      //  System.out.println(image.type());
		
		Size sz = new Size(kernel_size,kernel_size);

         // Applying the Gabor filter.
        Mat kernel = Imgproc.getGaborKernel(sz, sig, th, lm, gm);
        Mat dest = new Mat(mat1.rows(), mat1.cols(), image.type());
        Imgproc.filter2D(mat1, dest, mat1.type(), kernel);
        // Save the result.
        //String filename = "gaborFiltered.png";
        //Highgui.imwrite(filename, dest);
        return dest;
        
	}
}
