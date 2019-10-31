package localBinaryPattern;

public class LBPreturnValue {


	public double[][] matrix ;
	public int decimalValue;
	public int uniformValue;
	
	public void initialize()
	{
		for(int i=0;i<3;i++)
		{
			for(int j=0;j<3;j++)
			{
				matrix[i][j]=0;
			}
		}
		
		decimalValue=0;
	}
	public double[][] getMatrix() {
		return matrix;
	}

	public void setMatrix(double[][] matrix) {
		this.matrix = matrix;
	}

	public int getDecimalValue() {
		return decimalValue;
	}

	public void setDecimalValue(int decimalValue) {
		this.decimalValue = decimalValue;
	}
	
	public int getUniformValue() {
		return uniformValue;
	}
	public void setUniformValue(int uniformValue) {
		this.uniformValue = uniformValue;
	}
	
	
}
