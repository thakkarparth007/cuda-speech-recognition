package speech;

import java.io.File;
import java.util.ArrayList;
import java.util.Scanner;

public class Predict
{
	public static void initPrediction(String filename)
	{
		try
		{
			Learn.inputLayer=new ArrayList<Double>();
			Scanner fileScanner=new Scanner(new File(filename));
			while(fileScanner.hasNext())
			{
				Double positive=fileScanner.nextDouble();
				Double negative=fileScanner.nextDouble();
				Double val=positive>negative?positive:-negative;
				Learn.inputLayer.add(Learn.normalize(val));
			}
			fileScanner.close();
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
		predict();
	}
	public static void predict()
	{
		for(int i=0;i<Learn.secondLayerLimit;i++)
		{
			double sum=Learn.bias;
			for(int j=0;j<Learn.inputLayerLimit;j++)
			{
				sum+=(Learn.inputLayer.get(j)*Learn.inputConnection[i][j]);
			}
			Learn.secondLayer.add(Learn.activation(sum,false));
		}
		for(int i=0;i<Learn.thirdLayerLimit;i++)
		{
			double sum=Learn.bias;
			for(int j=0;j<Learn.secondLayerLimit;j++)
			{
				sum+=(Learn.secondLayer.get(j)*Learn.secondConnection[i][j]);
			}
			Learn.thirdLayer.add(Learn.activation(sum,false));
		}
		for(int i=0;i<Learn.fourthLayerLimit;i++)
		{
			double sum=Learn.bias;
			for(int j=0;j<Learn.thirdLayerLimit;j++)
			{
				sum+=(Learn.thirdLayer.get(j)*Learn.thirdConnection[i][j]);
			}
			Learn.fourthLayer.add(Learn.activation(sum,false));
		}
		for(String key: Learn.phonemeIDMapping.keySet())
		{
			int index=Learn.phonemeIDMapping.get(key);
			System.out.println("Phoneme: "+key+" Probability: "+Learn.format.format(Learn.fourthLayer.get(index)));
		}
	}
}