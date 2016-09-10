package speech;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;


public class Learn
{
	public static final int maxIterations=50000;
	public static final double bias=0.0;
	public static final double learningRate=0.05;
	public static final double minValue=0.0;
	public static final double maxValue=5000.0;
	public static final double learningRateInitial=0.01;
	public static DecimalFormat format=new DecimalFormat("#.######");
	public static Random random=new Random();
	public static int inputLayerLimit=10;
	public static int secondLayerLimit=20;
	public static int thirdLayerLimit=20;
	public static int fourthLayerLimit=44;
	public static int finalLayerLimit=1;
	public static ArrayList<Double> inputLayer=new ArrayList<Double>();
	public static ArrayList<Double> secondLayer=new ArrayList<Double>();
	public static ArrayList<Double> thirdLayer=new ArrayList<Double>();
	public static ArrayList<Double> fourthLayer=new ArrayList<Double>();
	//public static double finalOutput;
	public static Map<String,Integer> phonemeIDMapping=new HashMap<String,Integer>(); 
	public static double[][] inputConnection=new double[secondLayerLimit][inputLayerLimit];
	public static double[][] secondConnection=new double[thirdLayerLimit][secondLayerLimit];
	public static double[][] thirdConnection=new double[fourthLayerLimit][thirdLayerLimit];
	//public static double[] finalConnection=new double[fourthLayerLimit];
	static
	{
		phonemeIDMapping.put("a",1);
		phonemeIDMapping.put("a",2);
		phonemeIDMapping.put("e",3);
		phonemeIDMapping.put("i",4);
		phonemeIDMapping.put("o",5);
		phonemeIDMapping.put("u",6);
		phonemeIDMapping.put("ae",7);
		phonemeIDMapping.put("ee",8);
		phonemeIDMapping.put("ie",9);
		phonemeIDMapping.put("oe",10);
		phonemeIDMapping.put("ue",11);
		phonemeIDMapping.put("oo",12);
		phonemeIDMapping.put("ar",13);
		phonemeIDMapping.put("ur",14);
		phonemeIDMapping.put("or",15);
		phonemeIDMapping.put("au",16);
		phonemeIDMapping.put("er",17);
		phonemeIDMapping.put("ow",18);
		phonemeIDMapping.put("oi",19);
		phonemeIDMapping.put("b",20);
		phonemeIDMapping.put("d",21);
		phonemeIDMapping.put("f",22);
		phonemeIDMapping.put("g",23);
		phonemeIDMapping.put("h",24);
		phonemeIDMapping.put("j",25);
		phonemeIDMapping.put("k",26);
		phonemeIDMapping.put("l",27);
		phonemeIDMapping.put("m",28);
		phonemeIDMapping.put("n",29);
		phonemeIDMapping.put("p",30);
		phonemeIDMapping.put("r",31);
		phonemeIDMapping.put("s",32);
		phonemeIDMapping.put("t",33);
		phonemeIDMapping.put("v",34);
		phonemeIDMapping.put("w",35);
		phonemeIDMapping.put("wh",36);
		phonemeIDMapping.put("y",37);
		phonemeIDMapping.put("z",38);
		phonemeIDMapping.put("th",39);
		phonemeIDMapping.put("ch",40);
		phonemeIDMapping.put("sh",41);
		phonemeIDMapping.put("zh",42);
		phonemeIDMapping.put("ng",43);
	}
	private static double getRandomWeight()
	{
		Double val=random.nextDouble();
		//System.out.println(val);
		return val;
	}
	public static double normalize(double value)
	{
		double val=(value-minValue)/(maxValue-minValue);
		//System.out.println(val);
		return val;
	}
	public static void initFromFile()
	{
		int i=0,j=0;
		try
		{
			Scanner fileScanner=new Scanner(new File("inputConnection"));
			for(i=0;i<secondLayerLimit;i++)
			{
				for(j=0;j<inputLayerLimit;j++)
				{
					try
					{
						inputConnection[i][j]=fileScanner.nextDouble();
					}
					catch(Exception e)
					{
						//inputConnection[i][j]=getRandomWeight();
					}
				}
			}
			fileScanner.close();
			fileScanner=new Scanner(new File("secondConnection"));
			for(i=0;i<thirdLayerLimit;i++)
			{
				for(j=0;j<secondLayerLimit;j++)
				{
					try
					{
						secondConnection[i][j]=fileScanner.nextDouble();
					}
					catch(Exception e)
					{
						//secondConnection[i][j]=getRandomWeight();
					}
				}
			}
			fileScanner.close();
			fileScanner=new Scanner(new File("thirdConnection"));
			for(i=0;i<fourthLayerLimit;i++)
			{
				for(j=0;j<thirdLayerLimit;j++)
				{
					try
					{
						thirdConnection[i][j]=fileScanner.nextDouble();
					}
					catch(Exception e)
					{
						//thirdConnection[i][j]=getRandomWeight();
					}
				}
			}
			/*fileScanner=new Scanner(new File("finalConnection"));
			for(i=0;i<fourthLayerLimit;i++)
			{
				try
				{
					finalConnection[i]=fileScanner.nextDouble();
				}
				catch(Exception e)
				{
					finalConnection[i]=getRandomWeight();
				}
			}*/
			fileScanner.close();
		}
		catch(Exception e)
		{
			System.out.println(i+" "+j);
			e.printStackTrace();
		}
	}
	private static void init()
	{
		for(int i=0;i<secondLayerLimit;i++)
		{
			for(int j=0;j<inputLayerLimit;j++)
			{
				inputConnection[i][j]=getRandomWeight();
			}
		}
		for(int i=0;i<thirdLayerLimit;i++)
		{
			for(int j=0;j<secondLayerLimit;j++)
			{
				secondConnection[i][j]=getRandomWeight();
			}
		}
		for(int i=0;i<fourthLayerLimit;i++)
		{
			for(int j=0;j<thirdLayerLimit;j++)
			{
				thirdConnection[i][j]=getRandomWeight();
			}
		}
		/*for(int i=0;i<fourthLayerLimit;i++)
		{
			finalConnection[i]=getRandomWeight();
		}*/
	}
	public static double sigmoid(double value)
	{
		double val=1/(1+Math.exp(-value));
		return val;
	}
	public static double activation(double value,boolean derivative)
	{
		double val=derivative?value*(1-value):sigmoid(value);
		//System.out.println(format.format(val)+" "+format.format(value)+" "+derivative);
		return val;
	}
	public static void backPropagate(Double[] err,double error)
	{
		double[] delta=new double[thirdLayerLimit];
		for(int i=0;i<fourthLayerLimit;i++)
		{
			for(int j=0;j<thirdLayerLimit;j++)
			{
				double correction=(err[i]*activation(fourthLayer.get(i),true)*thirdLayer.get(j));
				thirdConnection[i][j]=thirdConnection[i][j]-learningRate*correction;
				delta[j]=delta[j]+(correction*thirdConnection[i][j]);
			}
		}
		double[] delta1=new double[secondLayerLimit];
		for(int i=0;i<thirdLayerLimit;i++)
		{
			for(int j=0;j<secondLayerLimit;j++)
			{
				double correction=(delta[i]*activation(thirdLayer.get(i),true)*secondLayer.get(j));
				secondConnection[i][j]=secondConnection[i][j]-learningRate*correction;
				delta1[j]=delta1[j]+(correction*secondConnection[i][j]);
			}
		}
		for(int i=0;i<secondLayerLimit;i++)
		{
			for(int j=0;j<inputLayerLimit;j++)
			{
				double correction=(delta1[i]*activation(secondLayer.get(i),true)*inputLayer.get(j));
				inputConnection[i][j]=inputConnection[i][j]-learningRate*correction;
			}
		}
	}
	public static void learn()
	{
		for(int i=0;i<secondLayerLimit;i++)
		{
			double sum=bias;
			for(int j=0;j<inputLayerLimit;j++)
			{
				sum+=(inputLayer.get(j)*inputConnection[i][j]);
			}
			secondLayer.add(activation(sum,false));
		}
		for(int i=0;i<thirdLayerLimit;i++)
		{
			double sum=bias;
			for(int j=0;j<secondLayerLimit;j++)
			{
				sum+=(secondLayer.get(j)*secondConnection[i][j]);
			}
			thirdLayer.add(activation(sum,false));
		}
		for(int i=0;i<fourthLayerLimit;i++)
		{
			double sum=bias;
			for(int j=0;j<thirdLayerLimit;j++)
			{
				sum+=(thirdLayer.get(j)*thirdConnection[i][j]);
			}
			fourthLayer.add(activation(sum,false));
		}
		/*double sum=bias;
		for(int i=0;i<fourthLayerLimit;i++)
		{
			sum+=activation(fourthLayer.get(i)*finalConnection[i],false);
		}
		finalOutput=Math.abs(sum%44.0);*/
	}
	public static void learner()
	{
		try
		{
			for(int k=0;k<maxIterations;k++)
			{
				//System.out.println("Iteration: "+(k+1)+"...");
				File phonemeDirectory=new File("preprocess"+File.separator);
				File[] fileList=phonemeDirectory.listFiles();
				for(int i=0;i<fileList.length;i++)
				{
					Scanner fileScanner=new Scanner(fileList[i]);
					String phoneme=fileList[i].getName().split("_")[1];
					int expectedOutput=phonemeIDMapping.get(phoneme);
					while(fileScanner.hasNext())
					{
						Double positive=fileScanner.nextDouble();
						Double negative=fileScanner.nextDouble();
						Double val=positive>negative?positive:-negative;
						inputLayer.add(normalize(val));
					}
					learn();
					Double[] err=new Double[fourthLayerLimit];
					double error=0.0;
					for(int m=0;m<fourthLayerLimit;m++)
					{
						err[m]=(m+1)==expectedOutput?fourthLayer.get(m)-0.99:fourthLayer.get(m)-0.01;
						double s=(m+1)==expectedOutput?0.99:0.01;
						error+=(0.5*(s-fourthLayer.get(m))*(s-fourthLayer.get(m)));
					}
					if((k+1)%10000==0)
					{
						System.out.print("Error: Iteration "+(k+1)+" Phoneme "+phoneme+" [");
						DecimalFormat formatter=new DecimalFormat("#.####");
						for(int m=0;m<err.length;m++)
						{
							System.out.print(formatter.format(err[m])+" ");
						}
						System.out.print("]\n");
					}
					backPropagate(err,error);
					inputLayer=new ArrayList<Double>();
					secondLayer=new ArrayList<Double>();
					thirdLayer=new ArrayList<Double>();
					fourthLayer=new ArrayList<Double>();
					fileScanner.close();
				}
			}
		}
		catch(FileNotFoundException e)
		{
			e.printStackTrace();
		}
	}
	private static void commitToFile()
	{
		try
		{
			PrintWriter filePrinter=new PrintWriter(new File("inputConnection"));
			DecimalFormat doubleFormat=new DecimalFormat("#.######");
			for(int i=0;i<secondLayerLimit;i++)
			{
				filePrinter.print(doubleFormat.format(inputConnection[i][0]));
				for(int j=1;j<inputLayerLimit;j++)
				{
					filePrinter.print("\t"+doubleFormat.format(inputConnection[i][j]));
				}
				filePrinter.println();
				filePrinter.flush();
				filePrinter.close();
			}
			filePrinter=new PrintWriter(new File("secondConnection"));
			for(int i=0;i<thirdLayerLimit;i++)
			{
				filePrinter.print(doubleFormat.format(secondConnection[i][0]));
				for(int j=1;j<secondLayerLimit;j++)
				{
					filePrinter.print("\t"+doubleFormat.format(secondConnection[i][j]));
				}
				filePrinter.println();
				filePrinter.flush();
				filePrinter.close();
			}
			filePrinter=new PrintWriter(new File("thirdConnection"));
			for(int i=1;i<fourthLayerLimit;i++)
			{
				filePrinter.print(doubleFormat.format(thirdConnection[i][0]));
				for(int j=0;j<thirdLayerLimit;j++)
				{
					filePrinter.print("\t"+doubleFormat.format(thirdConnection[i][j]));
				}
				filePrinter.println();
				filePrinter.flush();
				filePrinter.close();
			}
			/*filePrinter=new PrintWriter(new File("finalConnection"));
			for(int i=0;i<fourthLayerLimit;i++)
			{
				filePrinter.println(doubleFormat.format(finalConnection[i]));
			}*/
			filePrinter.close();
			filePrinter.flush();
			filePrinter.close();
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}
	public static void main(String args[]) throws Exception
	{
		Scanner inputScanner=new Scanner(System.in);
		System.out.print("1. Learn\n2. Predict\nPlease enter choice:");
		String option=inputScanner.nextLine();
		if(option.equals("1"))
		{
			init();
			System.out.println("Learning phase started...");
			learner();
			commitToFile();
			System.out.println("Learning phase finished...");
		}
		else
		{
			initFromFile();
		}
		System.out.println("Prediction phase started...");
		Predict.initPrediction("test");
		System.out.println("Prediction phase finished...");
		inputScanner.close();
	}
}