import java.io.File;
import java.io.PrintWriter;
import java.util.Random;


public class DataGenerator
{
	public static Random random=new Random();
	public static void main(String args[]) throws Exception
	{
		PrintWriter printer=new PrintWriter(new File("input.txt"));
		for(long i=0;i<10000000;i++)
		{
			double age=getRandom(0,100);
			if(age<20)
			{
				printer.println(getRandom(120,150)+" 4");
			}
			else if(age>=20 && age<40)
			{
				printer.println(getRandom(105,120)+" 3");
			}
			if(age>=40 && age<60)
			{
				printer.println(getRandom(90,105)+" 2");
			}
			if(age>=60)
			{
				printer.println(getRandom(70,90)+" 1");
			}
		}
		printer.flush();
		printer.close();
	}
	private static double getRandom(int min,int max)
	{
		return (double)(random.nextInt((max-min)+1)+min);
	}
}