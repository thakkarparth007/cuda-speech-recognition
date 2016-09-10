package speech;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class PreProcess
{
	public static void main(String args[]) throws Exception
	{
		File dir=new File("phonemes"+File.separator);
		File[] files=dir.listFiles();
		Scanner inputScanner;
		List<ArrayList<Integer>> values=new ArrayList<ArrayList<Integer>>();
		int[][] positive=new int[10][files.length];
		int[][] negative=new int[10][files.length];
		for(int i=0;i<files.length;i++)
		{
			values.add(new ArrayList<Integer>());
			inputScanner=new Scanner(files[i]);
			while(inputScanner.hasNext())
			{
				values.get(i).add(inputScanner.nextInt());
			}
			int j=0;
			int part=values.get(i).size()/10+5;
			for(int k=0;k<values.get(i).size();k++)
			{
				int val=values.get(i).get(k);
				if(val>0)
				{
					positive[j][i]++;
				}
				else
				{
					negative[j][i]++;
				}
				if(k%part==0 && k>0)
				{
					j++;
				}
			}
			inputScanner.close();
		}
		for(int i=0;i<files.length;i++)
		{
			PrintWriter printer=new PrintWriter(new File("preprocess_"+files[i].getName()));
			for(int j=0;j<10;j++)
			{
				printer.println(positive[j][i]+"\t"+negative[j][i]);
			}
			printer.flush();
			printer.close();
		}
	}
}