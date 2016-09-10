import java.io.File;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.Scanner;

public class TransitionProbability
{
	public static void main(String args[]) throws Exception
	{
		Scanner fileScanner=new Scanner(new File("dictionary.txt"));
		double[][] transitionProbablilty=new double[26][26];
		double[][] count=new double[26][26];
		double[] letterCount=new double[26];
		double totalCount=0.0;
		while(fileScanner.hasNext())
		{
			String word=fileScanner.nextLine().toLowerCase();
			letterCount[((int)word.charAt(0))-97]+=1.0;
			totalCount+=1.0;
			for(int i=0;i<word.length()-1;i++)
			{
				int curLetter=((int)word.charAt(i))-97;
				int nextLetter=((int)word.charAt(i+1))-97;
				if(curLetter>25 || curLetter<0 || nextLetter>25 || nextLetter<0)
				{
					continue;
				}
				//System.out.println(curLetter+" "+nextLetter);
				count[curLetter][nextLetter]+=1.0;
			}
		}
		PrintWriter out=new PrintWriter(new File("transition_probability.txt"));
		DecimalFormat format=new DecimalFormat("#.##########");
		for(int i=0;i<26;i++)
		{
			double c=0.0;
			for(int j=0;j<26;j++)
			{
				c+=count[i][j];
			}
			for(int j=0;j<26;j++)
			{
				transitionProbablilty[i][j]=(count[i][j]/c);
				out.print(format.format(transitionProbablilty[i][j])+"\t");
			}
			out.println();
		}
		out.flush();
		out.close();
		fileScanner.close();
		out=new PrintWriter(new File("initial_probability.txt"));
		for(int i=0;i<26;i++)
		{
			double prob=letterCount[i]/totalCount;
			out.println(format.format(prob));
		}
		out.flush();
		out.close();
	}
}