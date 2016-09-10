
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include<stdio.h>
#include<stdlib.h>
#include<string>
#include<vector>
#include<iostream>
#include<time.h>
#include<map>
#include<cmath>
#include<limits>
#include<fstream>

//#include<cuda.h>

using namespace std;

//static const int WORK_SIZE = 256;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * See cuda.h for error code descriptions.
 */
#define CHECK_CUDA_RESULT(N) {											\
	CUresult result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n"]= __LINE__,	\
			result);													\
		exit(1);														\
	} }

/*int main(int argc, char **argv)
{
	CUmodule module;
	CUcontext context;
	CUdevice device;
	CUdeviceptr deviceArray;
	CUfunction process;

	void *kernelArguments[] = { &deviceArray };
	int deviceCount;
	unsigned int idata[WORK_SIZE], odata[WORK_SIZE];

	for (int i = 0; i < WORK_SIZE; ++i) {
		idata[i] = i;
	}

	CHECK_CUDA_RESULT(cuInit(0));
	CHECK_CUDA_RESULT(cuDeviceGetCount(&deviceCount));
	if (deviceCount == 0) {
		printf("No CUDA-compatible devices found\n");
		exit(1);
	}
	CHECK_CUDA_RESULT(cuDeviceGet(&device, 0));
	CHECK_CUDA_RESULT(cuCtxCreate(&context, 0, device));

	CHECK_CUDA_RESULT(cuModuleLoad(&module, "bitreverse.fatbin"));
	CHECK_CUDA_RESULT(cuModuleGetFunction(&process, module, "bitreverse"));

	CHECK_CUDA_RESULT(cuMemAlloc(&deviceArray, sizeof(int) * WORK_SIZE));
	CHECK_CUDA_RESULT(
			cuMemcpyHtoD(deviceArray, idata, sizeof(int) * WORK_SIZE));

	CHECK_CUDA_RESULT(
			cuLaunchKernel(process, 1, 1, 1, WORK_SIZE, 1, 1, 0, NULL, kernelArguments, NULL));

	CHECK_CUDA_RESULT(
			cuMemcpyDtoH(odata, deviceArray, sizeof(int) * WORK_SIZE));

	for (int i = 0; i < WORK_SIZE; ++i) {
		printf("Input value: %u, output value: %u\n"]= idata[i], odata[i]);
	}

	CHECK_CUDA_RESULT(cuMemFree(deviceArray));
	CHECK_CUDA_RESULT(cuCtxDestroy(context));

	return 0;
}*/
int maxIterations=50000;
double bias=0.0;
double learningRate=0.05;
double minValue=0.0;
double maxValue=5000.0;
double learningRateInitial=0.01;
double integer_maximum=(double)numeric_limits<int>::max();
double integer_minimum=(double)numeric_limits<int>::min();
const int inputLayerLimit=10;
const int secondLayerLimit=20;
const int thirdLayerLimit=20;
const int fourthLayerLimit=44;
vector<double> inputLayer;
vector<double> secondLayer;
vector<double> thirdLayer;
vector<double> fourthLayer;
//double finalOutput;
map<string,int> phonemeIDMapping;
double inputConnection[secondLayerLimit][inputLayerLimit];
double secondConnection[thirdLayerLimit][secondLayerLimit];
double thirdConnection[fourthLayerLimit][thirdLayerLimit];
string phonemes[41];
//double[] finalConnection=new double[fourthLayerLimit];
void initLearningData()
{
	phonemes[0]="a";
	phonemes[1]="ae";
	phonemes[2]="ar";
	phonemes[3]="au";
	phonemes[4]="b";
	phonemes[5]="ch";
	phonemes[6]="d";
	phonemes[7]="e";
	phonemes[8]="ee";
	phonemes[9]="er";
	phonemes[10]="f";
	phonemes[11]="g";
	phonemes[12]="h";
	phonemes[13]="ie";
	phonemes[14]="j";
	phonemes[15]="k";
	phonemes[16]="l";
	phonemes[17]="m";
	phonemes[18]="n";
	phonemes[19]="ng";
	phonemes[20]="o";
	phonemes[21]="oe";
	phonemes[22]="oi";
	phonemes[23]="oo";
	phonemes[24]="or";
	phonemes[25]="ow";
	phonemes[26]="p";
	phonemes[27]="r";
	phonemes[28]="s";
	phonemes[29]="sh";
	phonemes[30]="t";
	phonemes[31]="th";
	phonemes[32]="u";
	phonemes[33]="ue";
	phonemes[34]="ur";
	phonemes[35]="v";
	phonemes[36]="w";
	phonemes[37]="wh";
	phonemes[38]="y";
	phonemes[39]="z";
	phonemes[40]="zh";
	for(int i=0;i<41;i++)
	{
		phonemeIDMapping[phonemes[i]]=(i+1);
	}
	/*phonemeIDMapping["a"]=1;
	phonemeIDMapping["a"]=2;
	phonemeIDMapping["e"]=3;
	phonemeIDMapping["i"]=4;
	phonemeIDMapping["o"]=5;
	phonemeIDMapping["u"]=6;
	phonemeIDMapping["ae"]=7;
	phonemeIDMapping["ee"]=8;
	phonemeIDMapping["ie"]=9;
	phonemeIDMapping["oe"]=10;
	phonemeIDMapping["ue"]=11;
	phonemeIDMapping["oo"]=12;
	phonemeIDMapping["ar"]=13;
	phonemeIDMapping["ur"]=14;
	phonemeIDMapping["or"]=15;
	phonemeIDMapping["au"]=16;
	phonemeIDMapping["er"]=17;
	phonemeIDMapping["ow"]=18;
	phonemeIDMapping["oi"]=19;
	phonemeIDMapping["b"]=20;
	phonemeIDMapping["d"]=21;
	phonemeIDMapping["f"]=22;
	phonemeIDMapping["g"]=23;
	phonemeIDMapping["h"]=24;
	phonemeIDMapping["j"]=25;
	phonemeIDMapping["k"]=26;
	phonemeIDMapping["l"]=27;
	phonemeIDMapping["m"]=28;
	phonemeIDMapping["n"]=29;
	phonemeIDMapping["p"]=30;
	phonemeIDMapping["r"]=31;
	phonemeIDMapping["s"]=32;
	phonemeIDMapping["t"]=33;
	phonemeIDMapping["v"]=34;
	phonemeIDMapping["w"]=35;
	phonemeIDMapping["wh"]=36;
	phonemeIDMapping["y"]=37;
	phonemeIDMapping["z"]=38;
	phonemeIDMapping["th"]=39;
	phonemeIDMapping["ch"]=40;
	phonemeIDMapping["sh"]=41;
	phonemeIDMapping["zh"]=42;
	phonemeIDMapping["ng"]=43;*/
}
double getRandomWeight()
{
	double val=(double)rand();
	val=(val-integer_minimum)/(integer_maximum-integer_minimum);
	//cout<<val<<endl;
	return val;
}
double normalize(double value)
{
	double val=(value-minValue)/(maxValue-minValue);
	//cout<<val<<endl;
	return val;
}
/*void initFromFile()
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
					inputConnection[i][j]=fileScanner.nextdouble();
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
					secondConnection[i][j]=fileScanner.nextdouble();
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
					thirdConnection[i][j]=fileScanner.nextdouble();
				}
				catch(Exception e)
				{
					//thirdConnection[i][j]=getRandomWeight();
				}
			}
		}
		fileScanner=new Scanner(new File("finalConnection"));
		for(i=0;i<fourthLayerLimit;i++)
		{
			try
			{
				finalConnection[i]=fileScanner.nextdouble();
			}
			catch(Exception e)
			{
				finalConnection[i]=getRandomWeight();
			}
		}
		fileScanner.close();
	}
	catch(Exception e)
	{
		cout<<i+" "+j);
		e.printStackTrace();
	}
}*/
void init()
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
double sigmoid(double value)
{
	double val=1/(1+exp(-value));
	//cout<<val<<endl;
	return val;
}
double activation(double value,int derivative)
{
	double val=derivative==1?value*(1-value):sigmoid(value);
	//cout<<val<<endl;
	return val;
}
void backPropagate(double err[],double error)
{
	double delta[thirdLayerLimit];
	for(int i=0;i<fourthLayerLimit;i++)
	{
		for(int j=0;j<thirdLayerLimit;j++)
		{
			double correction=(err[i]*activation(fourthLayer[i],1)*thirdLayer[j]);
			thirdConnection[i][j]=thirdConnection[i][j]-learningRate*correction;
			delta[j]=delta[j]+(correction*thirdConnection[i][j]);
		}
	}
	double delta1[secondLayerLimit];
	for(int i=0;i<thirdLayerLimit;i++)
	{
		for(int j=0;j<secondLayerLimit;j++)
		{
			double correction=(delta[i]*activation(thirdLayer[i],1)*secondLayer[j]);
			secondConnection[i][j]=secondConnection[i][j]-learningRate*correction;
			delta1[j]=delta1[j]+(correction*secondConnection[i][j]);
		}
	}
	for(int i=0;i<secondLayerLimit;i++)
	{
		for(int j=0;j<inputLayerLimit;j++)
		{
			double correction=(delta1[i]*activation(secondLayer[i],1)*inputLayer[j]);
			inputConnection[i][j]=inputConnection[i][j]-learningRate*correction;
		}
	}
}
void learn()
{
	for(int i=0;i<secondLayerLimit;i++)
	{
		double sum=bias;
		for(int j=0;j<inputLayerLimit;j++)
		{
			sum+=(inputLayer[j]*inputConnection[i][j]);
		}
		secondLayer.push_back(activation(sum,0));
	}
	for(int i=0;i<thirdLayerLimit;i++)
	{
		double sum=bias;
		for(int j=0;j<secondLayerLimit;j++)
		{
			sum+=(secondLayer[j]*secondConnection[i][j]);
		}
		thirdLayer.push_back(activation(sum,0));
	}
	for(int i=0;i<fourthLayerLimit;i++)
	{
		double sum=bias;
		for(int j=0;j<thirdLayerLimit;j++)
		{
			sum+=(thirdLayer[j]*thirdConnection[i][j]);
		}
		fourthLayer.push_back(activation(sum,0));
	}
	/*double sum=bias;
	for(int i=0;i<fourthLayerLimit;i++)
	{
		sum+=activation(fourthLayer[i)*finalConnection[i],0);
	}
	finalOutput=Math.abs(sum%44.0);*/
}
void learner()
{
	for(int k=0;k<maxIterations;k++)
	{
		//cout<<"Iteration: "+(k+1)+"..."<<endl;
		for(int i=0;i<41;i++)
		{
			string filename="preprocess/preprocess_"+phonemes[i]+"_1";
			//cout<<filename;
			ifstream file(filename.c_str());
			string phoneme=phonemes[i];
			int expectedOutput=phonemeIDMapping[phoneme];
			while(!file.eof())
			{
				double positive,negative;
				file>>positive>>negative;
				double val=positive>negative?positive:-negative;
				//cout<<val;
				inputLayer.push_back(normalize(val));
			}
			learn();
			double err[fourthLayerLimit];
			double error=0.0;
			for(int m=0;m<fourthLayerLimit;m++)
			{
				err[m]=(m+1)==expectedOutput?fourthLayer[m]-0.99:fourthLayer[m]-0.01;
				double s=(m+1)==expectedOutput?0.99:0.01;
				error+=(0.5*(s-fourthLayer[m])*(s-fourthLayer[m]));
			}
			if((k+1)%10000==0)
			{
				cout<<"Error: Iteration "<<(k+1)<<" Phoneme "<<phoneme<<" [";
				for(int m=0;m<fourthLayerLimit;m++)
				{
					cout<<err[m]<<" ";
				}
				cout<<"]"<<endl;
			}
			backPropagate(err,error);
			inputLayer.clear();
			secondLayer.clear();
			thirdLayer.clear();
			fourthLayer.clear();
			file.close();
		}
	}
}
void commitToFile()
{
	/*try
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
		filePrinter=new PrintWriter(new File("finalConnection"));
		for(int i=0;i<fourthLayerLimit;i++)
		{
			filePrinter.println(doubleFormat.format(finalConnection[i]));
		}
		filePrinter.close();
		filePrinter.flush();
		filePrinter.close();
	}
	catch(Exception e)
	{
		e.printStackTrace();
	}*/
}
void predict()
{
	for(int i=0;i<secondLayerLimit;i++)
	{
		double sum=bias;
		for(int j=0;j<inputLayerLimit;j++)
		{
			sum+=(inputLayer[j]*inputConnection[i][j]);
		}
		secondLayer.push_back(activation(sum,0));
	}
	for(int i=0;i<thirdLayerLimit;i++)
	{
		double sum=bias;
		for(int j=0;j<secondLayerLimit;j++)
		{
			sum+=(secondLayer[j]*secondConnection[i][j]);
		}
		thirdLayer.push_back(activation(sum,0));
	}
	for(int i=0;i<fourthLayerLimit;i++)
	{
		double sum=bias;
		for(int j=0;j<thirdLayerLimit;j++)
		{
			sum+=(thirdLayer[j]*thirdConnection[i][j]);
		}
		fourthLayer.push_back(activation(sum,0));
	}
	for(map<string,int>::iterator key=phonemeIDMapping.begin();key!=phonemeIDMapping.end();++key)
	{
		cout<<"Phoneme: "<<key->first<<" Probability: "<<fourthLayer[key->second]<<endl;
	}
}
void initPrediction(string filename)
{
	ifstream file(filename.c_str());
	while(!file.eof())
	{
		double positive,negative;
		file>>positive>>negative;
		double val=positive>negative?positive:-negative;
		inputLayer.push_back(normalize(val));
	}
	predict();
}
int main()
{
	srand(time(NULL));
	cout<<"1. Learn\n2. Predict\nPlease enter choice:";
	int option;
	cin>>option;
	if(option==1)
	{
		initLearningData();
		init();
		cout<<"Learning phase started..."<<endl;
		learner();
		commitToFile();
		cout<<"Learning phase finished..."<<endl;
	}
	else
	{
		//initFromFile();
	}
	cout<<"Prediction phase started..."<<endl;
	initPrediction("test");
	cout<<"Prediction phase finished..."<<endl;
	return 0;
}
