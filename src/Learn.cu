//Native Libraries
#include<stdio.h>
#include<stdlib.h>
#include<string>
#include<vector>
#include<iostream>
#include<time.h>
#include<map>
#include<limits>
#include<fstream>
#include<iomanip>
#include<ctime>

//CUDA Libraries
#include<cuda.h>
#include<math_functions.h>

//Size Macros
#define inputLimit 10
#define secondLimit 20
#define thirdLimit 20
#define fourthLimit 41

//Error Checking
#define checkError(ans) {gpuAssert((ans),__FILE__,__LINE__);}

using namespace std;

//Global Variables
int maxIterations=50000;
__constant__ double bias=0.0;
__constant__ double learningRate=0.5;
double minValue=0.0;
double maxValue=5000.0;
double integer_maximum=(double)numeric_limits<int>::max();
double integer_minimum=(double)numeric_limits<int>::min();
int inputCS=inputLimit*secondLimit*sizeof(double),inputLS=inputLimit*sizeof(double);
int secondCS=thirdLimit*secondLimit*sizeof(double),secondLS=secondLimit*sizeof(double);
int thirdCS=thirdLimit*fourthLimit*sizeof(double),thirdLS=thirdLimit*sizeof(double);
int fourthLS=fourthLimit*sizeof(double);
double *inputLayerD,*secondLayerD,*thirdLayerD,*fourthLayerD;
double *inputConnectionD;
double *secondConnectionD;
double *thirdConnectionD;
double inputLayer[inputLimit];
double secondLayer[secondLimit];
double thirdLayer[thirdLimit];
double fourthLayer[fourthLimit];
map<string,int> phonemeIDMapping;
double inputConnection[secondLimit*inputLimit];
double secondConnection[thirdLimit*secondLimit];
double thirdConnection[fourthLimit*thirdLimit];
string phonemes[41];
double timeTaken=0;

inline void gpuAssert(cudaError_t code,const char *file,int line,bool abort=true)
{
	if(abort && code!=cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d Code: %d\n",cudaGetErrorString(code),file,line,code);
		exit(code);
	}
}

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
}
double getRandomWeight()
{
	double val=(double)rand();
	val=(val-integer_minimum)/(integer_maximum-integer_minimum);
	return val;
}
double normalize(double value)
{
	double val=(value-minValue)/(maxValue-minValue);
	return val;
}
void init()
{
	for(int i=0;i<secondLimit;i++)
	{
		for(int j=0;j<inputLimit;j++)
		{
			int index=i*inputLimit+j;
			inputConnection[index]=getRandomWeight();
		}
	}
	for(int i=0;i<thirdLimit;i++)
	{
		for(int j=0;j<secondLimit;j++)
		{
			int index=i*secondLimit+j;
			secondConnection[index]=getRandomWeight();
		}
	}
	for(int i=0;i<fourthLimit;i++)
	{
		for(int j=0;j<thirdLimit;j++)
		{
			int index=i*thirdLimit+j;
			thirdConnection[index]=getRandomWeight();
		}
	}
	checkError(cudaMalloc((void**)&inputConnectionD,inputCS));
	checkError(cudaMemcpy(inputConnectionD,inputConnection,inputCS,cudaMemcpyHostToDevice));
	checkError(cudaMalloc((void**)&secondConnectionD,secondCS));
	checkError(cudaMemcpy(secondConnectionD,secondConnection,secondCS,cudaMemcpyHostToDevice));
	checkError(cudaMalloc((void**)&thirdConnectionD,thirdCS));
	checkError(cudaMemcpy(thirdConnectionD,thirdConnection,thirdCS,cudaMemcpyHostToDevice));
	checkError(cudaMalloc((void**)&inputLayerD,inputLS));
	checkError(cudaMemcpy(inputLayerD,inputLayer,inputLS,cudaMemcpyHostToDevice));
	checkError(cudaMalloc((void**)&secondLayerD,secondLS));
	checkError(cudaMemcpy(secondLayerD,secondLayer,secondLS,cudaMemcpyHostToDevice));
	checkError(cudaMalloc((void**)&thirdLayerD,thirdLS));
	checkError(cudaMemcpy(thirdLayerD,thirdLayer,thirdLS,cudaMemcpyHostToDevice));
	checkError(cudaMalloc((void**)&fourthLayerD,fourthLS));
	checkError(cudaMemcpy(fourthLayerD,fourthLayer,fourthLS,cudaMemcpyHostToDevice));
}
__device__ double sigmoid(double value)
{
	double val=1/(1+exp(-value));
	return val;
}
__device__ double activation(double value,int derivative)
{
	double val=derivative==1?value*(1-value):sigmoid(value);
	if(isnan(val))
	{
		val=0.0;
	}
	return val;
}
__global__ void backPropagate4to3(double *err,double *thirdC,double *fourth,double *third,double *delta)
{
	int i = blockIdx.x;
	int j = blockIdx.y;
	int index = i*thirdLimit + j;
	double correction=(err[i]*activation(fourth[i],1)*third[j]);
	thirdC[index]=thirdC[index]-learningRate*correction;
	delta[j]=delta[j]+(correction*thirdC[index]);
}
__global__ void backPropagate3to2(double *secondC,double *third,double *second,double *delta,double *delta1)
{
	int i=blockIdx.x;
	int j = blockIdx.y;
	int index=i*secondLimit+j;
	double correction=(delta[i]*activation(third[i],1)*second[j]);
	secondC[index]=secondC[index]-learningRate*correction;
	delta1[j]=delta1[j]+(correction*secondC[index]);
}
__global__ void backPropagate2to1(double *inputC,double *second,double *input,double *delta)
{
	int i=blockIdx.x;
	int j = blockIdx.y;
	int index=i*inputLimit+j;
	double correction=(delta[i]*activation(second[i],1)*input[j]);
	inputC[index]=inputC[index]-learningRate*correction;
}
void backPropagate(double *err)
{
	double *deltaD,*delta1D,*errD;
	checkError(cudaMalloc((void**)&errD,fourthLS));
	checkError(cudaMemcpy(errD,err,fourthLS,cudaMemcpyHostToDevice));
	checkError(cudaMalloc((void**)&deltaD,thirdLS));
	checkError(cudaMalloc((void**)&delta1D,secondLS));
	checkError(cudaMemset(deltaD,0,thirdLS));
	checkError(cudaMemset(delta1D,0,secondLS));
	time_t start=time(NULL);
	
	dim3 lim4to3(fourthLimit, thirdLimit);
	dim3 lim3to2(thirdLimit, secondLimit);
	dim3 lim2to1(secondLimit,inputLimit);
	backPropagate4to3<<<lim4to3,1>>>(errD,thirdConnectionD,fourthLayerD,thirdLayerD,deltaD);
	checkError(cudaThreadSynchronize());
	backPropagate3to2<<<lim3to2,1>>>(secondConnectionD,thirdLayerD,secondLayerD,deltaD,delta1D);
	checkError(cudaThreadSynchronize());
	backPropagate2to1<<<lim2to1,1>>>(inputConnectionD,secondLayerD,inputLayerD,delta1D);
	checkError(cudaThreadSynchronize());
	time_t end=time(NULL);
	timeTaken+=abs(difftime(start,end));
	cudaFree(errD);
	cudaFree(deltaD);
	cudaFree(delta1D);
}
__global__ void learn1to2(double *input,double *inputC,double *second)
{
	int i=blockIdx.x;
	double sum=0.0;
	for(int j=0;j<inputLimit;j++)
	{
		int index=i*inputLimit+j;
		sum+=(input[j]*inputC[index]);
	}
	second[i]=activation(sum,0);
}
__global__ void learn2to3(double *second,double *secondC,double *third)
{
	int i=blockIdx.x;
	double sum=0.0;
	for(int j=0;j<secondLimit;j++)
	{
		int index=i*secondLimit+j;
		sum+=(second[j]*secondC[index]);
	}
	third[i]=activation(sum,0);
}
__global__ void learn3to4(double *third,double *thirdC,double *fourth)
{
	int i=blockIdx.x;
	double sum=0.0;
	for(int j=0;j<thirdLimit;j++)
	{
		int index=i*thirdLimit+j;
		sum+=(third[j]*thirdC[index]);
	}
	fourth[i]=activation(sum,0);
}
void copyBackData()
{
	checkError(cudaMemcpy(inputConnection,inputConnectionD,inputCS,cudaMemcpyDeviceToHost));
	checkError(cudaMemcpy(secondConnection,secondConnectionD,secondCS,cudaMemcpyDeviceToHost));
	checkError(cudaMemcpy(thirdConnection,thirdConnectionD,thirdCS,cudaMemcpyDeviceToHost));
}
void printArray(double *arr,int size)
{
	cout<<"[ ";
	for(int i=0;i<size;i++)
	{
		cout<<arr[i]<<" ";
	}
	cout<<"]"<<endl;
}
void learner()
{
	for(int k=0;k<maxIterations;k++)
	{
		for(int i=0;i<41;i++)
		{
			string filename="Sound/preprocess/preprocess_"+phonemes[i]+"_1";
			ifstream file(filename.c_str());
			string phoneme=phonemes[i];
			int expectedOutput=phonemeIDMapping[phoneme];
			int j=0;
			while(j<inputLimit)
			{
				double positive,negative;
				file>>positive>>negative;
				double val=positive>negative?positive:-negative;
				inputLayer[j]=normalize(val);
				j++;
			}
			checkError(cudaMemcpy(inputLayerD,inputLayer,inputLS,cudaMemcpyHostToDevice));
			time_t start=time(NULL);
			learn1to2<<<secondLimit,1>>>(inputLayerD,inputConnectionD,secondLayerD);
			learn2to3<<<thirdLimit,1>>>(secondLayerD,secondConnectionD,thirdLayerD);
			learn3to4<<<fourthLimit,1>>>(thirdLayerD,thirdConnectionD,fourthLayerD);
			time_t end=time(NULL);
			timeTaken+=abs(difftime(start,end));
			checkError(cudaMemcpy(fourthLayer,fourthLayerD,fourthLS,cudaMemcpyDeviceToHost));
			checkError(cudaMemcpy(thirdLayer,thirdLayerD,thirdLS,cudaMemcpyDeviceToHost));
			checkError(cudaMemcpy(secondLayer,secondLayerD,secondLS,cudaMemcpyDeviceToHost));
			checkError(cudaMemcpy(inputLayer,inputLayerD,inputLS,cudaMemcpyDeviceToHost));
			double err[fourthLimit];
			double error=0.0;
			for(int m=0;m<fourthLimit;m++)
			{
				err[m]=(m+1)==expectedOutput?fourthLayer[m]-1/*0.99999*/:fourthLayer[m];//-0.00001;
				//double s=(m+1)==expectedOutput?0.99999:0.00001;
				//error+=(0.5*(s-fourthLayer[m])*(s-fourthLayer[m]));
			}
			if((k+1)%1000==0)
			{
				cout<<"Error: Iteration "<<(k+1)<<" Phoneme "<<phoneme<<" [";
				double avg_error=0.0;
				for(int m=0;m<fourthLimit;m++)
				{
					avg_error+=abs(err[m]);
					cout<<setprecision(6)<<err[m]<<" ";
				}
				avg_error=avg_error/fourthLimit;
				cout<<"]"<<endl<<setprecision(6)<<avg_error<<endl;
			}
			backPropagate(err);
			cudaMemset(inputLayerD,0,inputLS);
			cudaMemset(secondLayerD,0,secondLS);
			cudaMemset(thirdLayerD,0,thirdLS);
			cudaMemset(fourthLayerD,0,fourthLS);
			file.close();
		}
	}
	copyBackData();
}
void commitToFile()
{
	int i=0,j=0;
	ofstream outputFile("inputConnection");
	for(i=0;i<secondLimit;i++)
	{
		for(j=0;j<inputLimit;j++)
		{
			int index=i*inputLimit+j;
			outputFile<<setprecision(6)<<inputConnection[index]<<endl;
		}
	}
	outputFile.close();
	outputFile.open("secondConnection");
	for(i=0;i<thirdLimit;i++)
	{
		for(j=0;j<secondLimit;j++)
		{
			int index=i*inputLimit+j;
			outputFile<<setprecision(6)<<secondConnection[index]<<endl;
		}
	}
	outputFile.close();
	outputFile.open("thirdConnection");
	for(i=0;i<fourthLimit;i++)
	{
		for(j=0;j<thirdLimit;j++)
		{
			int index=i*inputLimit+j;
			outputFile<<setprecision(6)<<thirdConnection[index]<<endl;
		}
	}
	outputFile.close();
}
double sigmoidPredict(double value)
{
	double val=1/(1+exp(-value));
	return val;
}
double activationPredict(double value,int derivative)
{
	double val=derivative==1?value*(1-value):sigmoidPredict(value);
	if(isnan(val))
	{
		val=0.0;
	}
	return val;
}
void predict()
{
	cout << "No error till here (1)\n";
	for(int i=0;i<secondLimit;i++)
	{
		double sum=0.0;
		for(int j=0;j<inputLimit;j++)
		{
			int index=i*inputLimit+j;
			sum+=(inputLayer[j]*inputConnection[index]);
		}
		secondLayer[i]=activationPredict(sum,0);
	}
	cout << "No error till here (2)\n";
	for(int i=0;i<thirdLimit;i++)
	{
		double sum=0.0;
		for(int j=0;j<secondLimit;j++)
		{
			int index=i*secondLimit+j;
			sum+=(secondLayer[j]*secondConnection[index]);
		}
		thirdLayer[i]=activationPredict(sum,0);
	}
	cout << "No error till here (3)\n";
	for(int i=0;i<fourthLimit;i++)
	{
		double sum=0.0;
		for(int j=0;j<thirdLimit;j++)
		{
			int index=i*thirdLimit+j;
			sum+=(thirdLayer[j]*thirdConnection[index]);
		}
		fourthLayer[i]=activationPredict(sum,0);
	}
	cout << "No error till here (4)\n";
	for(map<string,int>::iterator key=phonemeIDMapping.begin();key!=phonemeIDMapping.end();++key)
	{
		cout<<fixed<<setprecision(6)<<"Phoneme: "<<key->first<<" Probability: "<<fourthLayer[key->second]<<endl;
	}
}
void initPrediction(string filename)
{
	ifstream file(filename.c_str());
	int i=0;
	cout << "Hi\n";
	while(file)
	{
		cout << i << " ";
		double positive,negative;
		file>>positive>>negative;
		double val=positive>negative?positive:-negative;
		inputLayer[i]=normalize(val);
		i++;
	}
	cout << "Ohno\n";
	predict();
}
int main()
{
	time_t start=time(NULL);
	srand(start);
	initLearningData();
	init();
	cout<<"Learning phase started..."<<endl;
	learner();
	cout<<"Learning phase finished..."<<endl;
	cout<<"Prediction phase started..."<<endl;
	initPrediction("Sound/test");
	cout<<"Prediction phase finished..."<<endl;
	time_t end=time(NULL);
	double total_time=abs(difftime(start,end));
	double avg_time=timeTaken/(double)maxIterations;
	cout<<"Total time taken: "<<total_time<<" seconds"<<endl;
	cout<<"Total iterations: "<<maxIterations<<endl;
	cout<<"Average time for one iteration: "<<avg_time<<" seconds"<<endl;
	cout<<"Time taken for copying to and from the device: "<<(total_time-timeTaken)<<" seconds"<<endl;
	commitToFile();
	return 0;
}
