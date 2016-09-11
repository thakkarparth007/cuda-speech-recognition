//Native Libraries
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <iostream>
#include <time.h>
#include <map>
#include <limits>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <algorithm>

//CUDA Libraries
#include <cuda.h>
#include <math_functions.h>

//Size Macros
#define inputLimit 10
#define secondLimit 20
#define thirdLimit 20
#define fourthLimit 41

//Error Checking
#define checkError(ans) {gpuAssert((ans),__FILE__,__LINE__);}

__device__ inline void atomicadd(float* address, float value){
#if __CUDA_ARCH__ >= 200 // for Fermi, atomicAdd supports floats
  atomicAdd(address,value);
#elif __CUDA_ARCH__ >= 110
// float-atomic-add from 
// [url="http://forums.nvidia.com/index.php?showtopic=158039&view=findpost&p=991561"]http://forums.nvidia.com/index.php?showtop...st&p=991561[/url]
  float old = value;
  while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);
#endif
}

using namespace std;

//Global Variables
int maxIterations=50000;
__constant__ float bias=0.0;
__constant__ float learningRate=0.1;
float minValue=0.0;
float maxValue=5000.0;
float integer_maximum=(float)numeric_limits<int>::max();
float integer_minimum=(float)numeric_limits<int>::min();
int inputCS=inputLimit*secondLimit*sizeof(float),inputLS=inputLimit*sizeof(float);
int secondCS=thirdLimit*secondLimit*sizeof(float),secondLS=secondLimit*sizeof(float);
int thirdCS=thirdLimit*fourthLimit*sizeof(float),thirdLS=thirdLimit*sizeof(float);
int fourthLS=fourthLimit*sizeof(float);
float *aInputLayerD,*aSecondLayerD,*aThirdLayerD,*aFourthLayerD;
float *zSecondLayerD,*zThirdLayerD,*zFourthLayerD;
float *inputConnectionD, *secondConnectionD, *thirdConnectionD;
float *delta4D, *delta3D, *delta2D, *errD;

// store the activations (a values) of the following layers
float aInputLayer[inputLimit];
float aSecondLayer[secondLimit];
float aThirdLayer[thirdLimit];
float aFourthLayer[fourthLimit];

// store the combined inputs (z values) of the following layers
float zSecondLayer[secondLimit];
float zThirdLayer[thirdLimit];
float zFourthLayer[fourthLimit];

map<string,int> phonemeIDMapping;
float inputConnection[secondLimit*inputLimit];
float secondConnection[thirdLimit*secondLimit];
float thirdConnection[fourthLimit*thirdLimit];
string phonemes[41];
float timeTaken=0;

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
float getRandomWeight()
{
//	float val=(float)rand();
//	val=(val-integer_minimum)/(integer_maximum-integer_minimum);
//	return val;
	return (rand() - RAND_MAX/2.0) / (RAND_MAX * 1.0);
}
float normalize(float value)
{
	float val=(value-minValue)/(maxValue-minValue);
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

	checkError(cudaMalloc((void**)&aInputLayerD,inputLS));
	checkError(cudaMalloc((void**)&aSecondLayerD,secondLS));
	checkError(cudaMalloc((void**)&aThirdLayerD,thirdLS));
	checkError(cudaMalloc((void**)&aFourthLayerD,fourthLS));

	checkError(cudaMalloc((void**)&zSecondLayerD,secondLS));
	checkError(cudaMalloc((void**)&zThirdLayerD,thirdLS));
	checkError(cudaMalloc((void**)&zFourthLayerD,fourthLS));

	checkError(cudaMalloc((void**)&errD,fourthLS));
	checkError(cudaMalloc((void**)&delta4D,fourthLS));
	checkError(cudaMalloc((void**)&delta3D,thirdLS));
	checkError(cudaMalloc((void**)&delta2D,secondLS));
}
__device__ float sigmoid(float value)
{
	float val=1/(1+exp(-value));
	return val;
}
__device__ float activation(float value,int derivative)
{
	float val=sigmoid(value);
	val *= derivative==1? (1-val): 1;
	if(isnan(val))
	{
		val=0.0;
	}
	return val;
}
__global__ void calculateError(float *aFourth, float *err, int expectedOutput)
{
	int i = threadIdx.x;
	err[i] = aFourth[i] - (i + 1 == expectedOutput);
}
__global__ void backPropagate4to3(float *thirdC,float *zFourth,float *aThird,float *err,float *delta4)
{
	//__shared__ float deltas[thirdLimit];
	int i=threadIdx.x;
	int j=threadIdx.y;

	int index=i*thirdLimit+j;
	delta4[i]=err[i]*activation(zFourth[i],1);
	float correction=delta4[i]*aThird[j];
	thirdC[index]=thirdC[index]-learningRate*correction;
	//if(i == j && i == 0)
	//	printf("(%d,%d):err=%f,zFourth=%f,%f*%f=%f\t%f\n", i, j, err[i], zFourth[i], delta4[i], aThird[j], correction,thirdC[index]);
}
__global__ void backPropagate3to2(float *thirdC, float *secondC,float *zThird,float *aSecond,float *delta4,float *delta3)
{
	int i=threadIdx.x;
	int j=threadIdx.y;
	int index=i*secondLimit+j;

	// calculate delta3 from delta4
	float dotprod_ith_term = 0;
	for(int k = 0; k < fourthLimit; k++) {
		dotprod_ith_term += thirdC[k*thirdLimit+i] * delta4[k];
	}

	delta3[i] = dotprod_ith_term * activation(zThird[i], 1);
	float correction=delta3[i]*aSecond[j];
	secondC[index]=secondC[index]-learningRate*correction;
}
__global__ void backPropagate2to1(float *secondC, float *inputC,float *zSecond,float *aInput,float *delta3,float *delta2)
{
	int i=threadIdx.x;
	int j=threadIdx.y;
	int index=i*inputLimit+j;

	// calculate delta2 from delta3
	float dotprod_ith_term = 0;
	for(int k = 0; k < thirdLimit; k++) {
		dotprod_ith_term += secondC[k*secondLimit+i] * delta3[k];
	}

	delta2[i] = dotprod_ith_term * activation(zSecond[i], 1);
	float correction=delta2[i]*aInput[j];
	inputC[index]=inputC[index]-learningRate*correction;
}
void backPropagate(int expectedOutput)
{
	time_t start=time(NULL);
	
	dim3 fourToThree(fourthLimit, thirdLimit);
	dim3 threeToTwo(thirdLimit, secondLimit);
	dim3 twoToOne(secondLimit, inputLimit);

	calculateError<<<1, fourthLimit>>>(aFourthLayerD, errD, expectedOutput);
	checkError(cudaThreadSynchronize());
	backPropagate4to3<<<1, fourToThree>>>(thirdConnectionD,zFourthLayerD,aThirdLayerD,errD,delta4D);
	checkError(cudaThreadSynchronize());
	backPropagate3to2<<<1, threeToTwo>>>(thirdConnectionD,secondConnectionD,zThirdLayerD,aSecondLayerD,delta4D,delta3D);
	checkError(cudaThreadSynchronize());
	backPropagate2to1<<<1, twoToOne>>>(secondConnectionD,inputConnectionD,zSecondLayerD,aInputLayerD,delta3D,delta2D);
	checkError(cudaThreadSynchronize());
	time_t end=time(NULL);
	timeTaken+=abs(difftime(start,end));
}
__global__ void learn1to2(float *aInput,float *inputC,float *aSecond, float *zSecond)
{
	int i=threadIdx.x;
	float sum=0.0;
	for(int j=0;j<inputLimit;j++)
	{
		int index=i*inputLimit+j;
		sum+=(aInput[j]*inputC[index]);
	}
	aSecond[i]=activation(sum,0);
	zSecond[i] = sum;
}
__global__ void learn2to3(float *aSecond,float *secondC,float *aThird, float *zThird)
{
	int i=threadIdx.x;
	float sum=0.0;
	for(int j=0;j<secondLimit;j++)
	{
		int index=i*secondLimit+j;
		sum+=(aSecond[j]*secondC[index]);
	}
	aThird[i]=activation(sum,0);
	zThird[i] = sum;
}
__global__ void learn3to4(float *aThird,float *thirdC,float *aFourth, float *zFourth)
{
	int i=threadIdx.x;
	float sum=0.0;
	for(int j=0;j<thirdLimit;j++)
	{
		int index=i*thirdLimit+j;
		sum+=(aThird[j]*thirdC[index]);
	}
	aFourth[i]=activation(sum,0);
	zFourth[i] = sum;
}
void copyBackData()
{
	checkError(cudaMemcpy(inputConnection,inputConnectionD,inputCS,cudaMemcpyDeviceToHost));
	checkError(cudaMemcpy(secondConnection,secondConnectionD,secondCS,cudaMemcpyDeviceToHost));
	checkError(cudaMemcpy(thirdConnection,thirdConnectionD,thirdCS,cudaMemcpyDeviceToHost));
}
void printArray(float *arr,int size)
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
	float preprocessedInputs[41][inputLimit];
	int sampleOrder[41]; // stores the order in which SGD is to be performed
						 // this array will be shuffled every iteration

	for(int i=0;i<41;i++)
	{
		string filename="Sound/preprocess/preprocess_"+phonemes[i]+"_1";
		ifstream file(filename.c_str());
		string phoneme=phonemes[i];
		int j=0;
		while(j<inputLimit)
		{
			float positive,negative;
			file>>positive>>negative;
			float val=positive>negative?positive:-negative;
			preprocessedInputs[i][j]=normalize(val);
			j++;
		}
		file.close();
		sampleOrder[i] = i;
	}

	for(int k=0;k<maxIterations;k++)
	{
		random_shuffle(sampleOrder, sampleOrder+41);

		for(int i=0;i<41;i++)
		{
			string phoneme=phonemes[sampleOrder[i]];
			int expectedOutput=phonemeIDMapping[phoneme];
			/*
			cout << "===============================================\n";
			cout << "Phoneme: " << phoneme << "\n";
			checkError(cudaMemcpy(thirdConnection,thirdConnectionD,inputCS,cudaMemcpyDeviceToHost));
			for(int lol=0;lol<fourthLimit;lol++)
			{
				for(int j=0;j<thirdLimit;j++)
				{
					int index=lol*thirdLimit+j;
					cout << setprecision(2) << thirdConnection[index] << " ";//=getRandomWeight();
				}
				cout << "\n";
			} 
			cout << "==============================================\n";*/
			
			//memcpy(aInputLayer, preprocessedInputs[sampleOrder[i]], inputLS);
			//checkError(cudaMemcpy(aInputLayerD,aInputLayer,inputLS,cudaMemcpyHostToDevice));
			checkError(cudaMemcpy(aInputLayerD,preprocessedInputs[sampleOrder[i]],inputLS,cudaMemcpyHostToDevice));
			time_t start=time(NULL);
			learn1to2<<<1,secondLimit>>>(aInputLayerD, inputConnectionD, aSecondLayerD, zSecondLayerD);
			learn2to3<<<1,thirdLimit>>>(aSecondLayerD, secondConnectionD, aThirdLayerD, zThirdLayerD);
			learn3to4<<<1,fourthLimit>>>(aThirdLayerD, thirdConnectionD, aFourthLayerD, zFourthLayerD);
			time_t end=time(NULL);
			timeTaken+=abs(difftime(start,end));
		//  checkError(cudaMemcpy(aFourthLayer,aFourthLayerD,fourthLS,cudaMemcpyDeviceToHost));
		//	checkError(cudaMemcpy(aThirdLayer,aThirdLayerD,thirdLS,cudaMemcpyDeviceToHost));
		//	checkError(cudaMemcpy(aSecondLayer,aSecondLayerD,secondLS,cudaMemcpyDeviceToHost));
		//	checkError(cudaMemcpy(aInputLayer,aInputLayerD,inputLS,cudaMemcpyDeviceToHost));
		//	float err[fourthLimit];
	//		float error=0.0;
		//	for(int m=0;m<fourthLimit;m++)
		//	{
		//		err[m]=(m+1)==expectedOutput? aFourthLayer[m]-1:aFourthLayer[m];
		//		float s=(m+1)==expectedOutput?0.99999:0.00001;
		//		error+=(0.5*(s-fourthLayer[m])*(s-fourthLayer[m]));
		//	}
			if((k)%1000==0)
			{
				checkError(cudaMemcpy(aFourthLayer,aFourthLayerD,fourthLS,cudaMemcpyDeviceToHost));
				cout<<"Error: Iteration "<<(k+1)<<" Phoneme "<<phoneme<<" [";
				float avg_error=0.0, cur_err = 0.0;
				for(int m=0;m<fourthLimit;m++)
				{
					cur_err = (m+1)==expectedOutput? aFourthLayer[m]-1:aFourthLayer[m];
					avg_error+=abs(cur_err);
					cout<<setprecision(6)<<cur_err<<" ";
				}
				avg_error=avg_error/fourthLimit;
				cout<<"]"<<endl<<setprecision(6)<<avg_error<<endl;
			}
			backPropagate(expectedOutput);
		//	cudaMemset(aInputLayerD,0,inputLS);
		//	cudaMemset(aSecondLayerD,0,secondLS);
		//	cudaMemset(aThirdLayerD,0,thirdLS);
		//	cudaMemset(aFourthLayerD,0,fourthLS);
		//	file.close();
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
float sigmoidPredict(float value)
{
	float val=1/(1+exp(-value));
	return val;
}
float activationPredict(float value,int derivative)
{
	float val=derivative==1?value*(1-value):sigmoidPredict(value);
	if(isnan(val))
	{
		val=0.0;
	}
	return val;
}
void predict()
{
	for(int i=0;i<secondLimit;i++)
	{
		float sum=0.0;
		for(int j=0;j<inputLimit;j++)
		{
			int index=i*inputLimit+j;
			sum+=(aInputLayer[j]*inputConnection[index]);
		}
		aSecondLayer[i]=activationPredict(sum,0);
		zSecondLayer[i] = sum;
	}
	for(int i=0;i<thirdLimit;i++)
	{
		float sum=0.0;
		for(int j=0;j<secondLimit;j++)
		{
			int index=i*secondLimit+j;
			sum+=(aSecondLayer[j]*secondConnection[index]);
		}
		aThirdLayer[i]=activationPredict(sum,0);
		zSecondLayer[i] = sum;
	}
	for(int i=0;i<fourthLimit;i++)
	{
		float sum=0.0;
		for(int j=0;j<thirdLimit;j++)
		{
			int index=i*thirdLimit+j;
			sum+=(aThirdLayer[j]*thirdConnection[index]);
		}
		aFourthLayer[i]=activationPredict(sum,0);
		zFourthLayer[i] = sum;
	}
	for(map<string,int>::iterator key=phonemeIDMapping.begin();key!=phonemeIDMapping.end();++key)
	{
		cout<<fixed<<setprecision(6)<<"Phoneme: "<<key->first<<" Probability: "<<aFourthLayer[key->second]<<endl;
	}
}
void initPrediction(string filename)
{
	ifstream file(filename.c_str());
	int i=0;
	while(file)
	{
		float positive,negative;
		file>>positive>>negative;
		float val=positive>negative?positive:-negative;
		aInputLayer[i]=normalize(val);
		i++;
	}
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
	float total_time=abs(difftime(start,end));
	float avg_time=timeTaken/(float)maxIterations;
	cout<<"Total time taken: "<<total_time<<" seconds"<<endl;
	cout<<"Total iterations: "<<maxIterations<<endl;
	cout<<"Average time for one iteration: "<<avg_time<<" seconds"<<endl;
	cout<<"Time taken for copying to and from the device: "<<(total_time-timeTaken)<<" seconds"<<endl;
	commitToFile();
	return 0;
}
