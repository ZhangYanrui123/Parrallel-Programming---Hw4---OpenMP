#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <omp.h>
#include <arm_neon.h>
using namespace std;

#define NUM_THREADS 4
#define cnt 1
#define COL 3799
#define ROW1 2759
#define ROW2 1953
#define FILE1 "/home/sTest/data/Groebner/6_3799_2759_1953/1.txt"
#define FILE2 "/home/sTest/data/Groebner/6_3799_2759_1953/2.txt"
#define FILE3 "/home/sTest/data/Groebner/6_3799_2759_1953/3.txt"
/*
FILE11 "/home/sTest/data/Groebner/1_130_22_8/1.txt"
FILE12 "/home/sTest/data/Groebner/1_130_22_8/2.txt"
FILE13 "/home/sTest/data/Groebner/1_130_22_8/3.txt"
FILE21 "/home/sTest/data/Groebner/2_254_106_53/1.txt"
FILE22 "/home/sTest/data/Groebner/2_254_106_53/2.txt"
FILE23 "/home/sTest/data/Groebner/2_254_106_53/3.txt"
FILE31 "/home/sTest/data/Groebner/3_562_170_53/1.txt"
FILE32 "/home/sTest/data/Groebner/3_562_170_53/2.txt"
FILE33 "/home/sTest/data/Groebner/3_562_170_53/3.txt"
FILE41 "/home/sTest/data/Groebner/4_1011_539_263/1.txt"
FILE42 "/home/sTest/data/Groebner/4_1011_539_263/2.txt"
FILE43 "/home/sTest/data/Groebner/4_1011_539_263/3.txt"
FILE51 "/home/sTest/data/Groebner/5_2362_1226_453/1.txt"
FILE52 "/home/sTest/data/Groebner/5_2362_1226_453/2.txt"
FILE53 "/home/sTest/data/Groebner/5_2362_1226_453/3.txt"
FILE61 "/home/sTest/data/Groebner/6_3799_2759_1953/1.txt"
FILE62 "/home/sTest/data/Groebner/6_3799_2759_1953/2.txt"
FILE63 "/home/sTest/data/Groebner/6_3799_2759_1953/3.txt"
FILE71 "/home/sTest/data/Groebner/7_8399_6375_4535/1.txt"
FILE72 "/home/sTest/data/Groebner/7_8399_6375_4535/2.txt"
FILE73 "/home/sTest/data/Groebner/7_8399_6375_4535/3.txt"
FILE81 "/home/sTest/data/Groebner/8_23045_18748_14325/1.txt"
FILE82 "/home/sTest/data/Groebner/8_23045_18748_14325/2.txt"
FILE83 "/home/sTest/data/Groebner/8_23045_18748_14325/3.txt"
FILE91 "/home/sTest/data/Groebner/9_37960_29304_14921/1.txt"
FILE92 "/home/sTest/data/Groebner/9_37960_29304_14921/2.txt"
FILE93 "/home/sTest/data/Groebner/9_37960_29304_14921/3.txt"
FILEA1 "/home/sTest/data/Groebner/10_43577_39477_54274/1.txt"
FILEA2 "/home/sTest/data/Groebner/10_43577_39477_54274/2.txt"
FILEA3 "/home/sTest/data/Groebner/10_43577_39477_54274/3.txt"
FILEB1 "/home/sTest/data/Groebner/11_85401_5724_756/1.txt"
FILEB2 "/home/sTest/data/Groebner/11_85401_5724_756/2.txt"
FILEB3 "/home/sTest/data/Groebner/11_85401_5724_756/3.txt"
*/
bool mapof[COL] = {false};
struct bitMap{
	int size;
	int* map;
	int first;
	bool empty;
	bool up;
	bitMap(){
		size = 0;
		map = nullptr;
		first = -1;
		empty = true;
		up = false;
	}
	bitMap(int col){
		size = col/32 + 1;
		map = new int[size];
		first = -1;
		empty = true;
		up = false;
	}
	void setCol(int col){
		size = col/32 + 1;
		map = new int[size];
	}
	void set(int index){
		int mapindex = index/32;
		int bitindex = index%32;
		//cout<<"mapindex = "<<mapindex<<" bitindex = "<<bitindex<<endl;
		map[size - mapindex - 1] |= 1 << bitindex;
	}
	void output(){
		for(int i = 0;i<size - 1;i++)
			cout<<map[i]<<" ";
		cout<<map[size-1]<<endl;
	}
	void clear(){
		for(int i = 0;i<size;i++)
			map[i] = 0;
		empty = true;
	}
	bool isEmpty(){
		bool flag= true;
		for(int i = 0;i<size;i++)
			if(map[i]!=0)
				flag = false;
		empty = flag;
		return flag;
	}
	int getFirst(){
		if(this->isEmpty())
			return -1;
		for(int i = 0;i<size;i++){
			if(map[i] == 0)
				continue;
			else{
				for(int j = 31;j>=0;j--){
					if(map[i] & (1<<j))
						return 32*(size - i - 1) + j;
				}
			}
		}
		return -1;
	}
};
void readFile(bitMap* mapG1, bitMap* mapG2, bitMap* mapGO){

	ifstream myfile1(FILE1,ios::in);
	if(!myfile1.is_open()){
		cout<<"open file 1 falied"<<endl;
	}
	string str;
	int row = 0;
	while (getline(myfile1, str)) {
        stringstream ss; //输入流
        ss << str; //向流中传值
        if (!ss.eof()) {
            int index;
            if(ss>>index){
            	row = index;
            	mapG1[row].set(index);
            	mapG1[row].first = index;
            	mapG1[row].empty = false;
            }
            while (ss >> index)
                mapG1[row].set(index);
        }
        row++;
    }
	myfile1.close();

	ifstream myfile2(FILE2,ios::in);
	if(!myfile2.is_open()){
		cout<<"open file 2 falied"<<endl;
	}
	row = 0;
	while (getline(myfile2, str)) {
        stringstream ss; //输入流
        ss << str; //向流中传值
        if (!ss.eof()) {
            int index;
            if(ss>>index){
            	mapG2[row].set(index);
            	mapG2[row].first = index;
            }
            while (ss >> index)
                mapG2[row].set(index);
        }
        row++;
    }
	myfile2.close();
	
	ifstream myfile3(FILE3,ios::in);
	if(!myfile3.is_open()){
		cout<<"open file 3 falied"<<endl;
	}
	row = 0;
	while (getline(myfile3, str)) {
        stringstream ss; //输入流
        ss << str; //向流中传值
        if (!ss.eof()) {
            int index;
            while (ss >> index)
                mapGO[row].set(index);
        }
        row++;
    }
	myfile3.close();
}

void deMap(bitMap* mapG){
	int num = 0;
	int size = mapG[0].size;
	for(int i = 0;i<ROW2;i++){
		cout<<"【"<<i+1<<"】:";
		for(;num<size;num++){
			for(int j = 31;j>=0;j--){
				if(mapG[i].map[num] & (1<<j))
					cout<<32*(size-num)-(32 - j)<<" ";
			}
		}
		cout<<endl;
		num = 0;
	}
}
void Ori(bitMap* mapG1,bitMap* mapG2){
	//遍历每个被消元行
	for(int i = 0;i<ROW2;i++){
		//cout<<"正在第"<<i<<"个被消元行"<<endl;
		while(!mapG2[i].isEmpty()){
			int first = mapG2[i].getFirst();
			if(!mapG1[first].isEmpty()){
				//被消元行和消元子按每个32bit异或
				for(int j = 0 ; j<COL/32+1;j++){
					mapG2[i].map[j] ^= mapG1[first].map[j];
				}
			}
			else{
				//升格为消元子
				mapG1[first] = mapG2[i];
				break;
			}
			//mapG2[i].output();
		}
	}
}

void Omp(bitMap* mapG1,bitMap* mapG2){
	int i,j;
	int first;
	//遍历每个被消元行
	for(i = 0;i<ROW2;i++){
		//cout<<"我是第"<<omp_get_thread_num()<<"个线程"<<endl;
		//cout<<"正在第"<<i<<"个被消元行"<<endl;
		while(!mapG2[i].isEmpty()){
			#pragma omp parallel num_threads(NUM_THREADS)
			first = mapG2[i].getFirst();
			//cout<<"first = " <<first<<endl;
			if(!mapG1[first].isEmpty()){
				//被消元行和消元子按每个32bit异或
				//#pragma omp parallel num_threads(NUM_THREADS)
				//#pragma omp for simd
				#pragma omp for schedule(dynamic)
				for(j = 0 ; j<COL/32+1;j++){
					//cout<<"j = "<<j<<";COL/32+1 = "<<COL/32+1<<endl;
					//cout<<"j = "<<j<<"之前:"<<mapG2[i].map[j];
					mapG2[i].map[j] ^= mapG1[first].map[j];
					//cout<<"第"<<omp_get_thread_num()<<"个线程AND j = "<<j<<"该字节："<<mapG2[i].map[j]<<endl;
					//cout<<",之后:"<<mapG2[i].map[j]<<endl;
					//cout<<"j = "<<j<<";"<<first<<"消元子："<<mapG1[first].map[j]<<endl;
				}
				//mapG2[i].output();
				
			}
			else{
				//升格为消元子
				mapG1[first] = mapG2[i];
				break;
			}
			//deMap(mapG2);
		}
	}

}
int main(){
	bitMap* mapG1;
	bitMap* mapG2;
	bitMap* mapGO;
	//教训：不要在主函数声明指针，在子函数new分配，这样是分配不了的，它还是空指针！！！

	struct timespec sts,ets;
	time_t dsec;
	long dnsec;
	
	float dsec_record = 0;
	float dnsec_record = 0;
	for (int i = 0; i < cnt; ++i){
		mapG1 = new bitMap[COL];
		for(int i = 0;i<COL;i++){
			mapG1[i].setCol(COL);
			mapG1[i].clear();
		}
		mapG2 = new bitMap[ROW2];
		for(int i = 0;i<ROW2;i++){
			mapG2[i].setCol(COL);
			mapG2[i].clear();
		}
		mapGO = new bitMap[ROW2];
		for(int i = 0;i<ROW2;i++){
			mapGO[i].setCol(COL);
			mapGO[i].clear();
		}
		readFile(mapG1,mapG2,mapGO);
		timespec_get(&sts, TIME_UTC);
		Ori(mapG1,mapG2);
		timespec_get(&ets, TIME_UTC);
		dsec=ets.tv_sec-sts.tv_sec;
		dnsec=ets.tv_nsec-sts.tv_nsec;
		if (dnsec_record<0){
			dnsec_record--;
			dnsec_record+=1000000000ll;
		}
		dsec_record+=(float)dsec;
		dnsec_record+=(float)dnsec;
	}
	deMap(mapG2);
	float ori = dsec_record * 1000 + dnsec_record/1000000;
	cout << "Ori =  " << ori << "ms" << endl;

	delete[] mapG1;
	delete[] mapG2;
	delete[] mapGO;
	mapG1 = new bitMap[COL];
	for(int i = 0;i<COL;i++){
		mapG1[i].setCol(COL);
		mapG1[i].clear();
	}
	mapG2 = new bitMap[ROW2];
	for(int i = 0;i<ROW2;i++){
		mapG2[i].setCol(COL);
		mapG2[i].clear();
	}
	mapGO = new bitMap[ROW2];
	for(int i = 0;i<ROW2;i++){
		mapGO[i].setCol(COL);
		mapGO[i].clear();
	}
	readFile(mapG1,mapG2,mapGO);
	dsec_record = 0;
	dnsec_record = 0;
	for (int i = 0; i < cnt; ++i){
		mapG1 = new bitMap[COL];
		for(int i = 0;i<COL;i++){
			mapG1[i].setCol(COL);
			mapG1[i].clear();
		}
		mapG2 = new bitMap[ROW2];
		for(int i = 0;i<ROW2;i++){
			mapG2[i].setCol(COL);
			mapG2[i].clear();
		}
		mapGO = new bitMap[ROW2];
		for(int i = 0;i<ROW2;i++){
			mapGO[i].setCol(COL);
			mapGO[i].clear();
		}
		readFile(mapG1,mapG2,mapGO);
		timespec_get(&sts, TIME_UTC);
		Omp(mapG1,mapG2);
		timespec_get(&ets, TIME_UTC);
		dsec=ets.tv_sec-sts.tv_sec;
		dnsec=ets.tv_nsec-sts.tv_nsec;
		if (dnsec_record<0){
			dnsec_record--;
			dnsec_record+=1000000000ll;
		}
		dsec_record+=(float)dsec;
		dnsec_record+=(float)dnsec;
	}
	float omp = dsec_record * 1000 + dnsec_record/1000000;
	deMap(mapG2);
	cout << "Omp =  " << omp << "ms" << endl;
	cout<<"Ratio = "<<ori/omp<<endl;
	return 0;
}