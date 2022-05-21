#include <iostream>
using namespace std;
#include <iomanip>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <arm_neon.h>
/*
int omp_get_num_threads(void);返回线程数
int omp_get_thread_num(void);返回编号从0开始
全局变量，临界区P23
*/
#define N 1024
#define NUM_THREADS 7
#define cnt 1
#define RANDOM_ADD 5
#define OUTPUTN 6

float** M;
void init() {
	M = new float* [N];
	for (int i = 0; i < N; i++) {
		M[i] = new float[N];
		for (int j = 0; j < i; j++)
			M[i][j] = 0;

		for (int j = i; j < N; j++)
			M[i][j] = rand() % 50;
	}
	for (int k = 0; k < RANDOM_ADD; k++) {
		for (int i = 0; i < N; i++) {
			int temp = rand() % N;
			for (int j = 0; j < N; j++)
				M[temp][j] += M[i][j];
		}
	}
}
void m_reset() {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < i; j++)
			M[i][j] = 0;

		for (int j = i; j < N; j++)
			M[i][j] = rand() % 50;
	}
	for (int k = 0; k < RANDOM_ADD; k++) {
		for (int i = 0; i < N; i++) {
			int temp = rand() % N;
			for (int j = 0; j < N; j++)
				M[temp][j] += M[i][j];
		}
	}
}
void m_reset2(){
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
			M[i][j] = rand() % 50;
	}
}
void output() {
	for (int i = 0; i < OUTPUTN; i++) {
		for (int j = 0; j < OUTPUTN; j++) {
			cout << setw(6) << fixed << setprecision(2)<< M[i][j];
		}
		cout << endl;
	}
}
void ori() {
	float tmp;
	int i,j,k;
	for (int k = 0; k < N; k++) {
		tmp = M[k][k];
		for (int j = k; j < N; j++) {
			M[k][j] = M[k][j] / tmp;
		}
		for (int i = k + 1; i < N; i++) {
			tmp = M[i][k];
			for (int j = k + 1; j < N; j++) {
				M[i][j] = M[i][j] - tmp * M[k][j];
			}
			M[i][k] = 0;
		}
	}
}


void omp(){
	int i,j,k;
	float tmp;
	#pragma omp parallel num_threads(NUM_THREADS),private(i,j,k,tmp)

	for(k = 1;k<N;k++){
		#pragma omp single
		{
			tmp = M[k][k];
			for(j = k+1;j<N;j++)
				M[k][j] = M[k][j]/tmp;
		
			M[k][k] = 1;
		}
		//#pragma omp for simd
		//#pragma omp for
		#pragma omp for schedule(dynamic)
		for (i = k+1; i <N; ++i){
			tmp = M[i][k];
			for(int j = k+1;j<N;j++)
				M[i][j] = M[i][j] - tmp * M[k][j];
		
		M[i][k] = 0;
		}
	}
}

void omp2(){
	int i,j,k;
	float tmp;
	#pragma omp parallel num_threads(NUM_THREADS),private(i,j,k,tmp)
	for(k = 1;k<N;k++){
			tmp = M[k][k];
		#pragma omp for
			for(j = k+1;j<N;j++)
				M[k][j] = M[k][j]/tmp;
			M[k][k] = 1;
		#pragma omp barrier
		#pragma omp single
		for (i = k+1; i <N; ++i){
			tmp = M[i][k];
			for(int j = k+1;j<N;j++)
				M[i][j] = M[i][j] - tmp * M[k][j];
		
		M[i][k] = 0;
		}
	}
}

void omp3(){
	int i,j,k;
	float tmp;
	#pragma omp parallel num_threads(NUM_THREADS),private(i,j,k,tmp)
	for(k = 1;k<N;k++){
			tmp = M[k][k];
		#pragma omp for
			for(j = k+1;j<N;j++)
				M[k][j] = M[k][j]/tmp;
			M[k][k] = 1;
		#pragma omp barrier
		#pragma omp for simd
		for (i = k+1; i <N; ++i){
			tmp = M[i][k];
			for(int j = k+1;j<N;j++)
				M[i][j] = M[i][j] - tmp * M[k][j];
		
		M[i][k] = 0;
		}
	}
}

void omp3_SIMD(){
	int i,j,k;
	float tmp;
	#pragma omp parallel num_threads(NUM_THREADS),private(i,j,k,tmp)
	for(k = 1;k<N;k++){
			tmp = M[k][k];
		#pragma omp for
			for(j = k+1;j<N;j++)
				M[k][j] = M[k][j]/tmp;
			M[k][k] = 1;
		#pragma omp barrier
		#pragma omp for
		for (i = k + 1; i < N; i++) {
		float32x4_t v1 = vmovq_n_f32(M[i][k]);
        float32x4_t v0,v2;
        for (j = k + 1; j <= N - 4; j += 4) {
            v2 = vld1q_f32(M[k] + j);
            v0 = vld1q_f32(M[i] + j);
            v2 = vmulq_f32(v1, v2);
            v0 = vsubq_f32(v0, v2);
            vst1q_f32(M[i] + j, v0);
        }
        for (j = j-4; j < N; j++)
            M[i][j] = M[i][j] - M[i][k] * M[k][j];
        M[i][k] = 0;
		}
	}
}

int main() {
	init();
	//int thread_count = strtol(argv[1],NULL,10);
	struct timespec sts,ets;
	time_t dsec;
	long dnsec;
	
	float dsec_record = 0;
	float dnsec_record = 0;
	for (int i = 0; i < cnt; ++i){
		m_reset2();
		timespec_get(&sts, TIME_UTC);
		ori();
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
	float ori = dsec_record * 1000 + dnsec_record/1000000;
	cout << "Ori =  " << ori << "ms" << endl;

	dnsec_record = 0;
	dsec_record = 0;
	for (int i = 0; i < cnt; ++i){
		m_reset();
		timespec_get(&sts, TIME_UTC);
		omp3_SIMD();
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
	cout << "Omp =  " << omp << "ms" << endl;
	cout<<"Ratio = "<<ori/omp<<endl;
	return 0;
}