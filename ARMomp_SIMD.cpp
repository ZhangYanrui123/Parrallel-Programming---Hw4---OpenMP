#include <iostream>
#include <iomanip>
#include <ctime>
#include <stdio.h>
#include <time.h>
#include <arm_neon.h>
#include <omp.h>

using namespace std;

#define N 1024
#define count 1
#define NUM_THREADS 4
float A[N][N];
void Alloc_aligned(float** f){
    f= new float*[N];
    for(int i = 0;i<N;i++){
        f[i] = (float*) aligned_alloc(32,N*N*sizeof(float));
        for(int j = 0;j<N;j++)
            f[i][j] = rand();
    }
}
void retrive(float** f){
    for(int i = 0;i<N;i++)
        delete[] f[i];
    delete[] f;
}
void ori_LU() {
	for (int k = 0; k < N; k++) {
		for (int j = k + 1; j < N; j++)
			A[k][j] = A[k][j] / A[k][k];
		A[k][k] = 1.0;
		for (int i = k + 1; i < N; i++) {
			for (int j = k + 1; j < N; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			A[i][k] = 0;
		}
	}
}

void NEON_LU_1() {
    for (int k = 0; k < N; k++) {
        float32x4_t v1 = vmovq_n_f32(A[k][k]);
        float32x4_t v0;
        int i;
        for (i = k + 1; i <= N - 4; i += 4) {
            v0 = vld1q_f32(A[k] + i);
            v0 = vdivq_f32(v0, v1);
            vst1q_f32(A[k] + i, v0);
        }
        for (i = i - 4; i < N; i++)
            A[k][i] = A[k][i] / A[k][k];
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++)
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            A[i][k] = 0;
        }
    }
}

void NEON_LU_2(){
    for (int k = 0; k < N; k++) {
        int temp = A[k][k];
        for (int j = k + 1; j < N; j++)
            A[k][j] = A [k][j] / temp;
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            float32x4_t v1 = vmovq_n_f32(A[i][k]);
            float32x4_t v0, v2;
            int j;
            for (j = k + 1; j <= N - 4; j += 4) {
                v2 = vld1q_f32(A[k] + j);
                v0 = vld1q_f32(A[i] + j);
                v2 = vmulq_f32(v1, v2);
                v0 = vsubq_f32(v0, v2);
                vst1q_f32(A[i] + j, v0);
            }
        for (j = j - 4; j < N; j++)
            A[i][j] = A[i][j] - A[i][k] * A[k][j];
        A[i][k] = 0;
        }
    }
}
void NEON_LU_3(){
    for (int k = 0; k < N; k++) {
        float32x4_t v1 = vmovq_n_f32(A[k][k]);
        float32x4_t v0;
        int i;
        for (i = k + 1; i <= N - 4; i += 4) {
            v0 = vld1q_f32(A[k] + i);
            v0 = vdivq_f32(v0, v1);
            vst1q_f32(A[k] + i, v0);
        }
        for (i = i-4; i < N; i++)
            A[k][i] = A[k][i] / A[k][k];
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            v1 = vmovq_n_f32(A[i][k]);
            float32x4_t v2;
            int j;
            for (j = k + 1; j <= N - 4; j += 4) {
                v2 = vld1q_f32(A[k] + j);
                v0 = vld1q_f32(A[i] + j);
                v2 = vmulq_f32(v1, v2);
                v0 = vsubq_f32(v0, v2);
                vst1q_f32(A[i] + j, v0);
            }
            for (j = j-4; j < N; j++)
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            A[i][k] = 0;
        }
    }
}

void NEON_LU_3_omp(){
    omp_set_num_threads(NUM_THREADS);
    for (int k = 0; k < N; k++) {
        float32x4_t v1 = vmovq_n_f32(A[k][k]);
        float32x4_t v0;
        int i;
        #pragma omp parallel
        {
        for (i = k + 1; i <= N - 4; i += 4) {
            v0 = vld1q_f32(A[k] + i);
            v0 = vdivq_f32(v0, v1);
            vst1q_f32(A[k] + i, v0);
        }
        #pragma omp barrier
        #pragma omp single
        for (i = i-4; i < N; i++)
            A[k][i] = A[k][i] / A[k][k];
        A[k][k] = 1.0;
        #pragma omp for
        for (int i = k + 1; i < N; i++) {
            v1 = vmovq_n_f32(A[i][k]);
            float32x4_t v2;
            int j;
            for (j = k + 1; j <= N - 4; j += 4) {
                v2 = vld1q_f32(A[k] + j);
                v0 = vld1q_f32(A[i] + j);
                v2 = vmulq_f32(v1, v2);
                v0 = vsubq_f32(v0, v2);
                vst1q_f32(A[i] + j, v0);
            }
            for (j = j-4; j < N; j++)
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            A[i][k] = 0;
        }
        }
    }
}

void NEON_LU_3_aligned(){
    for (int k = 0; k < N; k++) {
        long long addr;
        float32x4_t v1 = vmovq_n_f32(A[k][k]);
        float32x4_t v0;
        for (int i  = k + 1; i < N; i++) {
            if (i % 4 == 0)
                break;
            A[k][i] = A [k][i] / A[k][k];
        }
        int i;
        for (i = k+1; i<= N - 4; i += 4) {
            v0 = vld1q_f32(A[k] + i);
            v0 = vdivq_f32(v0, v1);
            vst1q_f32(A[k] + i, v0);
        }
        for (i = i-4; i < N; i++)
            A[k][i] = A[k][i] / A[k][k];
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            v1 = vmovq_n_f32(A[i][k]);
            float32x4_t v2;
            int j;
            for (j = k + 1; j < N; j++) {
                if (j % 4 == 0)
                    break;
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            for (; j <= N - 4; j += 4) {
                v2 = vld1q_f32(A[k] + j);
                v0 = vld1q_f32(A[i] + j);
                v2 = vmulq_f32(v1, v2);
                v0 = vsubq_f32(v0, v2);
                vst1q_f32(A[i] + j, v0);
            }
            for (j = j-4; j < N; j++)
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            A[i][k] = 0;
        }
    }
}
void output() {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N - 1; ++j)
			cout << setw(10)<<A[i][j] << " ";
		cout << setw(10)<< A[i][N-1] << endl;
	}
	cout << endl;
}
void m_reset(){
    for(int i = 0; i < N ;i++){
        for(int j = 0;j<i;j++)
            A[i][j] = 0;
        A[i][i] = 1.0;
        for(int j = i + 1;j<N;j++)
            A[i][j] = rand();
    }
    for(int k = 0;k<N;k++)
        for(int i = k+1;i<N;i++)
            for(int j =0;j<N;j++)
                A[i][j]+=A[k][j];
}
int main() {
	m_reset();
	//time measuring
	struct timespec sts,ets;
	timespec_get(&sts, TIME_UTC);
	for(int i = 0;i<count;i++)
	ori_LU();
    timespec_get(&ets, TIME_UTC);
    time_t dsec = ets.tv_sec - sts.tv_sec;
    long dnsec = ets.tv_nsec - sts.tv_nsec;
    if(dnsec < 0){
        dsec--;
        dnsec += 1000000000ll;
    }
    printf ("original = %lld.%09llds\n",dsec,dnsec);

    m_reset();
    timespec_get(&sts, TIME_UTC);
    for(int i = 0;i<count;i++)
	NEON_LU_1();
    timespec_get(&ets, TIME_UTC);
    dsec = ets.tv_sec - sts.tv_sec;
    dnsec = ets.tv_nsec - sts.tv_nsec;
    if(dnsec < 0){
        dsec--;
        dnsec += 1000000000ll;
    }
    printf ("NEON_1 = %lld.%09llds\n",dsec,dnsec);


    m_reset();
    timespec_get(&sts, TIME_UTC);
    for(int i = 0;i<count;i++)
	NEON_LU_2();
    timespec_get(&ets, TIME_UTC);
    dsec = ets.tv_sec - sts.tv_sec;
    dnsec = ets.tv_nsec - sts.tv_nsec;
    if(dnsec < 0){
        dsec--;
        dnsec += 1000000000ll;
    }
    printf ("NEON_2 = %lld.%09llds\n",dsec,dnsec);

    m_reset();
    timespec_get(&sts, TIME_UTC);
    for(int i = 0;i<count;i++)
	NEON_LU_3();
    timespec_get(&ets, TIME_UTC);
    dsec = ets.tv_sec - sts.tv_sec;
    dnsec = ets.tv_nsec - sts.tv_nsec;
    if(dnsec < 0){
        dsec--;
        dnsec += 1000000000ll;
    }
    printf ("NEON_3 = %lld.%09llds\n",dsec,dnsec);

    m_reset();
    timespec_get(&sts, TIME_UTC);
    for(int i = 0;i<count;i++)
    NEON_LU_3_omp();
    timespec_get(&ets, TIME_UTC);
    dsec = ets.tv_sec - sts.tv_sec;
    dnsec = ets.tv_nsec - sts.tv_nsec;
    if(dnsec < 0){
        dsec--;
        dnsec += 1000000000ll;
    }
    printf ("NEON_3_omp = %lld.%09llds\n",dsec,dnsec);

	return 0;
}