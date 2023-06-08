#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace cooperative_groups;

static const int threadsPerBlock = 32;
static const int tileSize = 4;
static const int elmentsPerBlock = threadsPerBlock * tileSize;

__global__
void bucket_sort(int * const bucket, int * const key, const int range, const int n) {
    const int offset_idx = elmentsPerBlock * blockIdx.x + tileSize * threadIdx.x;
    const grid_group grid = this_grid();

    int * const bucket_tmp = new int[range];

    for (int i = 0; i < range; i++) {
        bucket_tmp[i] = 0;
    }

    for (int i = 0; i < tileSize; i++) {
        const int idx = offset_idx + i;

        if (idx < n) {
            bucket_tmp[key[idx]]++;
        }
    }

    for (int i = 0; i < range; i++) {
        atomicAdd(bucket + i, bucket_tmp[i]);
    }

    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i=0, j=0; i<range; i++) {
          for (; bucket[i]>0; bucket[i]--) {
            key[j++] = i;
          }
        }
    }

    delete[] bucket_tmp;
}

int main() {
  int n = 50;
  int range = 5;
  //std::vector<int> key(n);
  int *key;
  int *bucket;

  cudaMallocManaged(&key, n * sizeof(int));
  cudaMallocManaged(&bucket, range * sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  /*
  std::vector<int> bucket(range); 
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }
  */
  void *args[] = {(void *)&bucket, (void *)&key, (void *)&range, (void *)&n};
  //bucket_sort<<<(n + elmentsPerBlock - 1) / elmentsPerBlock, threadsPerBlock>>>(bucket, key, range, n);
  cudaLaunchCooperativeKernel((void *)bucket_sort, (n + elmentsPerBlock - 1) / elmentsPerBlock, threadsPerBlock, args);
  cudaDeviceSynchronize();

  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }


  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
}
