#include <cstdio>
#include <cstdlib>
#include <vector>

static const int gridSize = 4;
static const int threadBlockSize = 1;

__global__
void bucket_sort(int * const bucket, int * const key, const int range, const int n) {
    const int size = (n + gridDim.x - 1) / gridDim.x;
    const int offset = size * blockIdx.x;

    int * const bucket_tmp = new int[range];

    for (int i = 0; i < range; i++) {
        bucket_tmp[i] = 0;
    }


    for (int i = 0; i < size; i++) {
        const int idx = offset + i;

        if (idx < n) {
            bucket_tmp[key[idx]]++;
        }
    }

    for (int i = 0; i < range; i++) {
        atomicAdd(bucket + i, bucket_tmp[i]);
    }

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
  bucket_sort<<<gridSize, threadBlockSize>>>(bucket, key, range, n);
  cudaDeviceSynchronize();

  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }


  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
