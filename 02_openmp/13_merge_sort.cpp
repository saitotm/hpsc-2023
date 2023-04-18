#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <vector>

static constexpr int MIN_PARALLEL_SIZE = 1000;

void merge(std::vector<int>& vec, int begin, int mid, int end) {
  std::vector<int> tmp(end-begin+1);
  int left = begin;
  int right = mid+1;
  for (int i=0; i<tmp.size(); i++) { 
    if (left > mid)
      tmp[i] = vec[right++];
    else if (right > end)
      tmp[i] = vec[left++];
    else if (vec[left] <= vec[right])
      tmp[i] = vec[left++];
    else
      tmp[i] = vec[right++]; 
  }
  for (int i=0; i<tmp.size(); i++) 
    vec[begin++] = tmp[i];
}

void merge_sort(std::vector<int>& vec, int begin, int end) {
    //printf("thread: %d, begin: %d, end:%d\n", omp_get_thread_num(), begin, end);
  if(begin < end) {
    int mid = (begin + end) / 2;

    const int size = end - begin;
if (size >= MIN_PARALLEL_SIZE) { // Sort Parallel
#pragma omp task shared(vec, begin, mid)
    merge_sort(vec, begin, mid);

#pragma omp task shared(vec, mid, end)
    merge_sort(vec, mid+1, end);

#pragma omp taskwait
    merge(vec, begin, mid, end);
} else { // Sort Sequential
    merge_sort(vec, begin, mid);

    merge_sort(vec, mid+1, end);

    merge(vec, begin, mid, end);
}
  }
}

int main() {
  int n = 20;
  //int n = 100000;
  std::vector<int> vec(n);
  for (int i=0; i<n; i++) {
    vec[i] = rand() % (10 * n);
    printf("%d ",vec[i]);
  }
  printf("\n");
#pragma omp parallel
  {
#pragma omp master
      merge_sort(vec, 0, n-1);
  }
  for (int i=0; i<n; i++) {
    printf("%d ",vec[i]);
  }
  printf("\n");
  printf("max %d\n", omp_get_max_threads());
}
