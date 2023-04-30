#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  float tmp[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    __m256 ivec = _mm256_set1_ps(i);
    __m256 jvec = _mm256_set_ps(0, 1, 2, 3, 4, 5, 6, 7);

    __m256 mask = _mm256_cmp_ps(ivec, jvec, _CMP_NEQ_OQ);

    __m256 xivec = _mm256_set1_ps(x[i]);
    __m256 xvec = _mm256_load_ps(x);

    __m256 yivec = _mm256_set1_ps(y[i]);
    __m256 yvec = _mm256_load_ps(y);

    __m256 rxvec = _mm256_sub_ps(xivec, xvec);
    __m256 ryvec = _mm256_sub_ps(yivec, yvec);

    __m256 rx2vec = _mm256_mul_ps(rxvec, rxvec);
    __m256 ry2vec = _mm256_mul_ps(ryvec, ryvec);

    __m256 rsqrtvec = _mm256_add_ps(rx2vec, ry2vec);
    rsqrtvec = _mm256_rsqrt_ps(rsqrtvec);
    __m256 rsqrt3vec = _mm256_mul_ps(rsqrtvec, rsqrtvec);
    rsqrt3vec = _mm256_mul_ps(rsqrt3vec, rsqrtvec);

    // Replace here with Intrinsic Operations
    __m256 fxvec = _mm256_load_ps(fx);
    __m256 fyvec = _mm256_load_ps(fy);

    __m256 mvec = _mm256_load_ps(m);

    __m256 fxvecdiff = _mm256_mul_ps(rxvec, mvec);
    __m256 fyvecdiff = _mm256_mul_ps(ryvec, mvec);

    fxvecdiff = _mm256_mul_ps(fxvec, rsqrt3vec);
    fyvecdiff = _mm256_mul_ps(fyvec, rsqrt3vec);

    fxvec = _mm256_sub_ps(fxvec, fxvecdiff);
    fyvec = _mm256_sub_ps(fyvec, fyvecdiff);

    __m256 zerovec = _mm256_setzero_ps();
    fxvec = _mm256_blendv_ps(fxvec, zerovec, mask);
    fyvec = _mm256_blendv_ps(fyvec, zerovec, mask);

    _mm256_store_ps(fx, fxvec);
    _mm256_store_ps(fy, fyvec);

    //for(int j=0; j<N; j++) {
    //  if(i != j) {
    //    float rx = x[i] - x[j];
    //    float ry = y[i] - y[j];
    //    float r = std::sqrt(rx * rx + ry * ry);
    //    fx[i] -= rx * m[j] / (r * r * r);
    //    fy[i] -= ry * m[j] / (r * r * r);
    //  }
    //}
    printf("%d %g %g\n",i,fx[i],fy[i]);

    for(int j=0; j<N; j++) {
        printf("- %d %g %g\n",j,fx[j],fy[j]);
    }
  }
}
