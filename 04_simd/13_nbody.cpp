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
    __m256 jvec = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);

    __m256 mask;

    _mm256_store_ps(tmp, ivec);
    for(int j=0; j<N; j++) {
        printf("%d ivec %d %g\n", i, j, tmp[j]);
    }

    _mm256_store_ps(tmp, jvec);
    for(int j=0; j<N; j++) {
        printf("%d jvec %d %g\n", i, j, tmp[j]);
    }

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

    fxvecdiff = _mm256_mul_ps(fxvecdiff, rsqrt3vec);
    fyvecdiff = _mm256_mul_ps(fyvecdiff, rsqrt3vec);

    __m256 zerovec = _mm256_setzero_ps();
    mask = _mm256_cmp_ps(ivec, jvec, _CMP_EQ_OQ);

    fxvecdiff = _mm256_blendv_ps(fxvecdiff, zerovec, mask);
    fyvecdiff = _mm256_blendv_ps(fyvecdiff, zerovec, mask);

    __m256 fxvecdiff2 = _mm256_permute2f128_ps(fxvecdiff,fxvecdiff,1);
    fxvecdiff2 = _mm256_add_ps(fxvecdiff2,fxvecdiff);
    fxvecdiff2 = _mm256_hadd_ps(fxvecdiff2,fxvecdiff2);
    fxvecdiff2 = _mm256_hadd_ps(fxvecdiff2,fxvecdiff2);

    __m256 fyvecdiff2 = _mm256_permute2f128_ps(fyvecdiff,fyvecdiff,1);
    fyvecdiff2 = _mm256_add_ps(fyvecdiff2,fyvecdiff);
    fyvecdiff2 = _mm256_hadd_ps(fyvecdiff2,fyvecdiff2);
    fyvecdiff2 = _mm256_hadd_ps(fyvecdiff2,fyvecdiff2);

    _mm256_store_ps(tmp, fxvec);
    for(int j=0; j<N; j++) {
        printf("fxvec0 %d %g\n",j, tmp[j]);
    }

    __m256 fxveci = _mm256_sub_ps(fxvec, fxvecdiff2);
    __m256 fyveci = _mm256_sub_ps(fyvec, fyvecdiff2);

    _mm256_store_ps(tmp, fxveci);
    for(int j=0; j<N; j++) {
        printf("%d fxveci %d %g\n", i, j, tmp[j]);
    }

    fxvec = _mm256_blendv_ps(fxvec, fxveci, mask);
    fyvec = _mm256_blendv_ps(fyvec, fyveci, mask);

    _mm256_store_ps(tmp, fxvec);
    for(int j=0; j<N; j++) {
        printf("fxvec1 %d %g\n",j, tmp[j]);
    }

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
