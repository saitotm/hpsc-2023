// module load cuda gcc/8.3.0

#include<iostream>
#include<vector>
#include<chrono>

using namespace std;

const int threadBlockX = 8;
const int threadBlockY = 4;

const int threadBlockSize = threadBlockX * threadBlockY;

__global__
void kernel(const int nx, const int ny) {
    const int ix = threadBlockX * blockIdx.x + threadIdx.x;
    const int iy = threadBlockY * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        /*
         b[(j)][(i)] = rho * (1 / dt *
                 ((u[(j)][(i+1)] - u[(j)][(i-1)]) / (2 * dx) + (v[(j+1)][(i)] - v[(j-1)][(i)]) / (2 * dy)) -
                 ((u[(j)][(i+1)] - u[(j)][(i-1)]) / (2 * dx)) * ((u[(j)][(i+1)] - u[(j)][(i-1)]) / (2 * dx)) - 2 * ((u[(j+1)][(i)] - u[(j-1)][(i)]) / (2 * dy) *
                  (v[(j)][(i+1)] - v[(j)][(i-1)]) / (2 * dx)) - ((v[(j+1)][(i)] - v[(j-1)][(i)]) / (2 * dy)) * ((v[(j+1)][(i)] - v[(j-1)][(i)]) / (2 * dy)));
                  */
    }
}

int main() {
    const int nx = 41;
    const int ny = 41;
    const int nt = 500;
    const int nit = 50;
    const float dx = 2. / (nx - 1);
    const float dy = 2. / (ny - 1);
    const float dt = .01;
    const float rho = 1;
    const float nu = .02;

    vector<float> x(nx);
    vector<float> y(ny);

    for (int i = 0; i < nx; i++) {
        x[(i)] = i * dx;
    }

    for (int j = 0; j < ny; j++) {
        y[(j)] = j * dy;
    }

    vector<float> u(ny * nx, 0);
    vector<float> v(ny * nx, 0);
    vector<float> p(ny * nx, 0);
    vector<float> b(ny * nx, 0);
    vector<float> un(ny * nx, 0);
    vector<float> vn(ny * nx, 0);
    vector<float> pn(ny * nx, 0);

    for (int n = 0; n < nt; n++) {
        const auto tic = chrono::steady_clock::now();
        for (int j = 1; j < ny-1; j++) {
            for (int i = 1; i < nx - 1; i++) {
             b[(j) * nx + (i)] = rho * (1 / dt *
                     ((u[(j) * nx + (i+1)] - u[(j) * nx + (i-1)]) / (2 * dx) + (v[(j+1) * nx + (i)] - v[(j-1) * nx + (i)]) / (2 * dy)) -
                     ((u[(j) * nx + (i+1)] - u[(j) * nx + (i-1)]) / (2 * dx)) * ((u[(j) * nx + (i+1)] - u[(j) * nx + (i-1)]) / (2 * dx)) - 2 * ((u[(j+1) * nx + (i)] - u[(j-1) * nx + (i)]) / (2 * dy) *
                      (v[(j) * nx + (i+1)] - v[(j) * nx + (i-1)]) / (2 * dx)) - ((v[(j+1) * nx + (i)] - v[(j-1) * nx + (i)]) / (2 * dy)) * ((v[(j+1) * nx + (i)] - v[(j-1) * nx + (i)]) / (2 * dy)));
            }
        }

        for (int it = 0; it < nit; it++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    pn[(j) * nx + (i)] = p[(j) * nx + (i)];
                }
            }

            for (int j = 1; j < ny - 1; j++) {
                for (int i = 1; i < nx - 1; i++) {
                    p[(j) * nx + (i)] = (dy * dy * (pn[(j) * nx + (i+1)] + pn[(j) * nx + (i-1)]) +
                               dx * dx * (pn[(j+1) * nx + (i)] + pn[(j-1) * nx + (i)]) -
                               b[(j) * nx + (i)] * dx * dx * dy * dy)\
                              / (2 * (dx * dx + dy * dy));
                }
            }
            for (int j = 1; j < ny - 1; j++) {
                p[(j) * nx + (nx - 1)] = p[(j) * nx + (nx-2)];
                p[(j) * nx + (0)] = p[(j) * nx + (1)];
            }

            for (int i = 1; i < nx - 1; i++) {
                p[(0) * nx + (i)] = p[(1) * nx + (i)];
                p[(ny - 1) * nx + (i)] = 0;
            }
        }

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                un[(j) * nx + (i)] = u[(j) * nx + (i)];
                vn[(j) * nx + (i)] = v[(j) * nx + (i)];
            }
        }

        for (int j = 1; j < ny - 1; j++) {
            for (int i = 1; i < nx - 1; i++) {
                u[(j) * nx + (i)] = un[(j) * nx + (i)] - un[(j) * nx + (i)] * dt / dx * (un[(j) * nx + (i)] - un[(j) * nx + (i - 1)])
                                   - un[(j) * nx + (i)] * dt / dy * (un[(j) * nx + (i)] - un[(j - 1) * nx + (i)])
                                   - dt / (2 * rho * dx) * (p[(j) * nx + (i+1)] - p[(j) * nx + (i-1)])\
                                   + nu * dt / dx * dx * (un[(j) * nx + (i+1)] - 2 * un[(j) * nx + (i)] + un[(j) * nx + (i-1)])
                                   + nu * dt / dy * dy * (un[(j+1) * nx + (i)] - 2 * un[(j) * nx + (i)] + un[(j-1) * nx + (i)]);

                v[(j) * nx + (i)] = vn[(j) * nx + (i)] - vn[(j) * nx + (i)] * dt / dx * (vn[(j) * nx + (i)] - vn[(j) * nx + (i - 1)])
                                   - vn[(j) * nx + (i)] * dt / dy * (vn[(j) * nx + (i)] - vn[(j - 1) * nx + (i)])
                                   - dt / (2 * rho * dx) * (p[(j+1) * nx + (i)] - p[(j-1) * nx + (i)])
                                   + nu * dt / dx * dx * (vn[(j) * nx + (i+1)] - 2 * vn[(j) * nx + (i)] + vn[(j) * nx + (i-1)])
                                   + nu * dt / dy * dy * (vn[(j+1) * nx + (i)] - 2 * vn[(j) * nx + (i)] + vn[(j-1) * nx + (i)]);
            }
        }

        for (int j = 0; j < ny; j++) {
            u[(j) * nx + (0)]  = 0;
            u[(j) * nx + (nx - 1)] = 0;
            v[(j) * nx + (0)]  = 0;
            v[(j) * nx + (nx - 1)] = 0;
        }

        for (int i = 0; i < nx; i++) {
            u[(0) * nx + (i)]  = 0;
            u[(ny - 1) * nx + (i)] = 1;
            v[(0) * nx + (i)]  = 0;
            v[(ny - 1) * nx + (i)] = 0;
        }

        const auto toc = chrono::steady_clock::now();
        const double time = chrono::duration<double>(toc - tic).count();
        printf("step=%d, %lf [(s)]\n", n, time);
    }

    for (int j = 0; j < 5; j++) {
        for (int i = 0; i < 5; i++) {
            printf("u[(%d) * nx + (%d)] = %e\n", j, i, u[(j) * nx + (i)]);
            printf("v[(%d) * nx + (%d)] = %e\n", j, i, v[(j) * nx + (i)]);
        }
    }

    const dim3 grid((nx + threadBlockX - 1)/ threadBlockX, (ny + threadBlockY - 1) / threadBlockY);
    const dim3 block(threadBlockX, threadBlockY);
    //kernel<<<grid, block>>>(nx, ny);

    cudaDeviceSynchronize();

    return 0;
}

