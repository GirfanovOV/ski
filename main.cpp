#include <cmath>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <chrono>
// #include <mpi.h>

#define A_1 (double)-1
#define A_2 (double) 2
#define B_1 (double)-2
#define B_2 (double) 2

// #define alpha_r 1.0
// #define alpha_l 1.0
// #define alpha_t 1.0

#define alpha_r 0.0
#define alpha_l 0.0
#define alpha_t 0.0

#define eps (double)1e-6

double h1;
double h2;

double *x;
double *y;

size_t M;
size_t N;

double xy_sq(size_t i, size_t j) { return (x[i]+y[j])*(x[i]+y[j]); }

double u(size_t i, size_t j) { return exp(1.0 - xy_sq(i,j)); }

double u_1(size_t i, size_t j) { return -2 * (x[i]+y[j]) * u(i,j); }

double k(size_t i, size_t j) { return 4 + x[i]; }

double u_lap(size_t i, size_t j)
{
    return  8 * k(i,j) * u(i, j) * xy_sq(i,j) - 
            2 * u(i,j) * (x[i] + y[j]) -
            4 * k(i,j) * u(i, j);
}

double q(size_t i, size_t j) { return xy_sq(i,j); }

double phi(size_t i, size_t j) { return u(i,j); }

double psi(size_t i, size_t j) { return (i ? 1 : -1) * k(i,j) * u_1(i,j); }
// double psi(size_t i, size_t j) { return k(i,j) * u_1(i,j); }

double F(size_t i, size_t j) { return -u_lap(i, j) + q(i, j)*u(i, j); }

// k(x_i - 0.5 * h1, y_j)
double a(size_t i, size_t j) { return 4 + x[i] - 0.5 * h1; }

// k(x_i, y_j - 0.5 * h2)
double b(size_t i, size_t j) { return 4 + x[i]; }

double w_x(double** w, size_t i, size_t j)
{
    return (w[i+1][j] - w[i][j]) / h1;
}

double w_y(double** w, size_t i, size_t j)
{
    return (w[i][j+1] - w[i][j]) / h2;
}

double w_x_b(double** w, size_t i, size_t j)
{
    return (w[i][j] - w[i-1][j]) / h1;
}

double w_y_b(double** w, size_t i, size_t j)
{
    return (w[i][j] - w[i][j-1]) / h2;
}

double w_axb_x(double** w, size_t i, size_t j)
{
    return (a(i+1,j) * w_x(w,i,j) - a(i,j) * w_x_b(w,i,j)) / h1;
}

double w_byb_y(double **w, size_t i, size_t j)
{
    return (b(i,j+1) * w_y(w,i,j) - b(i,j) * w_y_b(w,i,j)) / h2;
}

double lap_op(double** w, size_t i, size_t j)
{
    return w_axb_x(w, i, j) + w_byb_y(w, i, j);
}

/////////////////////////////////////////////////

double main_eq(double **w, size_t i, size_t j)
{
    return -lap_op(w, i, j) + q(i,j) * w[i][j];
}

// i == M
double right_bound_eq(double **w, size_t j)
{
    return  2 * a(M, j) * w_x_b(w,M,j) / h1 +
            q(M,j) * w[M][j] - w_byb_y(w,M,j);
}

// i == 0
double left_bound_eq(double **w, size_t j)
{
    return  -2 * a(1,j) * w_x_b(w,1,j) / h1 +
            q(0,j) * w[0][j] - w_byb_y(w,0,j);
}

// j == N
double top_bound_eq(double **w, size_t i)
{
    return  2 * b(i,N) * w_y_b(w,i,N) / h2 +
            q(i,N) * w[i][N] - w_axb_x(w, i, N);
}

double bot_1_bound_eq(double **w, size_t i)
{
    return  -w_axb_x(w,i,1) -
            (b(i,2) * w_y_b(w,i,2) - b(i,1) * w[i][1] / h2) / h2 +
            q(i,1) * w[i][1];
}

// i == M & j == N
double top_r_point_eq(double **w)
{
    return  2 * a(M,N) * w_x_b(w,M,N) / h1 +
            2 * b(M,N) * w_y_b(w,M,N) / h2 +
            q(M,N) * w[M][N];
            // (q(M,N) + 2 * alpha_r / h1 ) * w[M][N];
}

// i == 0 & j == N
double top_l_point_eq(double **w)
{
    return  -2 * a(1,N) * w_x_b(w,1,N) / h1 +
             2 * b(0,N) * w_y_b(w,0,N) / h2 +
            q(0,N) * w[0][N];
            // (q(0,N) + 2 * alpha_l / h1) * w[0][N];
            // (q(0,N) + 2 * alpha_t / h2) * w[0][N];
}

// i == M & j==1
double bot1_r_point_eq(double **w)
{
    return  2 * a(M,1) * w_x_b(w,M,1) / h1 +
            (q(M,1) + 2 * alpha_r / h1) * w[M][1] -
            (b(M,2) * w_y_b(w,M,2) - b(M,1) * w[M][1] / h2) / h2;
}

// i == 0 & j==1
double bot1_l_point_eq(double **w)
{
    return  -2 * a(1,1) * w_x_b(w,1,1) / h1 +
            (q(0,1) + 2 * alpha_l / h1) * w[0][1] -
            (b(0,2) * w_y_b(w,0,2) - b(0,1) * w[0][1] / h2) / h2;
}

/////////////////////////////////////////////////

double dot_prod(double **u1, double **u2)
{
    double res = 0;
    for(size_t i = 0; i <= M; ++i)
    {
        // double col_sum = u1[i][0] * u2[i][0] * 0.5;
        double col_sum = 0;
        for(size_t j = 1; j < N; ++j)
            col_sum += u1[i][j] * u2[i][j];
        col_sum += u1[i][N] * u2[i][N] * 0.5;

        res += col_sum * ((!i || i == M) ? 0.5 : 1.0);
    }
    return res * h1 * h2;
}

void matrx_sub(double **left, double **right, double **dest)
{
    for(int i=0; i<=M; ++i)
        for(int j=0; j<=N; ++j)
            dest[i][j] = left[i][j] - right[i][j];
}

void fill_ax(double *x, double *y)
{
    x = new double [M+1];
    for(int i=0; i <= M; ++i)
        x[i] = A_1 + i * h1;

    y = new double [N+1];
    for(int i=0; i <= N; ++i)
        y[i] = B_1 + i * h2;
}

void fill_B(double **B)
{
    // B side array
    // init
    for(int i=0; i <= M; ++i)
        B[i] = new double [N+1];
    ///////////////////////////

    // inner
    for(int i=1; i <= (M-1); ++i)
        for(int j=2; j <= (N-1); ++j)
            B[i][j] = F(i, j);

    // bot + 1
    for(int i=1; i <= M-1; ++i)
        B[i][1] = F(i, 1) + b(i, 1) * phi(i, 0) / (h2*h2);

    // sides
    for(int j=2; j <= (N-1); ++j)
    {
        //right
        B[M][j] = F(M, j) + 2 * psi(M, j) / h1;
        //left
        B[0][j] = F(0, j) + 2 * psi(0, j) / h1;
    }
    //top
    for(int i=1; i <= M-1; ++i)
        B[i][N] = F(i, N) + 2 * psi(i, N) / h2;

    // top_right
    B[M][N] = F(M, N) + (2/h1 + 2/h2) * psi(M, N);
    // top_left
    B[0][N] = F(0, N) + (2/h1 + 2/h2) * psi(0, N);
    // bot_1_right
    B[M][1] =    F(M, 1) + 2 * psi(M, 1) / h1 +
                    b(M, 1) * phi(M, 0) / (h2*h2);
    // bot_1_left
    B[0][1] =    F(0, 1) + 2 * psi(0, 1) / h1 +
                    b(0, 1) * phi(0, 0) / (h2*h2);

    // bottom
    for(int i=0; i <= M; ++i)
        B[i][0] = phi(i, 0);
}

void init_ndim_array(double **w)
{
    for(int i=0; i <= M; ++i)
        w[i] = new double [N+1];

    for(int i=0; i <= M; ++i)
        for(int j=0; j <= N; ++j)
            w[i][j] = 0;
}

void cpy_arr(double **dest, double **src)
{
    for(int i=0; i <= M; ++i)
        for(int j=0; j <= N; ++j)
            dest[i][j] = src[i][j];
}

void apply_A(double **w, double **w1)
{
    // inner points
    for(int i=1; i <= (M-1); ++i)
        for(int j=2; j <= (N-1); ++j)
            w1[i][j] = main_eq(w, i, j);

    // bot+1
    for(int i=1; i <= (M-1); ++i)
        w1[i][1] = bot_1_bound_eq(w,i);

    // right+left bounds
    for(int j=2; j <= (N-1); ++j)
    {
        //right
        w1[M][j] = right_bound_eq(w, j);
        //left
        w1[0][j] = left_bound_eq(w, j);
    }
    // top
    for(int i=1; i <= (M-1); ++i)
        w1[i][N] = top_bound_eq(w, i);
    
    // top_right
    w1[M][N] = top_r_point_eq(w);
    // top_left
    w1[0][N] = top_l_point_eq(w);
    // bot_right+1
    w1[M][1] = bot1_r_point_eq(w);
    // bot_left+1
    w1[0][1] = bot1_l_point_eq(w);
    
    // bottom
    for(int i=0; i <= M; ++i)
        w1[i][0] = phi(i, 0);
}

double max_norm(double **w)
{
    double norm = std::abs(w[0][0]);
    for(int i=0; i <= M; ++i)
        for(int j=0; j <= N; ++j)
            norm = std::max(norm, std::abs(w[i][j]));

    return norm;
}

bool test()
{
    double **t = new double* [M+1];
    for(int i=0; i <= M; ++i)
        t[i] = new double [N+1];

    for(int i=0; i <= M; ++i)
        for(int j=0; j <= N; ++j)
            t[i][j] = u(i,j);

    // std::cout << "here" << std::endl;
    double **t1 = new double* [M+1];
    for(int i=0; i <= M; ++i)
        t1[i] = new double [N+1];

    for(int i=0; i <= M; ++i)
        for(int j=0; j <= N; ++j)
            t1[i][j] =  8 * k(i, j) * u(i, j) * xy_sq(i, j) -
                        2 * u(i, j) * (x[i] + y[j]) -
                        4 * k(i, j) * u(i, j);

    double **t2 = new double* [M+1];
    for(int i =0; i <= M; ++i)
        t2[i] = new double [N+1];

    for(int i=0; i <= M; ++i)
        for(int j=0; j <= N; ++j)
            t2[i][j] = t1[i][j];

    for(int i=1; i < M; ++i)
        for(int j=1; j < N; ++j)
            t2[i][j] = lap_op(t,i,j);

    // matrx_sub(t2, t1, t);
    // std::cout << max_norm(t) << std::endl;

    for(int i=0; i <= M; ++i){
        for(int j=0; j <= N; ++j)
            std::cout << t2[i][j] - t1[i][j] << " ";
        std::cout << std::endl;
    }


    return true;
}

bool test1()
{
    double **B = new double *[M+1];
    double **w = new double *[M+1];
    double **r = new double *[M+1];
    
    for(int i=0; i <= M; ++i){
        B[i] = new double [N+1];
        w[i] = new double [N+1];
        r[i] = new double [N+1];
        for(int j=0; j <= N; ++j){
            // B[i][j] = 8 * u(i, j) * xy_sq(i,j) - 4 * u(i,j);
            B[i][j] = F(i, j);
            w[i][j] = u(i, j);
            r[i][j] = B[i][j];
        }
    }
    
    for(int i=1; i <= M-1; ++i)
        for(int j=1; j <= N-1; ++j)
            r[i][j] = main_eq(w, i, j);

    matrx_sub(B, r, w);
    std::cout << max_norm(w) << std::endl;
    return 1;
}

int main(int argc, char **argv)
{
    auto start = std::chrono::steady_clock::now();

    if (argc != 3) return 1;
    M = atoi(argv[1]);
    N = atoi(argv[2]);

    h1 = (A_2 - A_1) / M;
    h2 = (B_2 - B_1) / N;


    x = new double [M+1];
    for(int i=0; i <= M; ++i)
        x[i] = A_1 + i * h1;

    y = new double [N+1];
    for(int i=0; i <= N; ++i)
        y[i] = B_1 + i * h2;
    

    // B side arr
    double **B = new double *[M+1];
    fill_B(B);
    
    // w^k
    double **w = new double *[M+1];
    init_ndim_array(w);

    // A x (w^k)
    double **Aw = new double *[M+1];
    init_ndim_array(Aw);

    // w^{k+1}
    double **w_1 = new double *[M+1];
    init_ndim_array(w_1);

    double **r = new double *[M+1];
    init_ndim_array(r);

    double **Ar = new double *[M+1];
    init_ndim_array(Ar);


// TESTING 
///////////////////////////////////////////////////////
    for(int i=0; i <= M; ++i)
        for(int j=0; j <= N; ++j)
            w[i][j] = u(i,j);

    apply_A(w, Aw);
    // Aw[0][N] = B[0][N];
    // Aw[M][1] = B[M][1];

    for(int i=0; i <= M-0; ++i)
        for(int j=0; j <= N-0; ++j)
            w_1[i][j] = Aw[i][j] - B[i][j];


    double norm1 = std::abs(w_1[0][0]);
    int max_i1 = 0,  max_j1 = 0;
    for(int i=0; i <= M; ++i)
        for(int j=0; j <= N; ++j)
            if (norm1 < std::abs(w_1[i][j])){
                max_i1 = i;
                max_j1 = j;
                norm1 = std::abs(w_1[i][j]);
            }

    std::cout << "Final error : " << norm1 << std::endl;
    std::cout << "i = " << max_i1 << "; j = " << max_j1 << std::endl;

    // std::cout << "Final error : " <<  max_norm(w_1) << std::endl;

    std::ofstream out_apporx1("trash/out_approx.txt");
    std::ofstream out_ground1("trash/out_ground.txt");

    for(int i=0; i <= M; ++i)
    {
        for(int j = N; j >= 0; --j)
        {
            out_apporx1 << w_1[i][j] << " ";
            // out_apporx1 << w[i][j] << " ";
            out_ground1 << u(i, j) << " ";
        }
        out_apporx1 << std::endl;
        out_ground1 << std::endl;
    }

    return 0;

///////////////////////////////////////////////////////
    // main loop
    for(size_t it=0; it < 100000; ++it)
    {
        apply_A(w, Aw);
        matrx_sub(Aw, B, r);

        apply_A(r, Ar);
        double tau = dot_prod(Ar, r) / dot_prod(Ar, Ar);
        
        for(int i=0; i <= M; ++i)
            for(int j=1; j <= N; ++j)
                w_1[i][j] = w[i][j] - tau * r[i][j];

        std::swap(w, w_1);

        matrx_sub(w, w_1, w_1);
        double err = max_norm(w_1);
        // double err = sqrt(dot_prod(w_1, w_1));
        if(it % 5000 == 0)
            std::cout << "Iter " << it << " : " << err << std::endl;
        if (err < eps) break;
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << "Time taken (ms) : ";
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << std::endl;
    
    // bottom
    for(int i=0; i <= M; ++i)
        w[i][0] = phi(i, 0);


    for(int i=0; i <= M-0; ++i)
        for(int j=0; j <= N-0; ++j)
            w_1[i][j] = w[i][j] - u(i,j);

    double norm = std::abs(w_1[0][0]);
    int max_i = 0,  max_j = 0;
    for(int i=0; i <= M; ++i)
        for(int j=0; j <= N; ++j)
            if (norm < std::abs(w_1[i][j])){
                max_i = i;
                max_j = j;
                norm = std::abs(w_1[i][j]);
            }

    std::cout << "Final error : " << norm << std::endl;
    std::cout << "i = " << max_i << "; j = " << max_j << std::endl;

    // std::cout << "Final error : " <<  max_norm(w_1) << std::endl;

    std::ofstream out_apporx("trash/out_approx.txt");
    std::ofstream out_ground("trash/out_ground.txt");

    for(int i=0; i <= M; ++i)
    {
        for(int j = N; j >= 0; --j)
        {
            out_apporx << w[i][j] << " ";
            out_ground << u(i, j) << " ";
        }
        out_apporx << std::endl;
        out_ground << std::endl;
    }

    return 0;
}