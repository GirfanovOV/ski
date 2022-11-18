#include <cmath>
#include <iostream>
#include <cstdlib>
#include <fstream>
// #include <mpi.h>

#define A_1 (double)-1
#define A_2 (double) 2
#define B_1 (double)-2
#define B_2 (double) 2

#define alpha_r 1.0
#define alpha_l 1.0

#define eps (double)1e-6

double h1;
double h2;

double *x;
double *y;

size_t M;
size_t N;

double **corr; // right side B

double xy_sq(size_t i, size_t j) { return (x[i]+y[j])*(x[i]+y[j]); }

double u(size_t i, size_t j) { return exp(1.0 - xy_sq(i,j)); }

double u_1(size_t i, size_t j) { return -2 * (x[i]+y[j]) * u(i,j); }

double u_lap(size_t i, size_t j)
{
    return  u(i,j) * (8 * (x[i]+4) * xy_sq(i,j) -
            2 * (x[i]+y[j]) - 4 * (x[i] + 4));
}

double k(size_t i, size_t j) { return 4 + x[i]; }

double q(size_t i, size_t j) { return xy_sq(i,j); }

double phi(size_t i, size_t j) { return u(i,j); }

double psi(size_t i, size_t j) { return k(i,j) * u_1(i,j); }

double F(size_t i, size_t j) { return -u_lap(i, j) + q(i, j)*u(i, j); }

double a(size_t i, size_t j) { return k(i - 0.5 * h1, j); }

double b(size_t i, size_t j) { return k(i, j - 0.5 * h2); }

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

// i == M & j == N
double top_r_point_eq(double **w)
{
    return  2 * a(M,N) * w_x_b(w,M,N) / h1 +
            2 * b(M,N) * w_y_b(w,M,N) / h2 +
            (q(M,N) + 2 * alpha_r / h1) * w[M][N];
}

// i == 0 & j == N
double top_l_point_eq(double **w)
{
    return  -2 * a(1,N) * w_x_b(w,1,N) / h1 +
             2 * b(0,N) * w_y_b(w,0,N) / h2 +
            (q(0,N) + 2 * alpha_l / h1) * w[0][N];
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
        double col_sum = u1[i][0] * u2[i][0] * 0.5;
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

void fill_corr(double **corr)
{
    // B side array
    ////////////////////////////////////////////////////////////
    // corr = new double *[M+1];
    for(int i=0; i <= M; ++i)
        corr[i] = new double [N+1];


    //filling b
    // top + bottom
    for(int i=0; i <= M; ++i)
    {
        // bottom
        corr[i][0] = phi(i, 0);
        // bottom + 1
        corr[i][1] = F(i, 1) + b(i, 1) * phi(i, 0) / (h2*h2);
        // top
        corr[i][N] = F(i, N) + 2 * psi(i, N) / h2;
    }


    // bot_1_left
    corr[0][1] =    F(0, 1) + 2 * psi(0, 1) / h1 +
                    b(0, 1) * phi(0, 0) / (h2*h2);
    
    // bot_1_right
    corr[M][1] =    F(M, 1) + 2 * psi(M, 1) / h1 +
                    b(M, 1) * phi(M, 0) / (h2*h2);

    // top_left
    corr[0][N] = F(0, N) + (2/h1 + 2/h2) * psi(0, N);
    // top_right
    corr[M][N] = F(M, N) + (2/h1 + 2/h2) * psi(M, N);

    // sides
    for(int j=1; j <= (N-1); ++j)
    {
        //left
        corr[0][j] = F(0, j) + 2 * psi(0, j) / h1;
        //rifht
        corr[M][j] = F(M, j) + 2 * psi(M, j) / h1;
    }

    // inner
    for(int i=1; i <= (M-1); ++i)
        for(int j=2; j <= (N-1); ++j)
            corr[i][j] = F(i, j);
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
    // bottom
    for(int i=0; i <= M; ++i)
        w1[i][0] = phi(i, 0);
    
    // bot_left+1
    w1[0][1] = bot1_l_point_eq(w);
    //bot_right+1
    w1[M][1] = bot1_r_point_eq(w);
    // top_left
    w1[0][N] = top_l_point_eq(w);
    // top_right
    w1[M][N] = top_r_point_eq(w);

    // top
    for(int i=1; i <= (M-1); ++i)
        w1[i][N] = top_bound_eq(w, i);

    // left+right bounds
    for(int j=2; j <= (N-1); ++j)
    {
        //left
        w1[0][j] = left_bound_eq(w, j);
        //right
        w1[M][j] = right_bound_eq(w, j);
    }

    // inner points
    for(int i=1; i <= (M-1); ++i)
        for(int j=2; j <= (N-1); ++j)
            w1[i][j] = main_eq(w, i, j);
}

double max_norm(double **w)
{
    double norm = std::abs(w[2][2]);
    for(int i=0; i <= M; ++i)
        for(int j=0; j <= N; ++j)
            norm = std::max(norm, std::abs(w[i][j]));

    return norm;
}

int main(int argc, char **argv)
{
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
    double **corr = new double *[M+1];
    fill_corr(corr);
    
    // answ
    double **w = new double *[M+1];
    init_ndim_array(w);

    double **Aw = new double *[M+1];
    init_ndim_array(Aw);

    double **Ar = new double *[M+1];
    init_ndim_array(Ar);

    double **w_1 = new double *[M+1];
    init_ndim_array(w_1);

    double **tmp = new double *[M+1];
    init_ndim_array(tmp);

    // error array
    double **r = new double *[M+1];
    init_ndim_array(r);

    // main loop
    for(size_t it=0; it < 20; ++it)
    {
        apply_A(w, Aw); // 
        matrx_sub(Aw, corr, r);

        apply_A(r, Ar);
        double tau = dot_prod(Ar, r) / dot_prod(Ar, Ar);
        
        for(int i=0; i <= M; ++i)
            for(int j=0; j <= N; ++j)
                w_1[i][j] = w[i][j] - tau * r[i][j];

        matrx_sub(w_1, w, tmp);
        // double err = sqrt(dot_prod(tmp, tmp));
        double err = max_norm(tmp);
        std::cout << "Iter " << it << " : " << err << std::endl;
        std::swap(w, w_1);
        // if (err < eps) break;
    }

    // std::ofstream out_apporx("out_approx.txt");
    // std::ofstream out_ground("out_ground.txt");

    // for(int i=0; i <= M; ++i)
    // {
    //     for(int j=0; j <= N; ++j)
    //     {
    //         out_apporx << w[i][j] << " ";
    //         out_ground << F(i,j) << " ";
    //     }
    //     out_apporx << std::endl;
    //     out_ground << std::endl;
    // }

    return 0;
}