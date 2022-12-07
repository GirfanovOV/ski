#include <vector>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include <chrono>

using namespace std;

#define A_1 (double)-1
#define A_2 (double) 2
#define B_1 (double)-2
#define B_2 (double) 2

#define alpha_r 0.0
#define alpha_l 0.0
#define alpha_t 0.0

#define eps (double)1e-6

#define TAG_SYNC (int)123456
#define TAG_TAU (int)1234567


double h1;
double h2;

long M;
long N;

vector<double> x;
vector<double> y;

// k(x_i - 0.5 * h1, y_j)
double a(long i, long j) { return 4 + x[i] - 0.5 * h1; }
// k(x_i, y_j - 0.5 * h2)
double b(long i, long j) { return 4 + x[i]; }

class grid;

double w_axb_x(grid &f, long i, long j);

double w_byb_y(grid &f, size_t i, size_t j);

// (x+y)^2
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

double F(size_t i, size_t j) { return -u_lap(i, j) + q(i, j) * u(i, j); }

class grid
{
private:
    vector<vector<double> > data;
public:
    long x_start;
    long x_end;
    long y_start;
    long y_end;
    grid(long x_start, long x_end, long y_start, long y_end) :
        x_start(x_start), x_end(x_end), y_start(y_start), y_end(y_end)
    {
        data = vector<vector<double> >(x_end - x_start + 1);
        for(vector<double> &vec : data)
            vec = vector<double> (y_end - y_start + 1);

    }
    double& operator() (long i, long j) { return data[i-x_start][j-y_start]; }
    
    double lap_op(long i, long j) { return w_axb_x(*this, i, j) + w_byb_y(*this, i, j); }
    void lap_op(grid &dest)
    {
        for(long i=x_start+1; i <= x_end-1; ++i)
            for(long j=y_start+1; j <= y_end-1; ++j)
                dest(i,j) = lap_op(i,j);
    }

    double main_eq(long i, long j) { return -lap_op(i, j) + q(i,j) * (*this)(i,j); }
    void main_eq(grid &dest)
    {
        // #pragma omp parallel for
        for(long i=x_start+1; i <= x_end-1; ++i)
            for(long j=y_start+1; j <= y_end-1; ++j)
                dest(i,j) = main_eq(i,j);
    }

    double dot_prod(grid &other)
    {
        double res = 0;
        for(size_t i = x_start + (x_start != 0); i <= x_end - (x_end != M); ++i)
        {
            double col_sum = 0;
            for(size_t j = y_start + 1; j <= y_end-1; ++j)
                col_sum += (*this)(i,j) * other(i,j);
            col_sum += (*this)(i,y_end) * other(i,y_end) * (y_end == N ? 0.5 : 0.0);

            res += col_sum * ((!i || i == M) ? 0.5 : 1.0);
        }
        return res * h1 * h2;
    }

    void sub(grid &other, grid &dest)
    {
        for(int i=x_start; i<=x_end; ++i)
            for(int j=y_start; j<=y_end; ++j)
                dest(i,j) = (*this)(i,j) - other(i,j);
    }

    double max_norm()
    {
        double norm = abs((*this)(x_start+1, y_start+1));
        for(size_t i = x_start + (x_start != 0); i <= x_end - (x_end != M); ++i)
            for(size_t j = y_start + (y_start != 0); j <= y_end - (y_end != N); ++j)
                norm = max(norm, abs((*this)(i,j)));
        return norm;
    }

    bool is_in(long i, long j)
    { return (x_start <= i) && (i <= x_end) && (y_start <= j) && (j <= y_end); }
    
    void print_arr()
    {
        for(int i=x_start+(x_start != 0); i <= x_end-(x_end != M); ++i){
            for(int j=y_start+(y_start != 0); j <= y_end-(y_end != N); ++j)
                std::cout << (*this)(i,j) << " ";
            std::cout << endl;
        }

    }
    ~grid() {};
};

// Forward 1-st derivative on x
double fwd_diff_x(grid &f, long i, long j)
{ return (f(i+1,j) - f(i,j)) / h1; }

// Backward 1-st derivative on x
double bwd_diff_x(grid &f, long i, long j)
{ return fwd_diff_x(f, i-1, j); }

// Forward 1-st derivative on y
double fwd_diff_y(grid &f, long i, long j)
{ return (f(i,j+1) - f(i, j)) / h2; }

// Backward 1-st derivative on y
double bwd_diff_y(grid &f, long i, long j)
{ return fwd_diff_y(f, i, j-1); }
    
double w_axb_x(grid &f, long i, long j)
{ return (a(i+1,j) * fwd_diff_x(f,i,j) - a(i,j) * bwd_diff_x(f,i,j)) / h1; }

double w_byb_y(grid &f, size_t i, size_t j)
{ return (b(i,j+1) * fwd_diff_y(f,i,j) - b(i,j) * bwd_diff_y(f,i,j)) / h2; }

// i == M
double right_bound_eq(grid &f, size_t j)
{
    return  2 * a(M, j) * bwd_diff_x(f,M,j) / h1 +
            q(M,j) * f(M,j) - w_byb_y(f,M,j);
}

// i == 0
double left_bound_eq(grid &f, size_t j)
{
    return  -2 * a(1,j) * bwd_diff_x(f,1,j) / h1 +
            q(0,j) * f(0,j) - w_byb_y(f,0,j);
}

// j == N
double top_bound_eq(grid &f, size_t i)
{
    return  2 * b(i,N) * bwd_diff_y(f,i,N) / h2 +
            q(i,N) * f(i,N) - w_axb_x(f, i, N);
}

double bot_1_bound_eq(grid &f, size_t i)
{
    return  -w_axb_x(f,i,1) -
            (b(i,2) * bwd_diff_y(f,i,2) - b(i,1) * f(i,1) / h2) / h2 +
            q(i,1) * f(i,1);
}

// i == M & j == N
double top_r_point_eq(grid &f)
{
    return  2 * a(M,N) * bwd_diff_x(f,M,N) / h1 +
            2 * b(M,N) * bwd_diff_y(f,M,N) / h2 +
            q(M,N) * f(M,N);
}

// i == 0 & j == N
double top_l_point_eq(grid &f)
{
    return  -2 * a(1,N) * bwd_diff_x(f,1,N) / h1 +
             2 * b(0,N) * bwd_diff_y(f,0,N) / h2 +
            q(0,N) * f(0,N);
}

// i == M & j==1
double bot1_r_point_eq(grid &f)
{
    return  2 * a(M,1) * bwd_diff_x(f,M,1) / h1 +
            (q(M,1) + 2 * alpha_r / h1) * f(M,1) -
            (b(M,2) * bwd_diff_y(f,M,2) - b(M,1) * f(M,1) / h2) / h2;
}

// i == 0 & j==1
double bot1_l_point_eq(grid &f)
{
    return  -2 * a(1,1) * bwd_diff_x(f,1,1) / h1 +
            (q(0,1) + 2 * alpha_l / h1) * f(0,1) -
            (b(0,2) * bwd_diff_y(f,0,2) - b(0,1) * f(0,1) / h2) / h2;
}

void fill_B(grid &w)
{
    // B side array
    // inner
    for(int i=w.x_start+1; i <= w.x_end-1; ++i)
        for(int j=w.y_start+1; j <= w.y_end-1; ++j)
            w(i,j) = F(i, j);

    // bot + 1
    if(w.is_in(w.x_start, 1))
        for(int i=w.x_start+1; i <= w.x_end-1; ++i)
            w(i,1) = F(i, 1) + b(i, 1) * phi(i, 0) / (h2*h2);

    // right side
    if(w.is_in(M, w.y_start))
        for(int j=w.y_start+1; j <= w.y_end-1; ++j)
            w(M,j) = F(M, j) + 2 * psi(M, j) / h1;


    // left side
    if(w.is_in(0, w.y_start))
        for(int j=w.y_start; j <= w.y_end; ++j)
            w(0,j) = F(0, j) + 2 * psi(0, j) / h1;

    //top
    if(w.is_in(w.x_start, N))
        for(int i=w.x_start; i <= w.x_end; ++i)
            w(i,N) = F(i, N) + 2 * psi(i, N) / h2;

    // top_right
    if(w.is_in(M,N))
        w(M,N) = F(M, N) + (2/h1 + 2/h2) * psi(M, N);
    // top_left
    if(w.is_in(0,N))
        w(0,N) = F(0, N) + (2/h1 + 2/h2) * psi(0, N);
    // bot_1_right
    if(w.is_in(M,1))
        w(M,1) =   F(M, 1) + 2 * psi(M, 1) / h1 +
                                    b(M, 1) * phi(M, 0) / (h2*h2);
    // bot_1_left
    if(w.is_in(0,1))
        w(0,1) =   F(0, 1) + 2 * psi(0, 1) / h1 +
                                    b(0, 1) * phi(0, 0) / (h2*h2);

    // bottom
    if(w.is_in(w.x_start, 0))
        for(int i=w.x_start; i <= w.x_end; ++i)
            w(i,0) = phi(i, 0);
}

void apply_A(grid &w, grid &w1)
{
    w.main_eq(w1);

    // bot+1
    if(w.is_in(w.x_start, 1)){
        #pragma omp parallel for
        for(int i=w.x_start+1; i <= w.x_end-1; ++i)
            w1(i,1) = bot_1_bound_eq(w,i);
    }

    // right
    if(w.is_in(M, w.y_start))
        #pragma omp parallel for
        for(int j=w.y_start+1; j <= w.y_end-1; ++j)
            w1(M, j) = right_bound_eq(w, j);
    
    // left
    if(w.is_in(0, w.y_start))
        #pragma omp parallel for
        for(int j=w.y_start+1; j <= w.y_end-1; ++j)
            w1(0, j) = left_bound_eq(w, j);

    // top
    if(w.is_in(w.x_start, N))
        #pragma omp parallel for
        for(int i=w.x_start+1; i <= (w.x_end-1); ++i)
            w1(i,N) = top_bound_eq(w, i);
    
    // top_right
    if(w.is_in(M,N))
        w1(M,N) = top_r_point_eq(w);
    // top_left
    if(w.is_in(0,N))
        w1(0,N) = top_l_point_eq(w);
    // bot_right+1
    if(w.is_in(M,1))
        w1(M,1) = bot1_r_point_eq(w);
    // bot_left+1
    if(w.is_in(0,1))
        w1(0,1) = bot1_l_point_eq(w);
    
    // bottom
    if(w.is_in(w.x_start, 0))
        #pragma omp parallel for
        for(int i=w.x_start; i <= w.x_end; ++i)
            w1(i,0) = phi(i, 0);
}

void send_sync(grid &w, double *tmp, int xx, int yy, int nprocs_per_row)
{
        // sync
        if(xx != 0) {
            for(int k=w.y_start; k <= w.y_end; ++k)
                tmp[k-w.y_start] = w(w.x_start+1, k);
            int dest = (xx - 1) * nprocs_per_row + yy;
            MPI_Send(tmp, w.y_end-w.y_start+1, MPI_DOUBLE, dest, TAG_SYNC, MPI_COMM_WORLD);
        }

        // sync
        if(xx != nprocs_per_row-1) {
            for(int k=w.y_start; k <= w.y_end; ++k)
                tmp[k-w.y_start] = w(w.x_end-1, k);
            int dest = (xx + 1) * nprocs_per_row + yy;
            MPI_Send(tmp, w.y_end-w.y_start+1, MPI_DOUBLE, dest, TAG_SYNC, MPI_COMM_WORLD);
        }

        // sync
        if(yy != 0) {
            for(int k=w.x_start; k <= w.x_end; ++k)
                tmp[k-w.x_start] = w(k, w.y_start+1);
            int dest = xx * nprocs_per_row + (yy - 1);
            MPI_Send(tmp, w.x_end-w.x_start+1, MPI_DOUBLE, dest, TAG_SYNC, MPI_COMM_WORLD);
        }

        // sync
        if(yy != nprocs_per_row-1) {
            for(int k=w.x_start; k <= w.x_end; ++k)
                tmp[k-w.x_start] = w(k, w.y_end-1);
            int dest = xx * nprocs_per_row + (yy + 1);
            MPI_Send(tmp, w.x_end-w.x_start+1, MPI_DOUBLE, dest, TAG_SYNC, MPI_COMM_WORLD);
        }
}

void recv_sync(grid &w, double *tmp, int xx, int yy, int nprocs_per_row)
{
    if(xx != nprocs_per_row-1) {
        int src = (xx + 1) * nprocs_per_row + yy;
        MPI_Recv(tmp, w.y_end-w.y_start+1, MPI_DOUBLE, src, TAG_SYNC, MPI_COMM_WORLD, NULL);
        for(int k=w.y_start; k <= w.y_end; ++k)
            w(w.x_end, k) = tmp[k-w.y_start];
    }

    // sync
    if(xx != 0) {
        int src = (xx - 1) * nprocs_per_row + yy;
        MPI_Recv(tmp, w.y_end-w.y_start+1, MPI_DOUBLE, src, TAG_SYNC, MPI_COMM_WORLD, NULL);
        for(int k=w.y_start; k <= w.y_end; ++k)
            w(w.x_start, k) = tmp[k-w.y_start];
    }

    // sync
    if(yy != nprocs_per_row-1) {
        int src = xx * nprocs_per_row + (yy + 1);
        MPI_Recv(tmp, w.x_end-w.x_start+1, MPI_DOUBLE, src, TAG_SYNC, MPI_COMM_WORLD, NULL);
        for(int k=w.x_start; k <= w.x_end; ++k)
            w(k, w.y_end) = tmp[k-w.x_start];
    }

    // sync
    if(yy != 0) {
        int src = xx * nprocs_per_row + (yy - 1);
        MPI_Recv(tmp, w.x_end-w.x_start+1, MPI_DOUBLE, src, TAG_SYNC, MPI_COMM_WORLD, NULL);
        for(int k=w.x_start; k <= w.x_end; ++k)
            w(k, w.y_start) = tmp[k-w.x_start];
    }
}

int main(int argc, char **argv)
{
    auto start = std::chrono::steady_clock::now();

    if (argc != 3) return 1;
    M = atoi(argv[1]);
    N = atoi(argv[2]);

    h1 = (A_2 - A_1) / M;
    h2 = (B_2 - B_1) / N;

    x.reserve(M+1);
    for(int i=0; i <= M; ++i)
        x[i] = A_1 + i * h1;

    y.reserve(N+1);
    for(int i=0; i <= N; ++i)
        y[i] = B_1 + i * h2;

    int nprocs, rank;
    int nthreads, tid;

    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int nprocs_per_row = (int)(sqrt(nprocs));

    int x_per_proc = (int) (M / nprocs_per_row);
    int y_per_proc = (int) (N / nprocs_per_row);

    int xx = rank / nprocs_per_row;
    int yy = rank % nprocs_per_row;

    int x_start = xx * x_per_proc - (xx != 0);
    int x_end = x_start + x_per_proc + (xx != 0);
    int y_start = yy * y_per_proc - (yy != 0);
    int y_end = y_start + y_per_proc + (yy != 0);
    
    grid B(x_start, x_end, y_start, y_end);
    fill_B(B);

    grid w(x_start, x_end, y_start, y_end);
    grid Aw(x_start, x_end, y_start, y_end);
    grid w1(x_start, x_end, y_start, y_end);
    grid r(x_start, x_end, y_start, y_end);
    grid Ar(x_start, x_end, y_start, y_end);

    int cnt = x_end-x_start+10;
    double *tmp = new double [cnt];

    int max_iter = 50000;


    for(int it=0; it <= max_iter; ++it)
    {
        apply_A(w, Aw);
        Aw.sub(B, r);

        send_sync(r, tmp, xx, yy, nprocs_per_row);
        recv_sync(r, tmp, xx, yy, nprocs_per_row);
        
        apply_A(r, Ar);
        

        double Ar_x_r = Ar.dot_prod(r);
        double Ar_x_Ar = Ar.dot_prod(Ar);

        double tau;

        MPI_Allreduce(MPI_IN_PLACE, &Ar_x_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &Ar_x_Ar, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tau = Ar_x_r / Ar_x_Ar;

        for(int i=x_start; i <= x_end; ++i)
            for(int j=y_start; j <= y_end; ++j)
                w1(i,j) = w(i,j) - tau * r(i,j);        


        swap(w, w1);
        w.sub(w1, w1);

        double err = w1.max_norm();
        
        MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        
        send_sync(w, tmp, xx, yy, nprocs_per_row);
        recv_sync(r, tmp, xx, yy, nprocs_per_row);

        if(rank == 0)
            if(it % (max_iter / 10) == 0)
                std::cout << "Iter " << it << " : " << err << std::endl;
        
        if(err < eps) break;

    }

    #pragma omp parallel for collapse(2)
    for(int i=x_start; i <= x_end; ++i)
        for(int j=y_start; j <= y_end; ++j)
            w1(i,j) = u(i, j);

    double local_max = w1.max_norm();
    MPI_Allreduce(MPI_IN_PLACE, &local_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);


    if(!rank){ 
        std::chrono::duration<double> dur = std::chrono::steady_clock::now() - start;
        printf("grid: %ldx%ld, procs: %d, threads: %d, time: %f, max_err: %f\n",
                M, N, nprocs, omp_get_max_threads(), dur.count(), local_max);
    }

    MPI_Finalize();
    return 0;
}