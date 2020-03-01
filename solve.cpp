/* 
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD
 * 
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h>
#include <immintrin.h>

#ifdef _MPI_
#include <mpi.h>
#endif

using namespace std;

#define FUSED 1
#define ROW(rx, px, m) (rx < (px -(m%px)) ? m/px : m/px + 1) //get number of rows in the block
#define COLUMN(ry, py, n) (ry < (py - (m%py)) ? m/py : m/py + 1) // get number of columns in the block

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
double *alloc1D(int m,int n);
void printMat2(const char mesg[], double *E, int m, int n);

extern control_block cb;

#ifdef SSE_VEC
#define STEP 2
/*If you intend to vectorize using SSE instructions, you must disable the compiler's auto-vectorizer*/
__attribute__((optimize("no-tree-vectorize")))
#else 
#define STEP 1
#endif 

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq){
    double l2norm = sumSq /  (double) ((cb.m)*(cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

void solve_single(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){

 // Simulated time is different from the integer timestep number
 double t = 0.0;
 double *E = *_E, *E_prev = *_E_prev;
 double *R_tmp = R;
 double *E_tmp = *_E;
 double *E_prev_tmp = *_E_prev;
 double mx, sumSq;
 int niter;
 int m = cb.m, n=cb.n;
 int innerBlockRowStartIndex = (n+2)+1;
 int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);

#ifdef SSE_VEC
__m256d alpha_avx = _mm256_set1_pd(alpha);
__m256d four_avx = _mm256_set1_pd(4);
__m256d one_avx = _mm256_set1_pd(1);
__m256d minus_avx = _mm256_set1_pd(-1);
__m256d M1_avx = _mm256_set1_pd(M1);
__m256d M2_avx = _mm256_set1_pd(M2);
__m256d a_avx = _mm256_set1_pd(a);
__m256d b_avx = _mm256_set1_pd(b);
__m256d dt_avx = _mm256_set1_pd(dt);
__m256d kk_avx = _mm256_set1_pd(kk);
__m256d epsilon_avx = _mm256_set1_pd(epsilon);
#endif

 // We continue to sweep over the mesh until the simulation has reached
 // the desired number of iterations
  for (niter = 0; niter < cb.niters; niter++){
  
      if  (cb.debug && (niter==0)){
	  stats(E_prev,m,n,&mx,&sumSq);
          double l2norm = L2Norm(sumSq);
	  repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
	  if (cb.plot_freq)
	      plotter->updatePlot(E,  -1, m+1, n+1);
      }

   /* 
    * Copy data from boundary of the computational box to the
    * padding region, set up for differencing computational box's boundary
    *
    * These are physical boundary conditions, and are not to be confused
    * with ghost cells that we would use in an MPI implementation
    *
    * The reason why we copy boundary conditions is to avoid
    * computing single sided differences at the boundaries
    * which increase the running time of solve()
    *
    */
    
    // 4 FOR LOOPS set up the padding needed for the boundary conditions
    int i,j;

    // Fills in the TOP Ghost Cells
    for (i = 0; i < (n+2); i++) {
        E_prev[i] = E_prev[i + (n+2)*2];
    }

    // Fills in the RIGHT Ghost Cells
    for (i = (n+1); i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i-2];
    }

    // Fills in the LEFT Ghost Cells
    for (i = 0; i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i+2];
    }	

    // Fills in the BOTTOM Ghost Cells
    for (i = ((m+2)*(n+2)-(n+2)); i < (m+2)*(n+2); i++) {
        E_prev[i] = E_prev[i - (n+2)*2];
    }

//////////////////////////////////////////////////////////////////////////////
#ifdef FUSED
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) 
    {
        E_tmp = E + j;
	    E_prev_tmp = E_prev + j;
        R_tmp = R + j;
	    for(i = 0; i < n - n%STEP; i+=STEP) 
        {
            #ifdef SSE_VEC
                __m256d EC_avx = _mm256_loadu_pd(&E_prev_tmp[i]); //current
                __m256d ET_avx = _mm256_loadu_pd(&E_prev_tmp[i - (n+2)]); //top
                __m256d EB_avx = _mm256_loadu_pd(&E_prev_tmp[i + (n+2)]); //bottom
                __m256d EL_avx = _mm256_loadu_pd(&E_prev_tmp[i - 1]); //left
                __m256d ER_avx = _mm256_loadu_pd(&E_prev_tmp[i + 1]); //right
                __m256d E_avx = _mm256_add_pd(EC_avx, _mm256_mul_pd(alpha_avx,
                            _mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(ET_avx, EB_avx), _mm256_add_pd(EL_avx, ER_avx)), _mm256_mul_pd(four_avx, EC_avx))));
                __m256d R_avx = _mm256_loadu_pd(&R_tmp[i]);
                E_avx = _mm256_sub_pd(E_avx, _mm256_mul_pd(dt_avx,_mm256_add_pd(_mm256_mul_pd(EC_avx, R_avx),
                        _mm256_mul_pd(_mm256_mul_pd(_mm256_sub_pd(EC_avx, a_avx),_mm256_sub_pd(EC_avx, one_avx)),_mm256_mul_pd(kk_avx, EC_avx)))));
                R_avx = _mm256_add_pd(R_avx,_mm256_mul_pd(dt_avx,_mm256_mul_pd(_mm256_add_pd(epsilon_avx,
                        _mm256_div_pd(_mm256_mul_pd(M1_avx, R_avx), _mm256_add_pd(EC_avx, M2_avx))), _mm256_sub_pd(_mm256_mul_pd(minus_avx, R_avx),
                        _mm256_mul_pd(kk_avx, _mm256_mul_pd(EC_avx, _mm256_sub_pd(EC_avx, _mm256_add_pd(b_avx, one_avx))))))));
                _mm256_storeu_pd(&E_tmp[i], E_avx);
                _mm256_storeu_pd(&R_tmp[i], R_avx);
            #else
	        E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
            #endif
        }
        for (i=n-n%STEP; i<n; i++)
        {
            E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#else
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) 
    {
        E_tmp = E + j;
        E_prev_tmp = E_prev + j;
        for(i = 0; i < n-n%STEP; i+=STEP) 
        {
            #ifdef SSE_VEC
                __m256d EC_avx = _mm256_loadu_pd(&E_prev_tmp[i]);
                __m256d ET_avx = _mm256_loadu_pd(&E_prev_tmp[i - (n+2)]);
                __m256d EB_avx = _mm256_loadu_pd(&E_prev_tmp[i + (n+2)]);
                __m256d EL_avx = _mm256_loadu_pd(&E_prev_tmp[i - 1]);
                __m256d ER_avx = _mm256_loadu_pd(&E_prev_tmp[i + 1]);
                __m256d E_avx = _mm256_add_pd(EC_avx, _mm256_mul_pd(alpha_avx, _mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(ET_avx, EB_avx),
                                        _mm256_add_pd(EL_avx, ER_avx)), _mm256_mul_pd(four_avx, EC_avx))));
            _mm256_storeu_pd(&E_tmp[i], E_avx);
            #else
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            #endif
        }
        for (i = n-n%STEP; i<n; i++)
        {
            E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
        }
    }

    /* 
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */

    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) 
    {
        E_tmp = E + j;
        R_tmp = R + j;
	    E_prev_tmp = E_prev + j;
        for(i = 0; i < n-n%STEP; i+=STEP) 
        {
            #ifdef SSE_VEC
                __m256d E_avx = _mm256_loadu_pd(&E_tmp[i]);
                __m256d EC_avx = _mm256_loadu_pd(&E_prev_tmp[i]);
                __m256d R_avx = _mm256_loadu_pd(&R_tmp[i]);
                E_avx = _mm256_sub_pd(E_avx,_mm256_mul_pd(dt_avx,_mm256_add_pd(_mm256_mul_pd(EC_avx, R_avx),
                _mm256_mul_pd(_mm256_mul_pd(_mm256_sub_pd(EC_avx, a_avx),_mm256_sub_pd(EC_avx, one_avx)),_mm256_mul_pd(kk_avx, EC_avx)))));
                R_avx = _mm256_add_pd(R_avx,_mm256_mul_pd(dt_avx,_mm256_mul_pd(_mm256_add_pd(epsilon_avx,_mm256_div_pd(_mm256_mul_pd(M1_avx, R_avx),
                                    _mm256_add_pd(EC_avx, M2_avx))),_mm256_sub_pd(_mm256_mul_pd(minus_avx, R_avx),
                                    _mm256_mul_pd(kk_avx,_mm256_mul_pd(EC_avx,_mm256_sub_pd(EC_avx, _mm256_add_pd(b_avx, one_avx))))))));
                _mm256_storeu_pd(&E_tmp[i], E_avx);
                _mm256_storeu_pd(&R_tmp[i], R_avx);
            #else
                E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
                R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
            #endif
        }
        for (i=n-n%STEP; i<n; i++)
        {
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#endif
     /////////////////////////////////////////////////////////////////////////////////

   if (cb.stats_freq){
     if ( !(niter % cb.stats_freq)){
        stats(E,m,n,&mx,&sumSq);
        double l2norm = L2Norm(sumSq);
        repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
    }
   }

   if (cb.plot_freq){
          if (!(niter % cb.plot_freq)){
	    plotter->updatePlot(E,  niter, m, n);
        }
    }

   // Swap current and previous meshes
   double *tmp = E; E = E_prev; E_prev = tmp;

 } //end of 'niter' loop at the beginning

  //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters

  stats(E_prev,m,n,&Linf,&sumSq);
  L2 = L2Norm(sumSq);

  // Swap pointers so we can re-use the arrays
  *_E = E;
  *_E_prev = E_prev;
}

void solve_MPI(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf)
{

 // Simulated time is different from the integer timestep number
    double t = 0.0;

    double *E = *_E, *E_prev = *_E_prev;
    double *R_tmp = R;
    double *E_tmp = *_E;
    double *E_prev_tmp = *_E_prev;
    double mx, sumSq;
    int niter;
    int m = cb.m, n=cb.n, px=cb.px, py=cb.py;
    int rx, ry, rows, cols, index, i, j;
    bool noComm=cb.noComm;
    
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    int root = 0; 

    rx = myrank / py;
    ry = myrank % py;
    rows = ROW(rx, px, m)+2; // 2 side ghost
    cols = COLUMN(ry, py, n)+2; // 2 side ghost

    int innerBlockRowStartIndex = cols+1;
    int innerBlockRowEndIndex = (rows*cols - 1)- (cols-2) -(cols);

    #ifdef SSE_VEC
    __m256d alpha_avx = _mm256_set1_pd(alpha);
    __m256d four_avx = _mm256_set1_pd(4);
    __m256d one_avx = _mm256_set1_pd(1);
    __m256d minus_avx = _mm256_set1_pd(-1);
    __m256d M1_avx = _mm256_set1_pd(M1);
    __m256d M2_avx = _mm256_set1_pd(M2);
    __m256d a_avx = _mm256_set1_pd(a);
    __m256d b_avx = _mm256_set1_pd(b);
    __m256d dt_avx = _mm256_set1_pd(dt);
    __m256d kk_avx = _mm256_set1_pd(kk);
    __m256d epsilon_avx = _mm256_set1_pd(epsilon);
#endif


    double *send_west_ghost = alloc1D(rows, 1); 
    double *recv_west_ghost = alloc1D(rows, 1); 
    double *send_east_ghost = alloc1D(rows, 1); 
    double *recv_east_ghost = alloc1D(rows, 1);    

    for (niter = 0; niter < cb.niters; niter++)
    {
  
        if  (cb.debug && (niter==0))
        {
        stats(E_prev,m,n,&mx,&sumSq);
        double l2norm = L2Norm(sumSq);
        repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
        if (cb.plot_freq)
            plotter->updatePlot(E,  -1, m+1, n+1);
        }
        if (noComm)
        {
            //do nothing
        }
        else
        {
            MPI_Request recv_request[4];
            MPI_Request send_request[4];
            MPI_Status send_status[4];
            MPI_Status recv_status[4];
            //west ghost
            if (ry>0) 
            {
                int src = myrank-1;
                MPI_Irecv(recv_west_ghost, rows, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &recv_request[0]);

                for (index=1; index<rows*cols; index+=cols)
                {
                    send_west_ghost[(index-1)/cols]= E_prev[index];
                }
                int dest = myrank -1;
                MPI_Isend(send_west_ghost, rows, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_request[0]);
            }
            // east ghost 
            if (ry<py-1)
            {
                int dest = myrank+1;
                for (index = cols; index < cols * rows; index += cols) 
                {
                    send_east_ghost[(index-cols)/cols] = E_prev[index];
                }
                MPI_Isend(send_east_ghost, rows, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_request[1]);

                int src = myrank+1;
                MPI_Irecv(recv_east_ghost, rows, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &recv_request[1]);
            }
            // north ghost
            if (rx > 0) 
            {
                int dest =  myrank - py;
                MPI_Isend(E_prev + innerBlockRowStartIndex - 1, cols, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_request[2]);

                int src = myrank - py;
                MPI_Irecv(E_prev, cols, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &recv_request[2]);
            
            }
            // south ghost
            /*BOTTOM: send/rcv*/
            if (rx < px-1) 
            {
                int src = myrank + py;
                MPI_Irecv(E_prev + (innerBlockRowEndIndex-1) + cols, cols, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &recv_request[3]);
                

                int dest = myrank + py;
                MPI_Isend(E_prev + innerBlockRowEndIndex - 1, cols, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_request[3]);
            }

            // west wait
            if (ry > 0) 
            {
                MPI_Wait(&recv_request[0], &recv_status[0]);
                MPI_Wait(&send_request[0], &send_status[0]);
            }
            // east wait
            if (ry < py-1) 
            {
                MPI_Wait(&recv_request[1], &recv_status[1]);
                MPI_Wait(&send_request[1], &send_status[1]);
            }

            // north wait
            if (rx >0) 
            {
                MPI_Wait(&send_request[2], &send_status[2]);
                MPI_Wait(&recv_request[2], &recv_status[2]);
            }

            // south wait
            if (rx < px-1) 
            {
                MPI_Wait(&send_request[3], &send_status[3]);
                MPI_Wait(&recv_request[3], &recv_status[3]);
            }

            // west fill
            if (ry == 0 )
            {
                for (index = 0; index < rows * cols; index+=cols) 
                {
                    E_prev[index] = E_prev[index+2];
                }
            }
            else
            {
                for (index = 0; index < rows * cols; index+=cols) 
                {
                    E_prev[index] = recv_west_ghost[index/cols];
                }

            }

            // east fill
            if (ry==py-1)
            {
                for (index = cols-1; index < rows * cols; index+= cols) 
                {
                    E_prev[index] = E_prev[index-2];
                }
            }
            else
            {
                for (index = cols-1; index < rows * cols; index+=cols) 
                {
                    E_prev[index] = recv_east_ghost[(index -cols +1)/cols];
                }
            }

            // north fill
            if (rx == 0)
            {
                for (index = 0; index < cols; index++) 
                {
                    E_prev[index] = E_prev[index + 2*cols];
                }
            }// else received 

            // south fill
            if (rx == px-1)
            {
                for (index = rows * cols - cols; index < rows * cols; index++) 
                {
                    E_prev[index] = E_prev[index - cols*2];
                }
            }//else received
      }
#ifdef FUSED
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=cols)
    {
        E_tmp = E + j;
	    E_prev_tmp = E_prev + j;
        R_tmp = R + j;
	    for(i = 0; i < (cols-2)-(cols-2)%STEP; i+=STEP) 
        {
            #ifdef SSE_VEC
                __m256d EC_avx = _mm256_loadu_pd(&E_prev_tmp[i]); //current
                __m256d ET_avx = _mm256_loadu_pd(&E_prev_tmp[i - cols]); //top
                __m256d EB_avx = _mm256_loadu_pd(&E_prev_tmp[i + cols]); //bottom
                __m256d EL_avx = _mm256_loadu_pd(&E_prev_tmp[i - 1]); //left
                __m256d ER_avx = _mm256_loadu_pd(&E_prev_tmp[i + 1]); //right
                __m256d E_avx = _mm256_add_pd(EC_avx, _mm256_mul_pd(alpha_avx,
                            _mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(ET_avx, EB_avx), _mm256_add_pd(EL_avx, ER_avx)), _mm256_mul_pd(four_avx, EC_avx))));
                __m256d R_avx = _mm256_loadu_pd(&R_tmp[i]);
                E_avx = _mm256_sub_pd(E_avx, _mm256_mul_pd(dt_avx,_mm256_add_pd(_mm256_mul_pd(EC_avx, R_avx),
                        _mm256_mul_pd(_mm256_mul_pd(_mm256_sub_pd(EC_avx, a_avx),_mm256_sub_pd(EC_avx, one_avx)),_mm256_mul_pd(kk_avx, EC_avx)))));
                R_avx = _mm256_add_pd(R_avx,_mm256_mul_pd(dt_avx,_mm256_mul_pd(_mm256_add_pd(epsilon_avx,
                        _mm256_div_pd(_mm256_mul_pd(M1_avx, R_avx), _mm256_add_pd(EC_avx, M2_avx))), _mm256_sub_pd(_mm256_mul_pd(minus_avx, R_avx),
                        _mm256_mul_pd(kk_avx, _mm256_mul_pd(EC_avx, _mm256_sub_pd(EC_avx, _mm256_add_pd(b_avx, one_avx))))))));
                _mm256_storeu_pd(&E_tmp[i], E_avx);
                _mm256_storeu_pd(&R_tmp[i], R_avx);
            #else
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+cols]+E_prev_tmp[i-cols]);
                E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
                R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
            #endif
        }
        for (i=(cols-2)-(cols-2)%STEP; i<cols-2; i++)
        {
            E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+cols]+E_prev_tmp[i-cols]);
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#else
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=cols) 
    {
        E_tmp = E + j;
        E_prev_tmp = E_prev + j;
        for(i = 0; i < (cols-2)-(cols-2)%STEP; i+=STEP) 
        {
            #ifdef SSE_VEC
                __m256d EC_avx = _mm256_loadu_pd(&E_prev_tmp[i]);
                __m256d ET_avx = _mm256_loadu_pd(&E_prev_tmp[i - cols]);
                __m256d EB_avx = _mm256_loadu_pd(&E_prev_tmp[i + cols]);
                __m256d EL_avx = _mm256_loadu_pd(&E_prev_tmp[i - 1]);
                __m256d ER_avx = _mm256_loadu_pd(&E_prev_tmp[i + 1]);
                __m256d E_avx = _mm256_add_pd(EC_avx, _mm256_mul_pd(alpha_avx, _mm256_sub_pd(_mm256_add_pd(_mm256_add_pd(ET_avx, EB_avx),
                                        _mm256_add_pd(EL_avx, ER_avx)), _mm256_mul_pd(four_avx, EC_avx))));
            _mm256_storeu_pd(&E_tmp[i], E_avx);
            #else
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+cols]+E_prev_tmp[i-cols]);
            #endif
        }
        for (i= (cols-2)-(cols-2)%STEP; i<cols-2; i++)
        {
            E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+cols]+E_prev_tmp[i-cols]);
        }
    }
    /* 
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */

    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=cols) 
    {
        E_tmp = E + j;
        R_tmp = R + j;
	    E_prev_tmp = E_prev + j;
        for(i = 0; i < (cols-2)-(cols-2)%STEP; i+=STEP) 
        {
            #ifdef SSE_VEC
                __m256d E_avx = _mm256_loadu_pd(&E_tmp[i]);
                __m256d EC_avx = _mm256_loadu_pd(&E_prev_tmp[i]);
                __m256d R_avx = _mm256_loadu_pd(&R_tmp[i]);
                E_avx = _mm256_sub_pd(E_avx,_mm256_mul_pd(dt_avx,_mm256_add_pd(_mm256_mul_pd(EC_avx, R_avx),
                _mm256_mul_pd(_mm256_mul_pd(_mm256_sub_pd(EC_avx, a_avx),_mm256_sub_pd(EC_avx, one_avx)),_mm256_mul_pd(kk_avx, EC_avx)))));
                R_avx = _mm256_add_pd(R_avx,_mm256_mul_pd(dt_avx,_mm256_mul_pd(_mm256_add_pd(epsilon_avx,_mm256_div_pd(_mm256_mul_pd(M1_avx, R_avx),
                                    _mm256_add_pd(EC_avx, M2_avx))),_mm256_sub_pd(_mm256_mul_pd(minus_avx, R_avx),
                                    _mm256_mul_pd(kk_avx,_mm256_mul_pd(EC_avx,_mm256_sub_pd(EC_avx, _mm256_add_pd(b_avx, one_avx))))))));
                _mm256_storeu_pd(&E_tmp[i], E_avx);
                _mm256_storeu_pd(&R_tmp[i], R_avx);
            #else
                E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
                R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
            #endif
        }
        for (i= (cols-2)-(cols-2)%STEP; i<cols-2; i++)
        {
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#endif       
/////////////////////////////////////////////////////////////////////////////////

        if (cb.stats_freq)
        {
            if ( !(niter % cb.stats_freq))
                {
                    stats(E,m,n,&mx,&sumSq);
                    double l2norm = L2Norm(sumSq);
                    repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
                }
        }

        if (cb.plot_freq)
        {
                if (!(niter % cb.plot_freq))
                {
                    plotter->updatePlot(E,  niter, m, n);
                }
            }

        // Swap current and previous meshes
        double *tmp = E; E = E_prev; E_prev = tmp;
    } //end of 'niter' loop at the beginning

  //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters

    stats(E_prev,m,n,&Linf,&sumSq);
    if (noComm)
    {
        //do nothing
    }
    else{
            double _Linf, _sumSq;
            MPI_Reduce(&Linf, &_Linf, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);
            MPI_Reduce(&sumSq, &_sumSq, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
            Linf = _Linf;
            sumSq = _sumSq;
    }
    L2 = L2Norm(sumSq);

    // Swap pointers so we can re-use the arrays
    *_E = E;
    *_E_prev = E_prev;
}

void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf) 
{
#ifdef _MPI_
  solve_MPI(_E ,_E_prev, R, alpha, dt, plotter, L2, Linf);
#else
  solve_single(_E ,_E_prev, R, alpha, dt, plotter, L2, Linf);
#endif
}

void printMat2(const char mesg[], double *E, int m, int n){
    int i;
#if 0
    if (m>8)
      return;
#else
    if (m>34)
      return;
#endif
    printf("%s\n",mesg);
    for (i=0; i < (m+2)*(n+2); i++){
       int rowIndex = i / (n+2);
       int colIndex = i % (n+2);
       if ((colIndex>0) && (colIndex<n+1))
          if ((rowIndex > 0) && (rowIndex < m+1))
            printf("%6.3f ", E[i]);
       if (colIndex == n+1)
	    printf("\n");
    }
}