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

#ifdef _MPI_
#include <mpi.h>
#endif

using namespace std;

#define FUSED 0
#define ROW(rx, px, m) (rx < (px -(m%px)) ? m/px : m/px + 1) //get number of rows in the block
#define COLUMN(ry, py, n) (ry < (py - (m%py)) ? m/py : m/py + 1) // get number of columns in the block

#define ROW_INDEX(rx, px, m) (rx< (px -(m%px)) ? m/px*rx : m/px*rx+rx-(px-m%px)) // get number of start row index in the block
#define COLUMN_INDEX(ry, py, n) (ry < (py-(n%py)) ? n/py*ry : n/py*ry+ry-(py-n%py)) // get number of start column index in the block

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
double *alloc1D(int m,int n);
void printMat2(const char mesg[], double *E, int m, int n);

extern control_block cb;

 // The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq){
    double l2norm = sumSq /  (double) ((cb.m)*(cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

void solve_ghost(int start_idx, int end_idx, int cols, int step,double *E_tmp, double *R_tmp, double *E_prev_tmp, double *E ,double *R, double *E_prev, double dt, double alpha){
    int i,j;

    #ifdef AVX_VEC
        #pragma ivdep
    #endif
    #pragma prefetch
    for(j = start_idx; j <= end_idx; j+=cols) 
    {
        
        E_tmp = E + j;
        R_tmp = R + j;
	    E_prev_tmp = E_prev + j;
        #ifdef AVX_VEC
            #pragma ivdep
        #endif
        #pragma prefetch
        for(i = 0; i < step; i++) 
        {
            E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+cols]+E_prev_tmp[i-cols]);
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
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
	    for(i = 0; i < n; i++) 
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
        for(i = 0; i < n; i++) 
        {
            E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
        }
    }

    /* 
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */
    //solve_ode(innerBlockRowStartIndex+1, innerBlockRowEndIndex-1, cols, *E_tmp, *R_tmp, *E_prev_tmp, *E ,*R, *E_prev)
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) 
    {
        E_tmp = E + j;
        R_tmp = R + j;
	    E_prev_tmp = E_prev + j;
        for(i = 0; i < n; i++) 
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
    int m = cb.m, n=cb.n, py=cb.px, px=cb.py;
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
    int innerBlockRowEndIndex = rows*cols -2*cols+1;

    MPI_Datatype vec_t;
    MPI_Type_vector(rows, 1, cols, MPI_DOUBLE, &vec_t);
    MPI_Type_commit(&vec_t);

    double *E_plot = NULL;
    if (cb.plot_freq && (myrank==0))
    {
        E_plot = alloc1D(cb.m + 2, cb.n + 2);
    }    

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
            int count = -1;
            //west ghost
            if (ry>0) 
            {
                count ++;
                int src = myrank-1;
                MPI_Irecv(E_prev, 1, vec_t, src, 0, MPI_COMM_WORLD, &recv_request[count]);

                int dest = myrank -1;
                MPI_Isend(E_prev+1, 1, vec_t, dest, 0, MPI_COMM_WORLD, &send_request[count]);
            }
            // east ghost 
            if (ry<py-1)
            {
                count ++;
                int dest = myrank+1;
                MPI_Isend(E_prev+cols-2, 1, vec_t, dest, 0, MPI_COMM_WORLD, &send_request[count]);

                int src = myrank+1;
                MPI_Irecv(E_prev+cols-1, 1, vec_t, src, 0, MPI_COMM_WORLD, &recv_request[count]);
            }

            // north ghost
            if (rx > 0) 
            {
                count ++;
                int dest =  myrank - py;
                MPI_Isend(E_prev + cols, cols, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_request[count]);

                int src = myrank - py;
                MPI_Irecv(E_prev, cols, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &recv_request[count]);
            
            }
            // south ghost
            /*BOTTOM: send/rcv*/
            if (rx < px-1) 
            {
                count ++;
                int src = myrank + py;
                MPI_Irecv(E_prev + rows * cols -cols, cols, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &recv_request[count]);
                

                int dest = myrank + py;
                MPI_Isend(E_prev + rows * cols - 2*cols, cols, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_request[count]);
            }

            if (ry == 0 )
            {
                #ifdef AVX_VEC
                    #pragma ivdep
                #endif
                #pragma prefetch
                for (index = 0; index < rows * cols; index+=cols) 
                {
                    E_prev[index] = E_prev[index+2];
                }
            }
            if (ry==py-1)
                {
                    #ifdef AVX_VEC
                        #pragma ivdep
                    #endif
                    #pragma prefetch
                    for (index = cols-1; index < rows * cols; index+= cols) 
                    {
                        E_prev[index] = E_prev[index-2];
                    }
                }
            // north fill
            if (rx == 0)
                {
                    #ifdef AVX_VEC
                        #pragma ivdep
                    #endif
                    #pragma prefetch
                    for (index = 0; index < cols; index++) 
                    {
                        E_prev[index] = E_prev[index + 2*cols];
                    }
                }// else received 

            // south fill
            if (rx == px-1)
                {
                    #ifdef AVX_VEC
                        #pragma ivdep
                    #endif
                    #pragma prefetch
                    for (index = rows * cols - cols; index < rows * cols; index++) 
                    {
                        E_prev[index] = E_prev[index - cols*2];
                    }
                }//else received

        //do computation while sending and receiving            

            #ifdef AVX_VEC
                    #pragma ivdep
            #endif
            #pragma prefetch
            for(j = innerBlockRowStartIndex+1+cols; j <= innerBlockRowEndIndex+1-cols; j+=cols) 
            {
                E_tmp = E + j;
                E_prev_tmp = E_prev + j;
                #ifdef AVX_VEC
                    #pragma ivdep
                #endif
                #pragma prefetch
                for(i = 0; i < (cols-4); i++) 
                {          
                        E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+cols]+E_prev_tmp[i-cols]);
                }
                
            }
            /* 
            * Solve the ODE, advancing excitation and recovery variables
            *     to the next timtestep
            */
            #ifdef AVX_VEC
                    #pragma ivdep
            #endif
            #pragma prefetch
            for(j = innerBlockRowStartIndex+1+cols; j <= innerBlockRowEndIndex+1-cols; j+=cols) 
            {
                E_tmp = E + j;
                R_tmp = R + j;
                E_prev_tmp = E_prev + j;
                #ifdef AVX_VEC
                    #pragma ivdep
                #endif
                #pragma prefetch
                for(i = 0; i < (cols-4); i++) 
                {
                        E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
                        R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
                }
            }
            
            //wait must be in order as above
            
            MPI_Waitall(count+1,recv_request,recv_status);
            //MPI_Waitall(count+1,send_request,send_status);
    
            solve_ghost(innerBlockRowStartIndex, 2*cols-2,cols,cols-2, E_tmp,R_tmp, E_prev_tmp, E ,R,  E_prev,dt,alpha);
      
            solve_ghost(innerBlockRowEndIndex, rows*cols-cols-2 ,cols, cols-2,E_tmp,R_tmp, E_prev_tmp, E ,R,  E_prev,dt,alpha);
        
            solve_ghost(innerBlockRowStartIndex+cols, innerBlockRowEndIndex-cols,cols,1,E_tmp, R_tmp, E_prev_tmp,E ,R, E_prev, dt,alpha);
                  
            solve_ghost(3*cols-2, rows*cols-2*cols-2,cols,1,E_tmp, R_tmp, E_prev_tmp,E ,R, E_prev,dt,alpha);
       
        }
      
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
                    if (myrank)
                    {
                        MPI_Request send_request[1];
                        MPI_Status send_status[1];
                        int dest = 0;
                        MPI_Isend(E, rows * cols, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_request[0]);
                        MPI_Wait(&send_request[0], &send_status[0]);
                    }
                    else
                    {
                        for (int tmp_rank = nprocs-1; tmp_rank >=0 ; tmp_rank--)
                        {
                            if (tmp_rank)
                            {
                                int tmp_rx = tmp_rank / py;
                                int tmp_ry = tmp_rank % py;
                                int tmp_rows = ROW(tmp_rx, px, m)+2; // 2 side ghost
                                int tmp_cols = COLUMN(tmp_ry, py, n)+2; // 2 side ghost
                                double *subE_plot = alloc1D(tmp_rows, tmp_cols);
                                MPI_Request recv_request[1];
                                MPI_Status recv_status[1];
                                int src = tmp_rank;
                                MPI_Irecv(subE_plot, tmp_rows * tmp_cols, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &recv_request[0]);
                                MPI_Wait(&recv_request[0], &recv_status[0]);
                                int start_row = ROW_INDEX(tmp_rx, px, m); 
                                int start_col = COLUMN_INDEX(tmp_ry, py, n);
                                int i_start = (tmp_rx==0) ? 0 : 1;
                                int j_start = (tmp_ry==0) ? 0 : 1;
                                int i_stop = (tmp_rx==px-1) ? tmp_rows : tmp_rows-1;
                                int j_stop = (tmp_ry==py-1) ? tmp_cols : tmp_cols-1;
                                for (int i=i_start; i<i_stop; i++)
                                    for (int j=j_start; j<j_stop; j++)
                                    {
                                        int index = (start_row+i)*(n+2)+(start_col+j); // cancel side effect 
                                        E_plot[index] = subE_plot[i*tmp_cols+j];
                                    }
                            }
                            else
                            {
                                for (int i = 0; i <rows-1; i++) 
                                    for (int j = 0; j < cols-1; j++) 
                                    {
                                        E_plot[i*(n+2)+j] = E[i*cols+j];
                                    }
                            }  
                        }
                        plotter->updatePlot(E_plot,  niter, m, n);
                    }
                }
            }

        // Swap current and previous meshes
        double *tmp = E; E = E_prev; E_prev = tmp;
    } //end of 'niter' loop at the beginning

  //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters
    MPI_Type_free(&vec_t);
    stats(E_prev,rows-2,cols-2,&Linf,&sumSq);
    if (noComm)
    {
        //do nothing
    }
    else{
            MPI_Barrier(MPI_COMM_WORLD);
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