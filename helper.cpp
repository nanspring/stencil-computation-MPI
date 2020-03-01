/* 
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
// Needed for memalign
#include <malloc.h>
#include "cblock.h"
#ifdef _MPI_
#include <mpi.h>
#endif

using namespace std;

#define ROW(rx, px, m) (rx < (px -(m%px)) ? m/px : m/px + 1) //get number of rows in the block
#define COLUMN(ry, py, n) (ry < (py - (n%py)) ? n/py : n/py + 1) // get number of columns in the block

#define ROW_INDEX(rx, px, m) (rx< (px -(m%px)) ? m/px*rx : m/px*rx+rx-(px-m%px)) // get number of start row index in the block
#define COLUMN_INDEX(ry, py, n) (ry < (py-(n%py)) ? n/py*ry : n/py*ry+ry-(py-n%py)) // get number of start column index in the block

void printMat(const char mesg[], double *E, int m, int n);
double *alloc1D(int m,int n);
extern control_block cb;

//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
void init (double *E,double *E_prev,double *R,int m,int n){
    int i;

    for (i=0; i < (m+2)*(n+2); i++)
        E_prev[i] = R[i] = 0;

    for (i = (n+2); i < (m+1)*(n+2); i++) {
	int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

        // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
	if(colIndex == 0 || colIndex == (n+1) || colIndex < ((n+1)/2+1))
	    continue;

        E_prev[i] = 1.0;
    }

    for (i = 0; i < (m+2)*(n+2); i++) {
	int rowIndex = i / (n+2);		// gives the current row number in 2D array representation
	int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

        // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
	if(colIndex == 0 || colIndex == (n+1) || rowIndex < ((m+1)/2+1))
	    continue;

        R[i] = 1.0;
    }
    // We only print the meshes if they are small enough
    #if 0
        printMat("E_prev",E_prev,m,n);
        printMat("R",R,m,n);
    #endif

#ifdef _MPI_
    int px=cb.px, py=cb.py; //
    int nprocs, myrank, rx, ry, rows, cols;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    if (myrank == 0) 
    {
        for (int rank = nprocs-1; rank >=0 ; rank--)
        {
            rx = rank / py;
            ry = rank % py;
            rows = ROW(rx, px, m)+2; // 2 side ghost
            cols = COLUMN(ry, py, n)+2; // 2 side ghost

            double *subE = alloc1D(rows, cols);
            double *subE_prev = alloc1D(rows, cols);
            double *subR = alloc1D(rows, cols);

            int start_row = ROW_INDEX(rx, px, m); 
            int start_col = COLUMN_INDEX(ry, py, n);
            for (int i=0; i<rows; i++)
                for (int j=0; j<cols; j++)
                {
                    int index = (start_row+i)*(n+2)+(start_col+j); // cancel side effect 
                    subE[i*cols+j] = E[index];
                    subE_prev[i*cols+j] = E_prev[index];
                    subR[i*cols+j] = R[index];
                }
            MPI_Request send_request[3];
            MPI_Status send_status[3];

            if (rank != 0) 
            {
                int dest = rank;
                MPI_Isend(subE, rows * cols, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_request[0]);
                MPI_Isend(subE_prev, rows * cols, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_request[1]);
                MPI_Isend(subR, rows * cols, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_request[2]);

                MPI_Wait(&send_request[0], &send_status[0]);
                MPI_Wait(&send_request[1], &send_status[1]);
                MPI_Wait(&send_request[2], &send_status[2]);
            }
            else
            {
                for (int i = 0; i <rows; i++) 
                    for (int j = 0; j < cols; j++) 
                    {
                        E[i*cols+j] = subE[i*cols+j];
                        E_prev[i*cols+j] = subE_prev[i*cols+j];
                        R[i*cols+j] = subR[i*cols+j];
                    }
            }
        }
    }
    else
    {
        rx = myrank / py;
        ry = myrank % py;
        rows = ROW(rx, px, m)+2; // 2 side ghost
        cols = COLUMN(ry, py, n)+2; // 2 side ghost

        MPI_Request recv_request[3];
        MPI_Status recv_status[3];

        int src = 0;
        MPI_Irecv(E, rows * cols, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &recv_request[0]);
        MPI_Irecv(E_prev, rows * cols, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &recv_request[1]);
        MPI_Irecv(R, rows * cols, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &recv_request[2]);
        MPI_Wait(&recv_request[0], &recv_status[0]);
        MPI_Wait(&recv_request[1], &recv_status[1]);
        MPI_Wait(&recv_request[2], &recv_status[2]);
    }
#endif

}

double *alloc1D(int m,int n){
    int nx=n, ny=m;
    double *E;
    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E= (double*) memalign(16, sizeof(double)*nx*ny) );
    return(E);
}

void printMat(const char mesg[], double *E, int m, int n){
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
