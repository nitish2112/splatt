

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "cpd.h"
#include "matrix.h"
#include "mttkrp.h"
#include "timer.h"
#include "thd_info.h"
#include "util.h"

#include <math.h>
#include <time.h>

#define BILLION 1000000000L
#include "../args.h"

//#define DUMP

/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

int splatt_cpd_als(
    splatt_csf const * const tensors,
    splatt_idx_t const nfactors,
    double const * const options,
    splatt_kruskal * factored)
{
  matrix_t * mats[MAX_NMODES+1];

  idx_t nmodes = tensors->nmodes;

  rank_info rinfo;
  rinfo.rank = 0;

  /* allocate factor matrices */
  idx_t maxdim = tensors->dims[argmax_elem(tensors->dims, nmodes)];
  for(idx_t m=0; m < nmodes; ++m) {
    mats[m] = (matrix_t *) mat_rand(tensors[0].dims[m], nfactors);
  }
  mats[MAX_NMODES] = mat_alloc(maxdim, nfactors);

  val_t * lambda = (val_t *) splatt_malloc(nfactors * sizeof(val_t));

  /* do the factorization! */
  factored->fit = cpd_als_iterate(tensors, mats, lambda, nfactors, &rinfo,
      options);

  /* store output */
  factored->rank = nfactors;
  factored->nmodes = nmodes;
  factored->lambda = lambda;
  for(idx_t m=0; m < nmodes; ++m) {
    factored->dims[m] = tensors->dims[m];
    factored->factors[m] = mats[m]->vals;
  }

  /* clean up */
  mat_free(mats[MAX_NMODES]);
  for(idx_t m=0; m < nmodes; ++m) {
    free(mats[m]); /* just the matrix_t ptr, data is safely in factored */
  }
  return SPLATT_SUCCESS;
}


void splatt_free_kruskal(
    splatt_kruskal * factored)
{
  free(factored->lambda);
  for(idx_t m=0; m < factored->nmodes; ++m) {
    free(factored->factors[m]);
  }
}


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Resets serial and MPI timers that were activated during some CPD
*        pre-processing.
*
* @param rinfo MPI rank information.
*/
static void p_reset_cpd_timers(
  rank_info const * const rinfo)
{
  timer_reset(&timers[TIMER_ATA]);
#ifdef SPLATT_USE_MPI
  timer_reset(&timers[TIMER_MPI]);
  timer_reset(&timers[TIMER_MPI_IDLE]);
  timer_reset(&timers[TIMER_MPI_COMM]);
  timer_reset(&timers[TIMER_MPI_ATA]);
  timer_reset(&timers[TIMER_MPI_REDUCE]);
  timer_reset(&timers[TIMER_MPI_NORM]);
  timer_reset(&timers[TIMER_MPI_UPDATE]);
  timer_reset(&timers[TIMER_MPI_FIT]);
  MPI_Barrier(rinfo->comm_3d);
#endif
}


/**
* @brief Find the Frobenius norm squared of a Kruskal tensor. This equivalent
*        to via computing <X,X>, the inner product of X with itself. We find
*        this via \lambda^T (AtA * BtB * ...) \lambda, where * is the Hadamard
*        product.
*
* @param nmodes The number of modes in the tensor.
* @param lambda The vector of column norms.
* @param aTa An array of Gram Matrices (AtA, BtB, ...).
*
* @return The Frobenius norm of X, squared.
*/
static val_t p_kruskal_norm(
  idx_t const nmodes,
  val_t const * const restrict lambda,
  matrix_t ** aTa)
{
  idx_t const rank = aTa[0]->J;
  val_t * const restrict av = aTa[MAX_NMODES]->vals;

  val_t norm_mats = 0;

  /* use aTa[MAX_NMODES] as scratch space */
  for(idx_t i=0; i < rank; ++i) {
    for(idx_t j=i; j < rank; ++j) {
      av[j + (i*rank)] = 1.;
    }
  }

  /* aTa[MAX_NMODES] = hada(aTa) */
  for(idx_t m=0; m < nmodes; ++m) {
    val_t const * const restrict atavals = aTa[m]->vals;
    for(idx_t i=0; i < rank; ++i) {
      for(idx_t j=i; j < rank; ++j) {
        av[j + (i*rank)] *= atavals[j + (i*rank)];
      }
    }
  }

  /* now compute lambda^T * aTa[MAX_NMODES] * lambda */
  for(idx_t i=0; i < rank; ++i) {
    norm_mats += av[i+(i*rank)] * lambda[i] * lambda[i];
    for(idx_t j=i+1; j < rank; ++j) {
      norm_mats += av[j+(i*rank)] * lambda[i] * lambda[j] * 2;
    }
  }

  return fabs(norm_mats);
}


/**
* @brief Compute the inner product of a Kruskal tensor and an unfactored
*        tensor. Assumes that 'm1' contains the MTTKRP result along the last
*        mode of the two input tensors. This naturally follows the end of a
*        CPD iteration.
*
* @param nmodes The number of modes in the input tensors.
* @param rinfo MPI rank information.
* @param thds OpenMP thread data structures.
* @param lambda The vector of column norms.
* @param mats The Kruskal-tensor matrices.
* @param m1 The result of doing MTTKRP along the last mode.
*
* @return The inner product of the two tensors, computed via:
*         1^T hadamard(mats[nmodes-1], m1) \lambda.
*/
static val_t p_tt_kruskal_inner(
  idx_t const nmodes,
  rank_info * const rinfo,
  thd_info * const thds,
  val_t const * const restrict lambda,
  matrix_t ** mats,
  matrix_t const * const m1)
{
  idx_t const rank = mats[0]->J;
  idx_t const lastm = nmodes - 1;
  idx_t const dim = m1->I;

  val_t const * const m0 = mats[lastm]->vals;
  val_t const * const mv = m1->vals;

  val_t myinner = 0;
  #pragma omp parallel reduction(+:myinner)
  {
    int const tid = splatt_omp_get_thread_num();
    val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];

    for(idx_t r=0; r < rank; ++r) {
      accumF[r] = 0.;
    }

    #pragma omp for
    for(idx_t i=0; i < dim; ++i) {
      for(idx_t r=0; r < rank; ++r) {
        accumF[r] += m0[r+(i*rank)] * mv[r+(i*rank)];
      }
    }
    /* accumulate everything into 'myinner' */
    for(idx_t r=0; r < rank; ++r) {
      myinner += accumF[r] * lambda[r];
    }
  }
  val_t inner = 0.;

#ifdef SPLATT_USE_MPI
  timer_start(&timers[TIMER_MPI_FIT]);
  MPI_Allreduce(&myinner, &inner, 1, SPLATT_MPI_VAL, MPI_SUM, rinfo->comm_3d);
  timer_stop(&timers[TIMER_MPI_FIT]);
#else
  inner = myinner;
#endif

  return inner;
}


/**
* @brief Compute the fit of a Kruskal tensor, Z, to an input tensor, X. This
*        is computed via 1 - [sqrt(<X,X> + <Z,Z> - 2<X,Z>) / sqrt(<X,X>)].
*
* @param nmodes The number of modes in the input tensors.
* @param rinfo MPI rank information.
* @param thds OpenMP thread data structures.
* @param ttnormsq The norm (squared) of the original input tensor, <X,X>.
* @param lambda The vector of column norms.
* @param mats The Kruskal-tensor matrices.
* @param m1 The result of doing MTTKRP along the last mode.
* @param aTa An array of matrices (length MAX_NMODES)containing BtB, CtC, etc.
*
* @return The inner product of the two tensors, computed via:
*         \lambda^T hadamard(mats[nmodes-1], m1) \lambda.
*/
static val_t p_calc_fit(
  idx_t const nmodes,
  rank_info * const rinfo,
  thd_info * const thds,
  val_t const ttnormsq,
  val_t const * const restrict lambda,
  matrix_t ** mats,
  matrix_t const * const m1,
  matrix_t ** aTa)
{
  timer_start(&timers[TIMER_FIT]);

  /* First get norm of new model: lambda^T * (hada aTa) * lambda. */
  val_t const norm_mats = p_kruskal_norm(nmodes, lambda, aTa);

  /* Compute inner product of tensor with new model */
  val_t const inner = p_tt_kruskal_inner(nmodes, rinfo, thds, lambda, mats,m1);

  /*
   * We actually want sqrt(<X,X> + <Y,Y> - 2<X,Y>), but if the fit is perfect
   * just make it 0.
   */
  val_t residual = ttnormsq + norm_mats - (2 * inner);
  if(residual > 0.) {
    residual = sqrt(residual);
  }
  timer_stop(&timers[TIMER_FIT]);
  return 1 - (residual / sqrt(ttnormsq));
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
double cpd_als_iterate(
  splatt_csf const * const tensors,
  matrix_t ** mats,
  val_t * const lambda,
  idx_t const nfactors,
  rank_info * const rinfo,
  double const * const opts)
{
  idx_t const nmodes = tensors[0].nmodes;
  idx_t const nthreads = (idx_t) opts[SPLATT_OPTION_NTHREADS];

  /* Setup thread structures. + 64 bytes is to avoid false sharing.
   * TODO make this better */
  splatt_omp_set_num_threads(nthreads);
  thd_info * thds =  thd_init(nthreads, 3,
    (nmodes * nfactors * sizeof(val_t)) + 64,
    0,
    (nmodes * nfactors * sizeof(val_t)) + 64);

  matrix_t * m1 = mats[MAX_NMODES];

  /* Initialize first A^T * A mats. We redundantly do the first because it
   * makes communication easier. */
  matrix_t * aTa[MAX_NMODES+1];
  for(idx_t m=0; m < nmodes; ++m) {
    aTa[m] = mat_alloc(nfactors, nfactors);
    memset(aTa[m]->vals, 0, nfactors * nfactors * sizeof(val_t));
    mat_aTa(mats[m], aTa[m], rinfo, thds, nthreads);
  }
  /* used as buffer space */
  aTa[MAX_NMODES] = mat_alloc(nfactors, nfactors);

  /* mttkrp workspace */
  splatt_mttkrp_ws * mttkrp_ws = splatt_mttkrp_alloc_ws(tensors,nfactors,opts);

  double oldfit = 0;
  double fit = 0;
  val_t ttnormsq = csf_frobsq(tensors);

  /* setup timers */
  p_reset_cpd_timers(rinfo);
  sp_timer_t itertime;
  sp_timer_t modetime[MAX_NMODES];
  timer_start(&timers[TIMER_CPD]);

  struct timespec start, end;
  float diff;

  idx_t const niters = (idx_t) opts[SPLATT_OPTION_NITER];
//  for(idx_t it=0; it < niters; ++it) {
    idx_t  it = 0;
    timer_fstart(&itertime);
//    for(idx_t m=0; m < nmodes; ++m) {
      idx_t m = 0;
      timer_fstart(&modetime[m]);
      mats[MAX_NMODES]->I = tensors[0].dims[m];
      m1->I = mats[m]->I;

#ifdef DUMP
    FILE* fp = fopen("args.h", "w");

      /* print tensors */

      //sparsity structs first
      for(int x = 0; x < 2; x++){
        fprintf(fp, "splatt_idx_t fptr_vals_%d[] = {", x);
        fprintf(fp, "%d, ", *(tensors[x].pt->fptr[0]));
        fprintf(fp, "%d};\n\n", *(tensors[x].pt->fptr[1]));
      }

      for(int x = 0; x < 2; x++){
	fprintf(fp, "splatt_idx_t fids_vals_%d[] = {", x);
        for(int i = 0; i < 2; i++){
           fprintf(fp, "%d, ", *(tensors[x].pt->fids[i]));
        }
        fprintf(fp, "%d};\n\n", *(tensors[x].pt->fids[2]));
      }

      for(int x = 0; x < 2; x++){
	fprintf(fp, "splatt_val_t pt_vals_%d[] ={", x);
	for(int i = 0; i < tensors[x].nnz-1; i++){
           fprintf(fp, "%f, ", tensors[x].pt->vals[i]);
        }
        fprintf(fp, "%f};\n\n", tensors[x].pt->vals[tensors[x].nnz-1]);	
      }

      for(int s = 0; s < 2; s++){
	fprintf(fp, "csf_sparsity pt_struct_%d ={\n", s);
	fprintf(fp, ".nfibs = {");
	for (int x = 0; x < 2; x++){
	   fprintf(fp, "%d, ", tensors[s].pt->nfibs[x]);
	}
	fprintf(fp, "%d},\n", tensors[s].pt->nfibs[2]);

	fprintf(fp, ".fptr = {");
  	fprintf(fp, "&fptr_vals_%d[0], ", s);
	fprintf(fp, "&fptr_vals_%d[1]},\n", s);

	fprintf(fp, ".fids = {");
	for(int x = 0; x < 2; x++){
           fprintf(fp, "&fids_vals_%d[%d], ", s, x);
	}
	fprintf(fp, "&fids_vals_%d[2]},\n", s);

	fprintf(fp, ".vals = &pt_vals_%d[0]", s);

	fprintf(fp, "};\n\n");
      }

      //Now tensors
      fprintf(fp, "splatt_csf tensors_arr[] = {\n");
  
    for(int t = 0; t < 2; t++){
      fprintf(fp, "{.nnz = %d,\n.nmodes = %d,\n", tensors[t].nnz, tensors[t].nmodes);
      
      fprintf(fp, ".dims = {");
      for (int x = 0; x < 2; x++){
        fprintf(fp, "%d, ", tensors[t].dims[x]);
      }
      fprintf(fp, "%d},\n", tensors[t].dims[2]);

      fprintf(fp, ".dim_perm = {");
      for (int x = 0; x < 2; x++){
        fprintf(fp, "%d, ", tensors[t].dim_perm[x]);
      }
      fprintf(fp, "%d},\n", tensors[t].dim_perm[2]);

      fprintf(fp, ".dim_iperm = {");
      for (int x = 0; x < 2; x++){
        fprintf(fp, "%d, ", tensors[t].dim_iperm[x]);
      }
      fprintf(fp, "%d},\n", tensors[t].dim_iperm[2]);

      fprintf(fp, ".which_tile = %d,\n.ntiles = %d,\n", tensors[t].which_tile, tensors[t].ntiles);
    
      fprintf(fp, ".ntiled_modes = %d,\n.tile_dims = {", tensors[t].ntiled_modes);
      for(int x = 0; x < 2; x++){
        fprintf(fp, "%d, ", tensors[t].tile_dims[x]);
      }
      fprintf(fp, "%d},\n", tensors[t].tile_dims[2]);

      fprintf(fp, ".pt = &pt_struct_%d", t);
   
      fprintf(fp, "},\n");
   }
      
      fprintf(fp, "};\n\n");

      fprintf(fp, "splatt_csf * _tensors = &tensors_arr[0];");

      fprintf(fp, "\n\n\n");


      /* print mats */

      //Get vals
      for(int x = 0; x < 4; x++){
	fprintf(fp, "val_t mat_vals_%d[] ={", x);
	for(int i = 0; i < mats[x]->I * mats[x]->J - 1; i++){
          fprintf(fp, "%f, ", mats[x]->vals[i]);
        } 
        fprintf(fp, "%f};\n\n", mats[x]->vals[mats[x]->I * mats[x]->J - 1]);
      }

      //Actual matrices
      for(int x = 0; x < 4; x++){
	fprintf(fp, "matrix_t mat_%d = {\n", x);
	fprintf(fp, ".I = %d, .J = %d, ", mats[x]->I, mats[x]->J);
	fprintf(fp, ".vals = &mat_vals_%d[0], ", x);
	fprintf(fp, ".rowmajor = %d\n};\n\n", mats[x]->rowmajor); 
      }
      
      //Array of pointers to matrices
      fprintf(fp, "matrix_t * mat_ptrs[] = {");
      for(int x = 0; x < 3; x++){
	fprintf(fp, "&mat_%d, ", x);
      }
      fprintf(fp, "&mat_3};\n\n");

      //Param we are using is a double pointer
      fprintf(fp, "matrix_t ** _mats = &mat_ptrs[0];");


      fprintf(fp, "\n\n\n");

   
      /* print m */
      fprintf(fp, "int _m = 0;");


      fprintf(fp, "\n\n\n");


      /* print thds */
      

      /* print mttkrp_ws, */

      //Get actual privatize buffer
      fprintf(fp, "splatt_val_t true_pb = %d;\n\n", **mttkrp_ws->privatize_buffer);     

      //tree partitions
      for(int x = 0; x < 2; x++){
        fprintf(fp, "splatt_idx_t tree_part_%d = %d;\n\n", x, *mttkrp_ws->tree_partition[x]);
      }  

      //privatize buffer pointer
      fprintf(fp, "splatt_val_t * pb_ptr = &true_pb;\n\n");

      //the actual ws struct
      fprintf(fp, "splatt_mttkrp_ws ws = {\n");
      fprintf(fp, ".num_csf = %d, \n", mttkrp_ws->num_csf);
      fprintf(fp, ".mode_csf_map = {");
      for(int x = 0; x < 2; x++){
	fprintf(fp, "%d, ", mttkrp_ws->mode_csf_map[x]);
      }
      fprintf(fp, "%d}, \n", mttkrp_ws->mode_csf_map[2]);
      
      fprintf(fp, ".num_threads = %d, \n", mttkrp_ws->num_threads);
      
      fprintf(fp, ".tile_partition = {");
      for(int x = 0; x < 1; x++){
        fprintf(fp, "%d, ", mttkrp_ws->tile_partition[x]);
      }
      fprintf(fp, "%d}, \n", mttkrp_ws->tile_partition[1]);

      fprintf(fp, ".tree_partition = {");
      fprintf(fp, "&tree_part_0, &tree_part_1},\n");

      fprintf(fp, ".is_privatized = {");
      for(int x = 0; x < 2; x++){
        fprintf(fp, "%d, ", mttkrp_ws->is_privatized[x]);
      }
      fprintf(fp, "%d}, \n", mttkrp_ws->is_privatized[2]);

      fprintf(fp, ".privatize_buffer = &pb_ptr,\n");
      
      fprintf(fp, ".reduction_time = %lf", mttkrp_ws->reduction_time);

      fprintf(fp, "\n};\n\n");
      
      //Our param is a pointer
      fprintf(fp, "splatt_mttkrp_ws * _mttkrp_ws = &ws;");


      fprintf(fp, "\n\n\n");


      /*print opts */
 
      //Get true val
      fprintf(fp, "double true_opts = %lf;\n\n", *opts);
      //opts is a ptr
      fprintf(fp, "double * _opts = &true_opts;\n");


#endif

      /* M1 = X * (C o B) */
      timer_start(&timers[TIMER_MTTKRP]);
      clock_gettime(CLOCK_MONOTONIC, &start); /* mark start time */
      #ifdef DUMP
      mttkrp_csf(tensors, mats, m, thds, mttkrp_ws, opts);
      #else
      mttkrp_csf(_tensors, _mats, _m, thds, _mttkrp_ws, _opts);
      #endif
      clock_gettime(CLOCK_MONOTONIC, &end); /* mark the end time */
      diff += ( BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec );
      printf("Time taken: %f ns (%ld cycles)\n", diff,  (long int)(diff/0.5)); // number if cycles at 2GHz clock
      timer_stop(&timers[TIMER_MTTKRP]);

#if 0
      /* M2 = (CtC .* BtB .* ...)^-1 */
      calc_gram_inv(m, nmodes, aTa);
      /* A = M1 * M2 */
      memset(mats[m]->vals, 0, mats[m]->I * nfactors * sizeof(val_t));
      mat_matmul(m1, aTa[MAX_NMODES], mats[m]);
#else
      par_memcpy(mats[m]->vals, m1->vals, m1->I * nfactors * sizeof(val_t));
      mat_solve_normals(m, nmodes, aTa, mats[m],
          opts[SPLATT_OPTION_REGULARIZE]);
#endif

      /* normalize columns and extract lambda */
      if(it == 0) {
        mat_normalize(mats[m], lambda, MAT_NORM_2, rinfo, thds, nthreads);
      } else {
        mat_normalize(mats[m], lambda, MAT_NORM_MAX, rinfo, thds,nthreads);
      }

      /* update A^T*A */
      mat_aTa(mats[m], aTa[m], rinfo, thds, nthreads);
      timer_stop(&modetime[m]);
//    } /* foreach mode */

    fit = p_calc_fit(nmodes, rinfo, thds, ttnormsq, lambda, mats, m1, aTa);
    timer_stop(&itertime);

    if(rinfo->rank == 0 &&
        opts[SPLATT_OPTION_VERBOSITY] > SPLATT_VERBOSITY_NONE) {
      printf("  its = %3"SPLATT_PF_IDX" (%0.3fs)  fit = %0.5f  delta = %+0.4e\n",
          it+1, itertime.seconds, fit, fit - oldfit);
      if(opts[SPLATT_OPTION_VERBOSITY] > SPLATT_VERBOSITY_LOW) {
        for(idx_t m=0; m < nmodes; ++m) {
          printf("     mode = %1"SPLATT_PF_IDX" (%0.3fs)\n", m+1,
              modetime[m].seconds);
        }
      }
    }
    if(fit == 1. || 
        (it > 0 && fabs(fit - oldfit) < opts[SPLATT_OPTION_TOLERANCE])) {
      ///break;
    }
    oldfit = fit;
//  }
  timer_stop(&timers[TIMER_CPD]);

  cpd_post_process(nfactors, nmodes, mats, lambda, thds, nthreads, rinfo);

  /* CLEAN UP */
  splatt_mttkrp_free_ws(mttkrp_ws);
  for(idx_t m=0; m < nmodes; ++m) {
    mat_free(aTa[m]);
  }
  mat_free(aTa[MAX_NMODES]);
  thd_free(thds, nthreads);

  return fit;
}



void cpd_post_process(
  idx_t const nfactors,
  idx_t const nmodes,
  matrix_t ** mats,
  val_t * const lambda,
  thd_info * const thds,
  idx_t const nthreads,
  rank_info * const rinfo)
{
  val_t * tmp =  splatt_malloc(nfactors * sizeof(*tmp));

  /* normalize each matrix and adjust lambda */
  for(idx_t m=0; m < nmodes; ++m) {
    mat_normalize(mats[m], tmp, MAT_NORM_2, rinfo, thds, nthreads);
    for(idx_t f=0; f < nfactors; ++f) {
      lambda[f] *= tmp[f];
    }
  }

  free(tmp);
}


