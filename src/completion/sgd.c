

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "completion.h"
#include "../reorder.h"
#include "../timer.h"
#include "../util.h"

#include <math.h>
#include <omp.h>



/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Update a model based on a given observation.
*
* @param train The training data.
* @param nnz_index The index of the observation to update from.
* @param model The model to update.
* @param ws Workspace to use.
*/
static void p_update_model(
    sptensor_t const * const train,
    idx_t const nnz_index,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t const nfactors = model->rank;
  idx_t const nmodes = train->nmodes;
  idx_t const x = nnz_index;

  val_t * const restrict buffer = ws->thds[omp_get_thread_num()].scratch[0];

  /* compute the error */
  val_t const err = train->vals[x] - tc_predict_val(model, train, x, buffer);

  idx_t * * const ind = train->ind;

  /* update each of the factor (row-wise) */
  for(idx_t m=0; m < nmodes; ++m) {

    /* first fill buffer with the Hadamard product of all rows but current */
    idx_t moff = (m + 1) % nmodes;
    val_t const * const restrict init_row = model->factors[moff] +
        (ind[moff][x] * nfactors);
    for(idx_t f=0; f < nfactors; ++f) {
      buffer[f] = init_row[f];
    }
    for(moff = 2; moff < nmodes; ++moff) {
      idx_t const madj = (m + moff) % nmodes;
      val_t const * const restrict row = model->factors[madj] +
          (ind[madj][x] * nfactors);
      for(idx_t f=0; f < nfactors; ++f) {
        buffer[f] *= row[f];
      }
    }

    /* now actually update the row */
    val_t * const restrict update_row = model->factors[m] +
        (ind[m][x] * nfactors);
    val_t const reg = ws->regularization[m];
    val_t const rate = ws->learn_rate;
    for(idx_t f=0; f < nfactors; ++f) {
      update_row[f] += rate * ((err * buffer[f]) - (reg * update_row[f]));
    }
  }
}







/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void splatt_tc_sgd(
    sptensor_t * train,
    sptensor_t const * const validate,
    tc_model * const model,
    tc_ws * const ws)
{
  idx_t const nfactors = model->rank;
  val_t const * const restrict train_vals = train->vals;

  idx_t * perm = splatt_malloc(train->nnz * sizeof(*perm));

  sp_timer_t train_time;
  sp_timer_t test_time;
  timer_reset(&train_time);
  timer_reset(&test_time);

  /* init perm */
  for(idx_t n=0; n < train->nnz; ++n) {
    perm[n] = n;
  }

  val_t prev_obj = 0;
  val_t prev_val_rmse = 0;

  /* foreach epoch */
  for(idx_t e=0; e < ws->max_its; ++e) {
    timer_start(&train_time);

    /* new nnz ordering */
    shuffle_idx(perm, train->nnz);

    /* update model from all training observations */
    for(idx_t n=0; n < train->nnz; ++n) {
      p_update_model(train, perm[n], model, ws);
    }
    timer_stop(&train_time);

    /* compute RMSE and adjust learning rate */
    timer_start(&test_time);
    val_t const loss = tc_loss_sq(train, model, ws);
    val_t const frobsq = tc_frob_sq(model, ws);
    val_t const obj = loss + frobsq;
    val_t const train_rmse = sqrt(loss / train->nnz);
    val_t const val_rmse = tc_rmse(validate, model, ws);
    timer_stop(&test_time);

    printf("epoch:%4"SPLATT_PF_IDX"   obj: %0.5e   "
        "RMSE-tr: %0.5e   RMSE-vl: %0.5e time-tr: %0.3fs  time-ts: %0.3fs\n",
        e+1, obj, train_rmse, val_rmse, train_time.seconds, test_time.seconds);

    if(e > 0) {
      if(obj < prev_obj) {
        ws->learn_rate *= 1.05;
      } else {
        ws->learn_rate *= 0.50;
      }

      /* check convergence */
      if(fabs(val_rmse - prev_val_rmse) < 1e-8) {
        break;
      }
    }

    prev_obj = obj;
    prev_val_rmse = val_rmse;
  }

  splatt_free(perm);
}

