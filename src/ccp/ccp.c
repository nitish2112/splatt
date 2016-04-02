

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "ccp.h"
#include "../timer.h"


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/


/**
* @brief Counts the number of times a probe was performed. This is useful for
*        benchmarking pruning strategies.
*/
static idx_t nprobes = 0;


/**
* @brief Perform a linear search on an array for a value.
*
* @param weights The array to search.
* @param left The lower bound to begin at.
* @param right The upper (exclusive) bound of items.
* @param target The target value.
*
* @return The index j, where weights[j] <= target && weights[j+1] > target.
*/
static idx_t p_linear_search(
    idx_t const * const weights,
    idx_t const left,
    idx_t const right,
    idx_t const target)
{
  for(idx_t x=left; x < right-1; ++x) {
    if(target < weights[x+1]) {
      return x+1;
    }
  }

  return right;
}


/**
* @brief Perform a binary search on an array for a value.
*
* @param weights The array to search.
* @param left The lower bound to begin at.
* @param right The upper (exclusive) bound of items.
* @param target The target value.
*
* @return The index j, where weights[j] <= target && weights[j+1] > target.
*/
static idx_t p_binary_search(
    idx_t const * const weights,
    idx_t left,
    idx_t right,
    idx_t const target)
{
  while((right - left) > 8) {
    idx_t mid = left + ((right - left) / 2);

    if(weights[mid] <= target && weights[mid+1] > target) {
      return mid;
    }

    if(weights[mid] < target) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  return p_linear_search(weights, left, right, target);
}



/**
* @brief Find an epsilon-approximate partitioning using recursive bisection.
*        If we are using integer weights and eps=0, this is an optimal
*        partitioning.
*
* @param weights An array of workload weights, length 'nitems'.
* @param nitems The number of items we are partitioning.
* @param[out] parts A ptr into weights, marking each partition. THIS IS ASSUMED
*                   to be pre-allocated at least of size 'nparts+1'.
* @param nparts The number of partitions to compute.
* @param eps RB is used until the possible range of optimality is within 'eps.'
*            Use eps=0 to compute exact partitionings (with integer weights).
*
* @return The amount of work in the largest partition (i.e., the bottleneck).
*/
static idx_t p_eps_rb_partition_1d(
    idx_t * const weights,
    idx_t const nitems,
    idx_t * const parts,
    idx_t const nparts,
    idx_t const eps)
{
  idx_t const tot_weight = weights[nitems-1];
  idx_t lower = tot_weight / nparts;
  idx_t upper = tot_weight;

  do {
    idx_t mid = lower + ((upper - lower) / 2);
    if(lprobe(weights, nitems, parts, nparts, mid)) {
      upper = mid;
    } else {
      lower = mid+1;
    }
  } while(upper > lower + eps);

  return upper;
}




/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

idx_t partition_1d(
    idx_t * const weights,
    idx_t const nitems,
    idx_t * const parts,
    idx_t const nparts)
{
  timer_start(&timers[TIMER_PART]);
  prefix_sum_inc(weights, nitems);

  nprobes = 0;

  idx_t nparts_adj = nparts;

  /* if nparts > nitems, just truncate */
  if(nparts > nitems) {
    for(idx_t p=nitems; p <= nparts; ++p) {
      parts[p] = nitems;
    }
    nparts_adj = nitems;
  }

  /* use recursive bisectioning with 0 tolerance to get exact solution */
  idx_t bottleneck = p_eps_rb_partition_1d(weights, nitems,parts,nparts_adj,0);

  /* apply partitioning that we found */
  bool success = lprobe(weights, nitems, parts, nparts_adj, bottleneck);
  assert(success == true);

  timer_stop(&timers[TIMER_PART]);
  return bottleneck;
}



bool lprobe(
    idx_t const * const weights,
    idx_t const nitems,
    idx_t * const parts,
    idx_t const nparts,
    idx_t const bottleneck)
{
  ++nprobes;
  idx_t const wtotal = weights[nitems-1];
  /* initialize partitioning */
  parts[0] = 0;
  for(idx_t p=1; p <= nparts; ++p) {
    parts[p] = nitems;
  }

  idx_t bsum = bottleneck;
  idx_t step = nitems / nparts;
  for(idx_t p=1; p < nparts; ++p) {
    /* jump to the next bucket */
    while(step < nitems && weights[step] < bsum) {
      step += nitems / nparts;
    }

    /* find the end (exclusive) index of process p */
    parts[p] = p_binary_search(weights, step - (nitems/nparts),
        SS_MIN(step,nitems), bsum);

    /* we ran out of stuff to do */
    if(parts[p] == nitems) {
      /* check for pathological case when the last weight is larger than
       * bottleneck */
      idx_t const size_last = weights[nitems-1] - weights[parts[p-1]-1];
      return size_last < bottleneck;
    }
    bsum = weights[parts[p]-1] + bottleneck;
  }

  return bsum >= wtotal;
}



void prefix_sum_inc(
    idx_t * const weights,
    idx_t const nitems)
{
  for(idx_t x=1; x < nitems; ++x) {
    weights[x] += weights[x-1];
  }
}



void prefix_sum_exc(
    idx_t * const weights,
    idx_t const nitems)
{
  idx_t saved = weights[0];
  weights[0] = 0;
  for(idx_t x=1; x < nitems; ++x) {
    idx_t const tmp = weights[x];
    weights[x] = weights[x-1] + saved;
    saved = tmp;
  }
}
