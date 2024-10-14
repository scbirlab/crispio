"""Calculate fitness from sequencing counts."""

from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from argparse import ArgumentParser, FileType, Namespace
from collections import defaultdict
from functools import partial
from itertools import chain
import sys

from carabiner import pprint_dict, print_err
from carabiner.pd import read_table
import jax
jax.config.update("jax_enable_x64", True)
from jax import Array, grad, jit, jvp, lax, vmap
from jax.typing import ArrayLike
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax.scipy import stats

try: 
    jnp.arange(10.)
except RuntimeError:
    print_err('WARNING: No GPU. Falling back to CPU.')

import numpy as np
import numpy.typing as npt
import optax
import pandas as pd
from scipy.optimize import OptimizeResult, minimize, minimize_scalar
from scipy.sparse.linalg import LinearOperator
from scipy.stats import chi2
from tqdm.auto import trange

def linear(x): return x

_PARAM_DEPENDENCE = dict(fitness_ratio=('guide_name', jnp.exp),
                         log_expansion=('exp_group', jnp.square),
                         log_exposure_ratio=('exp_seq_group', linear),
                         log_inoculum=('guide_name', linear),
                         size=(None, jnp.exp))

_SPINNER = '|/-\\'

def _parse_arguments() -> Namespace:

    parser = ArgumentParser(description='''
    Calculate fitness of guides or barcodes from sequencing counts contained in a TSV file.
    ''')
    parser.add_argument('input', 
                        type=FileType('r'),
                        default=sys.stdin,
                        nargs='?',
                        help='Input counts file in TSV format. Default STDIN.')
    parser.add_argument('--name', '-n', 
                        type=str,
                        nargs='*',
                        default=['source_name'],
                        help='Column name containing names of guides or barcodes. Default: %(default)s')
    parser.add_argument('--reference', '-r', 
                        type=str,
                        nargs='*',
                        required=True,
                        help='Name of the guide or barcode indicating a reference (or negative) control. Required')
    parser.add_argument('--sequencing_group', '-s', 
                        type=str,
                        nargs='*',
                        required=True,
                        help='Column name indicating samples from the same sequencing run. Required')
    parser.add_argument('--expansion_group', '-e', 
                        type=str,
                        nargs='*',
                        required=True,
                        help='Column name indicating samples from the same timepoint. Required')
    parser.add_argument('--culture', '-l', 
                        type=str,
                        nargs='*',
                        required=True,
                        help='Column name indicating samples from the same culture. Required')
    parser.add_argument('--initial', '-i', 
                        type=str,
                        nargs='*',
                        required=True,
                        help='Name of the initial (t = 0) expansion_group. Required')
    parser.add_argument('--count', '-c', 
                        type=str,
                        default='count',
                        help='Column name containing counts. Default: %(default)s')
    parser.add_argument('--format', '-f', 
                        type=str,
                        choices=['TSV', 'CSV'],
                        default='TSV',
                        help='Format of input counts table. Default: %(default)s')
    parser.add_argument('--algorithm', '-a', 
                        type=str,
                        choices=['L-BFGS-B', 'BFGS', 'CG', 'Newton-CG', 'trust-ncg'],
                        default='L-BFGS-B',
                        help='Algorithm for model fitting. Default: %(default)s')
    parser.add_argument('--output', '-o', 
                        type=str,
                        required=True,
                        help='Output file prefix. Required')

    args = parser.parse_args()

    pprint_dict(args,
                'Calculating fitness with the following parameters')

    return args


def _featurize_col(x: pd.DataFrame,
                   ref: Iterable[str]) -> Tuple[Tuple[ArrayLike, pd.Series], ArrayLike]:

    x = x.astype(str)
    
    if len(ref) != x.shape[1]:
        raise AttributeError(f'Number of reference names {len(ref)} '
                             f'must be the same as number of columns ({x.shape[1]}): ' +
                             ', '.join(ref) + '; ' + ', '.join(x.columns))

    missing_refs = [_ref for i, _ref in enumerate(ref) 
                    if _ref not in [None, '[pick]'] 
                    and _ref not in x.iloc[:,i].values]

    if len(missing_refs) > 0:
        raise AttributeError('All references must be in the data table. '
                             'Missing: ' + 
                             ", ".join(missing_refs) +
                             ' from ' + ':'.join(x.columns))
    
    labels = (x.iloc[:, 0]
               .str.cat(x.iloc[:, 1:].astype(str), sep=':'))
    
    x_top1 = x.copy()
    
    for i, _ref in enumerate(ref):

        if _ref is None:

            ref[i] = str(None)

        elif _ref != '[pick]':

            x_top1 = x_top1[x_top1.iloc[:,i] == _ref].copy()

    ref = x_top1.sort_values(x.columns.tolist()).iloc[0, :]
    ref = ':'.join(ref)

    print_err('Reference for columns', 
              ':'.join(x.columns.values), 
              'is', ref, 
              flush=True)

    categories = np.asarray(sorted(np.setdiff1d(labels.unique(), [ref])))

    codes = pd.Categorical(labels, 
                           categories, 
                           ordered=True).codes.astype(np.int64)

    return (codes, labels), categories


def _get_guides(df: pd.DataFrame, 
                guide_col: Sequence[int]) -> pd.Series:

    if len(guide_col) > 0:
        return df[guide_col[0]].str.cat(df[guide_col[1:]], sep=':').unique()
    else:
        return df[guide_col[0]].unique()


def _pysetdiff(a, b):

    return sorted(set(a) - set(b))


def _featurize(df: pd.DataFrame, 
               x_cols: Mapping,
               y_col: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    
    input_rows = df.shape[0]
    df['idx_col'] = np.arange(input_rows)

    # remove anything missing in inoculum
    exp_col, exp_init = x_cols['exp_group']
    guide_col, guide_ref = x_cols['guide_name']
    inoculum_query = ' and '.join(f'{c} == "{val}"' for c, val in zip(exp_col, exp_init)) + f" and {y_col} > 1."

    print_err(f'Checking guides present in inoculum ({inoculum_query}):', end=' ')
    df = df.assign(**{c: lambda x: x[c].astype(str) for c in exp_col})
    inoculum_df = df.query(inoculum_query)
    print_err(f'There are {inoculum_df.shape[0]} rows in the inoculum out of {df.shape[0]} rows.')

    assert inoculum_df.shape[0] != df.shape[0], f"Inoculum is the same size as total data: {inoculum_df.shape[0]}"
    assert inoculum_df.shape[0] > 0, f"Inoculum has no data!"

    guides_in_inoculum_df = inoculum_df[guide_col].drop_duplicates()
    all_guides, guides_in_inoculum = _get_guides(df, guide_col), _get_guides(guides_in_inoculum_df, guide_col)
    guides_removed = _pysetdiff(all_guides, guides_in_inoculum)

    print_err(f"Removing {len(guides_removed)} guides absent in inoculum:", end=' ')
    if len(guides_removed) < 10:
        print_err(', '.join(guides_removed))
    else:
        print_err(', '.join(guides_removed[:3]), '...', ', '.join(guides_removed[-3:]))

    print_err(f"Initial data size is {df.shape[0]};", end=" ")
    df = df.merge(guides_in_inoculum_df, 
                  how='inner', on=guide_col)
    print_err(f"final data size is {df.shape[0]}.")

    remaining_guides = _get_guides(df, guide_col)
    print_err(f"Started with {len(all_guides)} guides, retained {len(remaining_guides)}")

    guides_not_in_inoculum = _pysetdiff(remaining_guides, guides_in_inoculum)
    assert len(guides_not_in_inoculum) == 0, f"Not all guides in inoculum! {len(guides_not_in_inoculum)} were retained."  # Should never happen
    guides_not_deleted = _pysetdiff(remaining_guides, guides_removed)
    assert len(guides_not_deleted) == len(remaining_guides), f"{len(guides_not_deleted)} non-inoculum guides have been retained!"  # Should never happen
    
    X = {name: _featurize_col(df[x], ref) 
         for name, (x, ref) in x_cols.items()}
    
    for name, ((_, labels), _) in X.items():
        df[name] = labels
    
    Y = df[y_col].astype(np.float64).values[:, np.newaxis]

    end_rows = df.shape[0]
    print_err(f"> Featurized input data and filtered unused guides. Started with {input_rows}, ended with {end_rows}")

    return df, X, Y


@jit
@partial(vmap, in_axes=(0, None))
def _sparse_matmul(a: ArrayLike, 
                   b: ArrayLike) -> Array:
    
    padding = jnp.zeros((1, b.shape[-1]))
    padded_b = jnp.concatenate([b, padding], axis=0)

    return jnp.sum(padded_b.take(a, axis=0), axis=0, keepdims=True)


# @jit
def _extract_param(pname: str, 
                   x: Mapping[str, ArrayLike], 
                   params: Mapping[str, ArrayLike]) -> Array:
    
    xname, f = _PARAM_DEPENDENCE[pname]
    
    return f(_sparse_matmul(x[xname], params[pname]))


@jit
def _model(params: Mapping[str, ArrayLike], 
           x: Mapping[str, ArrayLike], 
           count_offset: float) -> Array:

    fitness_ratio = _extract_param('fitness_ratio', x, params)

    inoculum = (_extract_param('log_inoculum', x, params) + count_offset)
    
    log_expansion = _extract_param('log_expansion', x, params)
    log_exposure_ratio = _extract_param('log_exposure_ratio', x, params) 
    
    log_count = (fitness_ratio - 1.) * log_expansion + log_exposure_ratio + inoculum

    return log_count


@jit
def _negative_binomial_neg_ll(y_true: ArrayLike,
                              y_pred: ArrayLike,
                              size: ArrayLike) -> Array:
    
    p = 1. / (1. + size * y_pred)
    n = 1. / size
    log_likelihood = stats.nbinom.logpmf(y_true, n, p)
    
    return -jnp.sum(log_likelihood)


def _runif(shape: int,
           scale: float = .001,
           range: Iterable[float] = (-1., 1.)) -> np.ndarray:

    return (np.random.uniform(*range, (shape, 1)) * scale).astype(np.float64)


def _initialize_params(x: Mapping[str, Array],
                       scale: float = .001,
                       range: Iterable[float] = (-1., 1.)) -> Dict[str, Union[float, Array]]:
    
    params = {p: _runif(shape=len(x[x_col][1]), 
                        scale=scale, 
                        range=range)
              if x_col is not None else _runif(1, scale, range) 
              for p, (x_col, _) in _PARAM_DEPENDENCE.items()}
    
    return params


def _unpack_x(x: Mapping[str, Array]) -> Dict[str, Array]:

    return {x_col: x_data for x_col, ((x_data, _), _) in x.items()}


def _make_objective(x0: Mapping) -> Callable[[ArrayLike], Array]:
    
    _, x0_unflatten = ravel_pytree(x0)
    
    @jit
    def _pred(p: ArrayLike, 
              x: ArrayLike,
              offset: ArrayLike) -> Tuple[Array, Array]:
        
        p = x0_unflatten(p)

        y_pred = jnp.exp(_model(params=p, 
                                x=x,
                                count_offset=offset,
                                # exposure_offset=log_exposure_offset,
                                # log_ref_expansion=log_ref_expansion,
                                # t0_offset=log_t0_offset
                                ))

        return jnp.exp(p['size']), y_pred

    @jit
    def _objective(p: ArrayLike,
                   x: ArrayLike,
                   y: ArrayLike,
                   offset: ArrayLike) -> Array:

        size, y_pred = _pred(p, x, offset)
        
        return _negative_binomial_neg_ll(y, y_pred, size)
      
    _grad = jit(grad(_objective))

    def _HVP(x: ArrayLike,
             y: ArrayLike,
             offset: ArrayLike) -> Callable[[ArrayLike, ArrayLike], Array]:

        grad = partial(_grad, x=x, y=y, offset=offset)

        @jit
        def _hvp(p: ArrayLike, 
                 v: ArrayLike) -> Array:
            
            return jvp(grad, (p, ), (v, ))[1]
        
        return _hvp
    
    return _pred, _objective, _grad, _HVP


def _warmup_params(objective: Callable,
                   x: Mapping[str, ArrayLike],
                   y: Mapping,
                   x0: Mapping,
                   offset: ArrayLike,
                   jac: Callable,
                   ref_only: bool = False, 
                   n_epochs: int = 1,
                   batch_size: int = 1) -> Tuple[Tuple[float], Array]:
    
    x0, x0_unflatten = ravel_pytree(x0)

    if ref_only:
        (guide_codes, _), _ = x['guide_name']
        ref_codes = (guide_codes < 0)
        x = {name: ((codes[ref_codes], labels.loc[ref_codes]), categories) 
             for name, ((codes, labels), categories) in x.items()}
        y = y[ref_codes]


    initial_loss = objective(x0, _unpack_x(x), y, offset)
    print_err(f'Stochastic training for {n_epochs} epochs: initial loss {initial_loss:.2f} ->', end=' ')
    
    n_batches = np.ceil(y.size / (batch_size + 1)).astype(int)
    n_steps = 10 * y.shape[0]

    schedule = optax.exponential_decay(0.0001, 
                                       transition_steps=n_steps // 2, 
                                       decay_rate=.5)
    
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(x0)

    @jit
    def update(opt_state, params, x, y):
        
        this_loss, this_grad = objective(params, x, y, offset), jac(params, x, y, offset)
        updates, opt_state = optimizer.update(this_grad, opt_state)
        params = optax.apply_updates(params, updates)
        
        return this_loss, this_grad, opt_state, params
    
    for _ in range(n_epochs):

        index_sets = np.array_split(np.random.choice(np.arange(y.size),
                                      size=y.size, 
                                      replace=y.size < batch_size), 
                                    n_batches)

        for i in range(n_batches): #, ncols=80):

            these_indices = index_sets[i]

            x_batch = {name: codes[these_indices] 
                       for name, ((codes, _), _) in x.items()}
            y_batch = y[these_indices]

            this_loss, _, opt_state, x0 = update(opt_state, x0, x_batch, y_batch)

            if jnp.isnan(this_loss):

                break

        new_loss = objective(x0, _unpack_x(x), y, offset)

    print_err(f'final loss {new_loss:.2f}')

    return (initial_loss, new_loss), x0_unflatten(x0)


def _pick_best_offset(objective: Callable,
                      x: Mapping[str, ArrayLike],
                      y: Mapping,
                      x0: Mapping,
                      jac: Callable) -> Tuple[float, Array]:

    def offset_objective(offset):
    
        ((start_loss, end_loss), 
          warm_parameters) = _warmup_params(objective, 
                                            x, y,
                                            x0, 
                                            offset,
                                            jac, 
                                            ref_only=True)
        return end_loss
    
    y_no_zero = y[y > 0.]
    y_min, y_max = np.min(y_no_zero), np.max(y_no_zero)

    if y_max <= y_min:
        y_max = y_min + 1.

    offset_bracket = (np.log(y_min), np.log(y_max))
    print_err(f"Offset bracket: {offset_bracket} [{y_min, y_max}]")
    best_offset = minimize_scalar(offset_objective,
                                  bracket=offset_bracket)
    print_err(f"Best offset: {best_offset}")

    ((start_loss, end_loss), 
     warm_parameters) = _warmup_params(objective, 
                                        x, y,
                                        x0, 
                                        best_offset.x,
                                        jac,
                                        n_epochs=5,
                                        batch_size=32)
    
    return best_offset.x, warm_parameters


def _fit_objective(objective: Callable,
                   x0: Mapping,
                   method: str,
                   jac: Callable,
                   hessp: Callable) -> Tuple[List[float], OptimizeResult]:
    
    n_iter, history = 0, []

    def _callback(p: ArrayLike) -> None:

        loss = objective(p)
        
        nonlocal n_iter, history

        n_iter += 1
        history.append(loss)

        print_err(('\r' + _SPINNER[n_iter % 4] + 
                   f' Iteration {n_iter}, loss: {loss:.2f}'), 
                   flush=True, end='')

        return None

    params_concat = np.concatenate(list(x0.values())).flatten()
    print_err(f'Fitting model with {params_concat.size} {params_concat.shape} '
              f'parameters using {method}...')
    optimized = minimize(objective,
                         x0=params_concat,
                         method=method,
                         jac=jac,
                         hessp=hessp,
                         callback=_callback)
    
    return history, optimized


def _get_hess_diag(hvp: Callable,
                   length: int) -> Callable[[int, ArrayLike], Tuple[int, Array]]:
    
    def f(carry: int, x: ArrayLike) -> Tuple[int, Array]:

        jax.debug.print("Calculating Hessian diagonal: {carry} / {length} ({pct_progress} %)",
                        carry=carry, 
                        length=length, 
                        pct_progress=100. * carry / length)

        return carry + 1, hvp(jnp.zeros((length, )).at[carry].set(1.))[carry]
    
    return f


def _get_hess_diag2(hvp: Callable,
                    length: int) -> Callable[[ArrayLike], Array]:

    _x0 = jnp.zeros((length, ))
    
    @jit
    @vmap
    def f(carry: ArrayLike) -> Array:

        # pct_progress =  carry / length
        # spinner = _SPINNER[carry % 4]
        jax.debug.print("Calculating Hessian diagonal: {carry} / {length} ({pct_progress} %)",
                        carry=carry, 
                        length=length, 
                        pct_progress=100. * carry / length)

        return hvp(_x0.at[carry].set(1.))[carry]
    
    return f


def _one_vec(length: int, i: int) -> Array:

    v = np.zeros((length, ))
    v[i] = np.ones((1,))

    return v


def _extract_diag(opt: OptimizeResult,
                  hvp: Optional[Callable[[ArrayLike], Array]] = None,
                  gradient: Optional[Callable[[ArrayLike], Array]] = None,
                  nd: int = 0) -> Array:
    
    dof = nd - opt.x.size

    if dof < 1:
        print_err(f'!!! WARNING: number of datapoints ({nd}) is less than '
                  f'number of parameters ({opt.x.size}), giving {dof} '
                  'degrees of freedom. '
                  'It\'s likely that the parameters and statistics are '
                  'meaningless.')
        
    dof = max(1, dof)
    var_residuals = opt.fun / dof
    var_residuals_no_nan = var_residuals if not np.isnan(var_residuals) else 1.

    print_err('Extracting standard errors...\n'
              f'> degrees of freedom: {dof}\n'
              f'> Variance of residuals: {var_residuals:.2f}')
    
    try:
        
        hess_inv = opt.hess_inv

    except AttributeError:

        def hess_inv(x):
            
            return hvp(opt.x, x)

    if isinstance(hess_inv, Callable):
        
        # id_matrix = np.eye(opt.x.shape[-1])
        
        if hvp is None and isinstance(hess_inv, 
                                      LinearOperator):

            print_err('> Using approximate inverse Hessian') 
            inv_hess_diag = np.asarray([hess_inv(_one_vec[opt.x.shape[-1], i])[i] 
                                        for i in trange(opt.x.shape[-1])])

        else:

            n_params = opt.x.size
            
            base_time, base_n = 244, 193433
            max_time = 5 * 60
            time_estimate = base_time * np.square(n_params / base_n) / 60.
            print_err(f"> Good Hessian approximation would take about {time_estimate:.1f} hours ({(time_estimate / 24.):.2f} days).")

            size_cutoff = int(base_n * np.sqrt(max_time / base_time))
            print_err(f"> Maximum number of parameters would be {size_cutoff} ({(max_time / 60.):.1f} hours). Here, there are {n_params} parameters.")

            if n_params <= size_cutoff:

                print_err('> Using approximated diagonal of inverse',
                          f'Hessian from exact Hessian diagonal. Expecting to take {time_estimate:.1f} hours.')
                
                def hess_inv(x):
                
                    return hvp(opt.x, x)
                
                # hess_idx = jnp.arange(n_params)[:, jnp.newaxis]
                # hess_diag_fun = _get_hess_diag2(hess_inv, n_params)
                # hess_diag = hess_diag_fun(hess_idx)         
                _, hess_diag = lax.scan(_get_hess_diag(hess_inv, n_params), 
                                        0, jnp.ones((n_params, 1)))
                hess_diag_l2 = jnp.sqrt(jnp.sum(jnp.square(hess_diag)))
                inv_hess_diag = hess_diag / hess_diag_l2

            elif gradient is not None and isinstance(gradient, Callable):

                print_err(f'> There are {n_params} parameters (> {size_cutoff}). Using super-rough',
                          'Hessian approximations (square of gradient).')
                
                hess_diag = jnp.square(gradient(opt.x))
                inv_hess_diag = 1. / hess_diag
                
            else:

                raise AttributeError(f"Cannot calculate Hessian for {n_params} >= {size_cutoff}",
                                     "parameters without a gradient function.")          

    else:

        inv_hess_diag = np.diag(hess_inv)

    return np.sqrt(var_residuals_no_nan * inv_hess_diag)


def _make_result_tables(opt: OptimizeResult,
                        hvp: Callable[[ArrayLike], Array],
                        gradient: Callable[[ArrayLike], Array],
                        x0: Mapping,
                        x_info: Mapping) -> pd.DataFrame:
    
    results = defaultdict(list)
    _, x0_unflatten = ravel_pytree(x0)

    stderr = _extract_diag(opt, hvp, 
                           gradient=gradient,
                           nd=x_info[list(x_info)[0]][0][0].shape[0])
    
    estimates = dict(_est=opt.x, 
                     _se=stderr,
                     _lower95ci=opt.x - 1.96 * stderr,
                     _upper95ci=opt.x + 1.96 * stderr,
                     _wald=(opt.x / stderr) ** 2.,
                     _wald_p=chi2.sf((opt.x / stderr) ** 2., df=1))
    transf_blocklist = ('_se', '_wald', '_wald_p')
    _opt = {name: x0_unflatten(item) 
            for name, item in estimates.items()}

    for param_name, (x_col, transformation) in _PARAM_DEPENDENCE.items():

        these_transf = dict(_=lambda x: x, 
                            _val=transformation)
        
        if x_col is not None:
            _, these_categories = x_info[x_col]
        else:
            these_categories = [0]
            x_col = 'X'

        results[x_col] += [pd.DataFrame(f(o[param_name]),
                                        columns=[param_name + suffix + suffix2],
                                        index=these_categories)
                           for suffix2, f in these_transf.items()
                           for suffix, o in _opt.items()
                           if not ((suffix2 == '_val') and (suffix in transf_blocklist))]
            
    results = {(x_col or 'X'): pd.concat(dfs, axis=1).reset_index(names=x_col or 'X') 
               for x_col, dfs in results.items()}

    return results


def main() -> None:

    args = _parse_arguments()

    delimiter = dict(TSV='\t', TXT='\t', CSV=',')
    counts_data = pd.read_csv(args.input, 
                              sep=delimiter[args.format])
    # counts_data = read_table(args.input, format=args.format)
    
    featurization = dict(guide_name=(args.name, args.reference),
                         seq_group=(args.sequencing_group, [None for _ in args.sequencing_group]),
                         exp_group=(args.expansion_group, args.initial),
                         exp_seq_group=(args.expansion_group + args.sequencing_group, 
                                        args.initial + ['[pick]' for _ in args.sequencing_group]),
                         culture_group=(args.culture, [None for _ in args.culture]),
                         guide_culture_group=(args.name + args.culture, 
                                              [None] * (len(args.name) + len(args.culture))))

    counts_data, X, Y = _featurize(counts_data,
                                   x_cols=featurization,
                                   y_col=args.count)
    
    initial_parameters = _initialize_params(X)

    pprint_dict({name: p.shape for name, p in initial_parameters.items()},
                'Initialized model parameters')
    
    model, objective, jacobian, hvp = _make_objective(x0=initial_parameters)
    count_offset, initial_parameters = _pick_best_offset(objective, 
                                                         X, Y,
                                                         initial_parameters, jacobian)
    
    this_objective = partial(objective, x=_unpack_x(X), y=Y, offset=count_offset)
    this_gradient = partial(jacobian, x=_unpack_x(X), y=Y, offset=count_offset)
    this_hvp = hvp(_unpack_x(X), Y, count_offset)
    history, optimized = _fit_objective(this_objective,
                                        x0=initial_parameters,
                                        method=args.algorithm,
                                        jac=this_gradient,
                                        hessp=this_hvp)
    
    print_err('\n', optimized)
    
    result_tables = _make_result_tables(optimized, 
                                        this_hvp, 
                                        gradient=this_gradient,
                                        x0=initial_parameters, 
                                        x_info=X)

    for x_col, result_table in result_tables.items():
        
        this_filename = f'{args.output}_params-{x_col}.tsv'
        print_err(f'Writing results for parameters depending on {x_col} to {this_filename}...')

        try:
            renamer = featurization[x_col][0]
        except KeyError:
            renamer = [x_col]

        new_cols = result_table[x_col].astype(str).str.split(':', expand=True)
        new_cols.columns = renamer
        
        (pd.concat([new_cols, result_table], axis=1)
           .to_csv(this_filename, 
                   sep='\t',
                   index=False))
        
    _, prediction = model(p=optimized.x, 
                          x=_unpack_x(X), 
                          offset=count_offset)

    counts_data[f'{args.count}_fitted'] = prediction

    this_filename = f'{args.output}_{args.count}-fit.tsv'
    print_err(f'Writing fitted {args.count} to {this_filename}...')
    counts_data.to_csv(this_filename, 
                       sep='\t',
                       index=False)

    return None


if __name__ == '__main__':

    main()