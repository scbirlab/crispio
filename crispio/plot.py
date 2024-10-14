from typing import Callable, Iterable, Optional, Tuple

from argparse import FileType, Namespace, ArgumentParser
from itertools import chain
import sys

from carabiner import colorblind_palette, print_err, pprint_dict
from carabiner.mpl import grid
from pandas import DataFrame, read_csv, merge
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

TFigAx = Tuple[plt.Figure, plt.Axes]

rcParams["font.family"] = 'DejaVu Sans'
cbpal = colorblind_palette()

def _parse_arguments() -> Namespace:

    parser = ArgumentParser(description='''
    Plot fitness results from TSV or CSVs emitted from `guidefitness`.
    ''')
    parser.add_argument('--fitness', '-p', 
                        type=FileType('r'), 
                        required=True,
                        help=('CSV/TSV containing fitness paramemeters. '
                              'Required'))
    parser.add_argument('--expansion', '-x', 
                        type=FileType('r'), 
                        required=True,
                        help='CSV/TSV containing expansion paramemeters. Required')
    parser.add_argument('--essentials', '-s', 
                        type=FileType('r'), 
                        default=None,
                        help='CSV containing essentiality data for comparison. Must be CSV (not TSV) format.')
    parser.add_argument('--essential_calls', 
                        type=str, 
                        default=None,
                        help='Column of essentials table containing text-label essentials calls.')
    parser.add_argument('--essential_scores',
                        type=str, 
                        default=None,
                        help='Column of essentials table containing numerical essentials scores.')
    # parser.add_argument('--seq', '-q', 
    #                     type=argparse.FileType('r'), 
    #                     help='CSV containing seq_group paramemeters. Required')
    parser.add_argument('--fitted', '-t', 
                        type=FileType('r'), 
                        help='CSV containing fitted count data. Required')
    parser.add_argument('--output', '-o', 
                        type=str,
                        required=True,
                        help='Output file prefix. Required')
    parser.add_argument('--control_column', '-m', 
                        type=str,
                        default=None,
                        help='Column to label negative controls.')
    parser.add_argument('--negative', '-n', 
                        type=str,
                        default=None,
                        help=('Search term for negative controls. '
                              'Matches are labelled.'))
    parser.add_argument('--reference', '-r', 
                        type=str,
                        default=None,
                        help='Name of reference.')
    parser.add_argument('--initial', '-i', 
                        type=str,
                        help=('Name of condition indicating the initial (t = 0) '
                              'expansion_group.'))
    parser.add_argument('--count', '-c', 
                        type=str,
                        default='guide_count',
                        help='Column name containing counts. '
                        'Default: %(default)s')
    parser.add_argument('--format', '-f', 
                        type=str,
                        choices=['TSV', 'CSV'],
                        default='TSV',
                        help='Format of input tables. Default: %(default)s')

    args = parser.parse_args()

    key_val_str = [f'{key}: {val}' for key, val in vars(args).items()]

    pprint_dict(args, 
                message='Plotting fitness results with the following parameters') 

    return args


def violin(df: DataFrame, 
           x: str, 
           y: str,
           w: float = 2.5,
           h: float = 2.5,
           optional: Optional[Iterable] = None,
           **kwargs) -> TFigAx:
    
    optional = optional or []

    groupings = df.groupby(x)
    labels = [label for (label, _) in groupings]
    data = [_df[y] for (_, _df) in groupings]

    if len(labels) > 2:
        h += .3

    fig, ax = grid(panel_size=w, aspect_ratio=w / h)

    parts = ax.violinplot(data, 
                          showmedians=True,
                          showextrema=True,
                          points=1000)
    
    for pc in parts['bodies']:
        pc.set_facecolor('steelblue')
        pc.set_edgecolor('dimgray')
        pc.set_alpha(.85)

    parts['cmedians'].set_color('lightgray')

    for opt in optional:
        opt(ax)

    ax.set(**kwargs)
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(.25, len(labels) + .75)

    if len(labels) > 2:
        ax.tick_params(axis='x', labelrotation=90)

    return fig, ax


def boxplot(df: DataFrame, 
           x: str, 
           y: str,
           w: float = 2.5,
           h: float = 2.5,
           optional: Optional[Iterable] = None,
           **kwargs) -> TFigAx:

    optional = optional or []

    groupings = df.groupby(x)
    labels = [label for (label, _) in groupings]
    data = [_df[y] for (_, _df) in groupings]

    if len(labels) > 2:
        h += .3

    fig, ax = grid(panel_size=w, aspect_ratio=w / h)

    parts = ax.boxplot(x=data, 
                       labels=labels,
                       sym='.',
                       meanline=True,
                       patch_artist=True,
                       flierprops=dict(color='dimgray', 
                                       alpha=1., 
                                       markersize=.1),
                       medianprops=dict(color='dimgray'),
                       boxprops=dict(color='dimgray',
                                     facecolor='steelblue',
                                     alpha=.8))

    for opt in optional:
        opt(ax)

    ax.set(**kwargs)
    
    if len(labels) > 2:
        ax.tick_params(axis='x', labelrotation=90)

    return fig, ax


def timeline(df: DataFrame, 
             x: str, 
             yline: str,
             ypoint: str,
             guide_col: str,
             essentiality_col: Optional[list] = None,
             sample_size: int = 10,
             w: float = 2.5,
             h: float = 2.5,
             optional: Optional[Iterable] = None,
             **kwargs) -> TFigAx:
    
    essentiality_col = essentiality_col or []
    optional = optional or []
    
    if not isinstance(essentiality_col, list):
        essentiality_col = [essentiality_col]
     
    ref_guide = df.query('_reference')[guide_col].unique()[0]
    guides_to_plot = (df[[guide_col, '_neg_control'] + essentiality_col]
                      .groupby(['_neg_control'] + essentiality_col)[guide_col]
                      .sample(n=sample_size, replace=True)
                      .unique().tolist()) + [ref_guide]
    n_guides = len(guides_to_plot)

    data_to_plot = (df[df[guide_col].isin(guides_to_plot)]
                    .sort_values(x)
                    .groupby([guide_col, '_neg_control'] + 
                             essentiality_col))

    n_rows = n_guides // 6
    n_cols = np.ceil(n_guides / n_rows).astype(int)

    fig, axes = grid(ncol=n_cols, nrow=n_rows,
                   panel_size=w, aspect_ratio=w / h,
                   squeeze=False,
                   sharex=True, sharey=True)

    for ax, (group_names, group_df) in zip(fig.axes, data_to_plot):

        mean_df = group_df.groupby(x)[yline].mean().reset_index()

        ax.plot(mean_df[x], mean_df[yline],
                color=cbpal[1])
        ax.scatter(group_df[x], group_df[ypoint],
                   color=cbpal[1], s=2.)
        
        for opt in optional:
            opt(ax)

        ax.set_title('\n'.join(map(str, group_names)))
        ax.set(**kwargs)

    for ax in fig.axes:
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelleft=True)

    return fig, axes


def plot(plotf: Callable,
         w: float = 4.,
         h: float = 2.5, 
         abline: bool = False,
         optional: Optional[Iterable] = None,
         **kwargs) -> TFigAx:
    
    optional = optional or []
    
    fig, ax = grid(panel_size=w, 
                   aspect_ratio=w / h)

    queries = dict(samples='(not _neg_control) and (not _reference) and (not _initial)', 
                   initial='_initial',
                   negative='_neg_control', 
                   reference='_reference')
    
    for i, (color, (label, query)) in enumerate(zip(chain(('dimgray',), cbpal), 
                                                queries.items())):
    
        ax = plotf(ax, i, color, label, query)

    if abline:
        ax.plot(ax.get_ylim(), ax.get_ylim(),
                color='lightgrey', 
                zorder=-1)
        
    for opt in optional:
        opt(ax)

    ax.set(**kwargs)
    ax.legend(bbox_to_anchor=(1.04, 1), 
              loc="upper left")

    return fig, ax


def volcano(df: DataFrame,
            x: str,
            p: str) -> TFigAx:
    
    def _volcano(ax: plt.Axes, 
                 i: int, 
                 color: str, 
                 label: str, 
                 query: str) -> plt.Axes:
        
        this_df = df.query(query)
        ax.scatter(this_df[x],
                   -np.log10(this_df[p]),
                   s=.1 + 2 * (i > 0),
                   alpha=.5,
                   color=color,
                   label=label,
                   zorder=i)         

        return ax

    return _volcano


def scatter(df: DataFrame,
            x: str,
            y: str) -> TFigAx:
    
    def _scatter(ax: plt.Axes, 
                 i: int, 
                 color: str, 
                 label: str, 
                 query: str) -> plt.Axes:
        
        this_df = df.query(query)
        # print(this_df[[x, y]])
        ax.scatter(this_df[x], this_df[y],
                   s=.1 + 2 * (i > 0),
                   alpha=.5,
                   color=color,
                   label=label,
                   zorder=i)
        
        return ax
    
    return _scatter


def histogram(df: DataFrame, 
              x: str,
              bins: int = 40,
              logbins: bool = False,
              density: bool = False) -> Callable:
    
    x_nonzero = df[x][df[x] > 0]

    bins = np.geomspace(x_nonzero.min(), x_nonzero.max(), 
                        num=bins)

    def _histogram(ax: plt.Axes, 
                   i: int, 
                   color: str, 
                   label: str, 
                   query: str) -> plt.Axes:
        
        this_df = df.query(query)
        _, _, patches = ax.hist(this_df[x],
                                bins=bins,
                                color=color,
                                label=label,
                                histtype='stepfilled',
                                density=density,
                                zorder=i)
        
        for patch in patches:
            patch.set_alpha(.7)
        
        return ax
        
    return _histogram


def pointline(df: DataFrame,
              x: str,
              y: str,
              upper: str,
              lower: str) -> Callable:
    
    def _pointline(ax: plt.Axes, 
                   i: int, 
                   color: str, 
                   label: str, 
                   query: str) -> plt.Axes:
        
        this_df = df.query(query).sort_values(x)
        ax.errorbar(x=this_df[x], y=this_df[y],
                    yerr=np.abs(this_df[[lower, upper]].values -
                                this_df[[y]].values).T,
                    color=color,
                    label=label,
                    fmt='o-',
                    markerfacecolor=color,
                    capsize=2.,
                    markersize=3.,
                    zorder=i)

        return ax

    return _pointline


def _savefig(fig: plt.Figure, 
             filename: str) -> None:

    this_filename = filename + '.png'

    print_err(f'Saving {this_filename}...')

    fig.savefig(this_filename, 
                bbox_inches='tight',
                facecolor='white',
                dpi=600.)
    
    return None
    

def main() -> None:

    args = _parse_arguments()

    delimiter = dict(TSV='\t', TXT='\t', CSV=',')

    (counts_data, 
     fitness_data,
     exp_data) = ((read_csv(arg, sep=delimiter[args.format])
                       .assign(_neg_control=False, 
                               _reference=False, 
                               _initial=False))
                       for arg in (args.fitted, args.fitness, args.expansion)) 
    
    # count_totals = counts_data.groupby('seq_group')[args.count].sum().reset_index()
    # seq_data_totals = pd.merge(seq_data, count_totals)

    fitness_col = 'fitness_ratio_est_val'
    all_lt_zero = np.all(fitness_data[fitness_col] <= 0.)
    use_log_fitness = 'log' if not all_lt_zero else 'linear'

    if args.essentials is not None:

        essentiality_data = read_csv(args.essentials)
        
        fitness_data = merge(fitness_data, essentiality_data,
                                how='left')
        counts_data = merge(counts_data, essentiality_data,
                               how='left')

        fitness_data[args.essential_calls] = fitness_data[args.essential_calls].fillna('NotInEssData')
        counts_data[args.essential_calls] = counts_data[args.essential_calls].fillna('NotInEssData')

    for d in (counts_data, fitness_data):
        
        if (args.control_column is not None and 
            args.negative is not None):

            if len(args.negative) == 0 or args.negative == "''":
                d['_neg_control'] = (d[args.control_column].isnull())
            else:
                d['_neg_control'] = d[args.control_column].str.contains(args.negative)

    if args.reference is not None:
        counts_data['_reference'] = counts_data['guide_name'] == args.reference

    if args.initial is not None:
        counts_data['_initial'] = counts_data['exp_group'].astype(str) == args.initial

    if args.essentials is not None:
        if args.essential_calls is not None:
            fig, ax = boxplot(fitness_data, 
                            x=args.essential_calls,
                            y=fitness_col,
                            xlabel='Essentiality call', ylabel='Fitness',
                            yscale=use_log_fitness,
                            optional=[lambda x: x.axhline(1., color='lightgray', zorder=-1)])
            
            _savefig(fig, f'{args.output}_ess-calls-vs-fitness')

        if args.essential_scores is not None:
            fig, ax = plot(scatter(fitness_data, 
                                   x=fitness_col,
                                   y=args.essential_scores),
                           xlabel='Fitness', ylabel='Essentiality score',
                           xscale=use_log_fitness,
                           optional=[lambda x: x.axvline(1., color='lightgray', 
                                                         zorder=-1)])
            
            _savefig(fig, f'{args.output}_ess-scores-vs-fitness')

    any_p_gt0 = np.any(fitness_data['fitness_ratio_wald_p_'] > 0.)
    fig, ax = plot(scatter(fitness_data, 
                           x=fitness_col,
                           y='fitness_ratio_wald_p_'),
                   xlabel='Fitness ratio', ylabel='Wald p-value',
                   xscale=use_log_fitness, 
                   yscale='log' if any_p_gt0 else 'linear',
                   optional=[lambda x: x.axvline(1., color='lightgray', 
                                                 zorder=-1)])
    
    _savefig(fig, f'{args.output}_volcano')
    
    fig, ax = plot(scatter(counts_data, 
                           x=f'{args.count}_fitted',
                           y=args.count),
                   xlabel='Model', ylabel='Observed',
                   xscale='log', yscale='log', 
                   abline=True)
    
    _savefig(fig, f'{args.output}_fit-vs-obs')

    fig, ax = timeline(counts_data, 
                       x='exp_group',
                       yline=f'{args.count}_fitted',
                       ypoint=args.count,
                       guide_col='guide_name',
                       essentiality_col=args.essential_calls,
                       xlabel='Expansion group', ylabel='Count',
                       yscale='log')
    
    _savefig(fig, f'{args.output}_fit-vs-obs_exp')
    
    fig, ax = plot(scatter(fitness_data, 
                           x='log_inoculum_est_val',
                           y=fitness_col),
                   xlabel='$\log$(Inoculum ratio)', 
                   ylabel='Fitness ratio',
                   yscale=use_log_fitness,
                   optional=[lambda x: x.axvline(0., color='lightgray', zorder=-1),
                             lambda x: x.axhline(1., color='lightgray', zorder=-1)])
    
    _savefig(fig, f'{args.output}_fitness-vs-inoculum')
    
    fig, ax = plot(histogram(fitness_data, 
                             x='log_inoculum_est_val',
                             density=True),
                   xlabel='$\log$(Inoculum ratio)', ylabel='Density',
                   optional=[lambda x: x.axvline(0., color='lightgray', zorder=-1)])
    
    _savefig(fig, f'{args.output}_inoculum-hist-density')

    fig, ax = plot(histogram(fitness_data, 
                             x='log_inoculum_est_val',
                             density=False),
                   xlabel='$\log$(Inoculum ratio)', ylabel='Density',
                   optional=[lambda x: x.axvline(0., color='lightgray', zorder=-1)])
    
    _savefig(fig, f'{args.output}_inoculum-hist')
    
    fig, ax = plot(histogram(fitness_data, 
                             x=fitness_col,
                             logbins=True,
                             density=True),
                   xlabel='Fitness ratio', ylabel='Density',
                   xscale=use_log_fitness,
                   optional=[lambda x: x.axvline(1., color='lightgray', zorder=-1)])
    
    _savefig(fig, f'{args.output}_fitness-hist-density')

    fig, ax = plot(histogram(fitness_data, 
                             x=fitness_col,
                             logbins=True,
                             density=False),
                   xlabel='Fitness ratio', ylabel='Density',
                   xscale=use_log_fitness,
                   optional=[lambda x: x.axvline(1., color='lightgray', zorder=-1)])
    
    _savefig(fig, f'{args.output}_fitness-hist')
    
    try:
        fig, ax = plot(pointline(exp_data, 
                                x='exp_group',
                                y='log_expansion_est_val',
                                upper='log_expansion_upper95ci_val', 
                                lower='log_expansion_lower95ci_val'),
                    xlabel='Expansion group', 
                    ylabel='$\log$(Expansion)',
                    optional=[lambda x: x.axhline(0., color='lightgray', zorder=-1)])
    except ValueError:  # no positive values (probably 95ci)
        print_err('WARNING: No positibe values to plot for Expansion.')
    else:
        _savefig(fig, f'{args.output}_expansion')
    

    # fig, ax = plot(scatter(seq_data_totals, 
    #                        x='log_exposure_ratio_est_val',
    #                        y=args.count),
    #                xlabel='log exposure', 
    #                ylabel='Total counts',
    #                yscale='log')
    
    # this_filename = f'{args.output}_count-vs-exposure.pdf'
    # print(f'Saving {args.count} vs exposure plot as {this_filename}...',
    #       file=sys.stderr)
    # fig.savefig(this_filename, 
    #             bbox_inches='tight')

    return None


if __name__ == '__main__':

    main()