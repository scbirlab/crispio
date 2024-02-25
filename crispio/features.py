from typing import (Any, Callable, Dict, Iterable, List, 
                    Tuple, Optional, Union)

from itertools import dropwhile, takewhile

from bioino import GffLine
from streq import (correlation, gc_content, purine_content, 
                   reverse_complement)

_FLOAT_FORMAT = '.3f'

def _format_float(f: Callable[[Any], float]) -> Callable[[Any], float]:

    def _f(*args, **kwargs):

        return format(f(*args, **kwargs), _FLOAT_FORMAT)

    return _f


@_format_float
def _autocorrelation(gff: GffLine, 
                     attribute: str,
                     scaffold: Optional[str] = None) -> float:
    
    seq = gff.attributes[attribute]

    return correlation(seq, 
                       scaffold or reverse_complement(seq),
                       wobble=True)


def _pam_leading_n(gff: GffLine,
                   scaffold: Optional[str] = None) -> str:

    x = takewhile(lambda x: x[0] == 'N', 
                  zip(gff.attributes['pam_search'], 
                      gff.attributes['pam_sequence']))

    return ''.join(item[1] for item in x)


def _pam_defined(gff: GffLine, 
                 scaffold: Optional[str] = None) -> str:

    x = dropwhile(lambda x: x[0] == 'N', 
                  zip(gff.attributes['pam_search'], 
                      gff.attributes['pam_sequence']))

    return ''.join(item[1] for item in x)


_FEATURIZERS = {'on_nontemplate_strand': lambda gff, _: gff.columns.strand != gff.attributes['ann_strand'],
                'context_up2': lambda gff, _: gff.attributes['guide_context_up'][-2:],
                'context_down2': lambda gff, _: gff.attributes['guide_context_down'][:2],
                'context_up_autocorr': lambda gff, _: _autocorrelation(gff, 'guide_context_up'),
                'pam_n': _pam_leading_n,
                'pam_def': _pam_defined,
                'pam_gc': lambda gff, _: _format_float(gc_content)(gff.attributes['pam_sequence']),
                'pam_autocorr': lambda gff, _: _autocorrelation(gff, 'pam_sequence'),
                'pam_scaff_corr': lambda gff, scaffold: _autocorrelation(gff, 'pam_sequence', scaffold),
                'guide_purine': lambda gff, _: _format_float(purine_content)(gff.attributes['guide_sequence']),
                'guide_gc': lambda gff, _: _format_float(gc_content)(gff.attributes['guide_sequence']),
                'seed_seq': lambda gff, _: gff.attributes['guide_sequence'][-5:],
                'guide_start3': lambda gff, _: gff.attributes['guide_sequence'][:3],
                'guide_end3': lambda gff, _: gff.attributes['guide_sequence'][-3:],
                'guide_autocorr': lambda gff, _: _autocorrelation(gff, 'guide_sequence'),
                'guide_scaff_corr': lambda gff, scaffold: _autocorrelation(gff, 'guide_sequence', scaffold)}


def get_features() -> List[str]:

    """Get the list of available features."""

    return list(_FEATURIZERS)


def featurize(gff: GffLine, 
              features: Optional[Union[str, Iterable[str]]] = None,
              scaffold: Optional[str] = None) -> Union[int, str, Dict[str, Union[int, str]]]:
    
    """Featurize a guide RNA represented by a `bioino.GffLine`.

    Depending on the feature to be calculated, the GFF should have attributes
    'pam_sequence', 'guide_sequence', 'guide_context_up', 'guide_context_down',
    and 'ann_strand'.

    Parameters
    ----------
    gff : bioino.GffLine
        Input guide RNA with additional attributes.
    features : str or list of str, optional
        The names of the features to be calculated. Default: calculate all.
    scaffold : str, optional
        Guide scaffold. Required for some features. If `features` is the default,
        scaffold must be provided.

    Returns
    -------
    dict, float, or str
        If `features` is a string, then returns the value of the feature.
        If it is a list, then returns a dictionary mapping feature names
        to values.

    Raises
    ------
    KeyError
        If any `features` are not supported.
    ValueError
        If `features` is neither a string nor iterable.
    AttributeError
        If `features` is default but `scaffold` is not provided.
    
    """
    
    if features is None:

        features = features or get_features()

        if scaffold is None:

            raise AttributeError("Scaffold must be provided to calculate all features.")

    if isinstance(features, str):

        return _FEATURIZERS[features](gff, scaffold)
    
    elif isinstance(features, Iterable):

        return {f"feat_{feature}": _FEATURIZERS[feature](gff, scaffold)
                for feature in features}
    
    else:

        raise ValueError(f"Requested feature {features} is not a string or iterable.")


def get_context(pam_start: int, 
                pam_end: int,
                guide_start: int, 
                guide_end: int,
                genome: str,
                reverse: bool,
                extra_bases: int = 20) -> Tuple[str, str]:
    
    if not reverse:
        guide_down = genome[pam_end:(pam_end + extra_bases)] 
        guide_up = genome[(guide_start - extra_bases):guide_start]
    else:
        guide_down = reverse_complement(genome[(pam_start - extra_bases):pam_start])
        guide_up = reverse_complement(genome[guide_end:(guide_end + extra_bases)])
    
    return guide_down, guide_up