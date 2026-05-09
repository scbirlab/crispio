from typing import Any, Callable, Dict, Iterable, List, Tuple, Optional, Union

from itertools import dropwhile, takewhile

from bioino import GffLine
from streq import correlation, gc_content, purine_content, reverse_complement

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
    return correlation(
        seq, 
        scaffold or reverse_complement(seq),
        wobble=True,
    )


def _pam_leading_n(
    gff: GffLine,
    scaffold: Optional[str] = None
) -> str:

    x = takewhile(
        lambda x: x[0] == 'N', 
        zip(
            gff.attributes['pam_search'], 
            gff.attributes['pam_sequence'],
        ),
    )
    return "".join(item[1] for item in x)


def _pam_defined(
    gff: GffLine, 
    scaffold: Optional[str] = None
) -> str:

    x = dropwhile(
        lambda x: x[0] == 'N', 
        zip(gff.attributes['pam_search'], 
        gff.attributes['pam_sequence'],
    ))
    return "".join(item[1] for item in x)


_FEATURIZERS = {
    'on_nontemplate_strand': lambda gff, _: gff.columns.strand != gff.attributes['ann_strand'],
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
    'guide_scaff_corr': lambda gff, scaffold: _autocorrelation(gff, 'guide_sequence', scaffold),
}


def get_features() -> List[str]:
    """Get the list of available features."""
    return list(_FEATURIZERS)


def featurize(
    gff: GffLine, 
    features: Optional[Union[str, Iterable[str]]] = None,
    scaffold: Optional[str] = None
) -> Union[int, str, Dict[str, Union[int, str]]]:
    
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

    Examples
    --------
    Build a representative GffLine (attributes required by the feature set):
 
    >>> from bioino import GffLine
    >>> gff_line = GffLine(
    ...     ['chr1', 'crispio', 'protospacer', 1, 20, '.', '+', '.'],
    ...     {
    ...         'guide_sequence':   'ATCGATCGATCGATCGATCG',
    ...         'pam_sequence':     'CGG',
    ...         'pam_search':       'NGG',
    ...         'guide_context_up': 'AAAAAAAAAAAAAAAAAACC',
    ...         'guide_context_down':'TTTTTTTTTTTTTTTTTTGG',
    ...         'ann_strand':       '-',
    ...     },
    ... )
 
    Single feature by name — returns the raw value (not wrapped in a dict):
 
    >>> from crispio.features import featurize
    >>> featurize(gff_line, 'guide_gc')
    '0.500'
    >>> featurize(gff_line, 'guide_purine')
    '0.500'
    >>> featurize(gff_line, 'seed_seq')
    'GATCG'
    >>> featurize(gff_line, 'guide_start3')
    'ATC'
    >>> featurize(gff_line, 'guide_end3')
    'TCG'
    >>> featurize(gff_line, 'pam_gc')
    '1.000'
    >>> featurize(gff_line, 'pam_n')
    'C'
    >>> featurize(gff_line, 'pam_def')
    'GG'
    >>> featurize(gff_line, 'context_up2')
    'CC'
    >>> featurize(gff_line, 'context_down2')
    'TT'
    >>> featurize(gff_line, 'on_nontemplate_strand')
    True
    >>> featurize(gff_line, 'guide_autocorr')
    '8.223'
    >>> featurize(gff_line, 'pam_autocorr')
    '1.500'
 
    List of features — returns a ``feat_``-prefixed dict:
 
    >>> featurize(gff_line, features=['guide_gc', 'seed_seq', 'guide_start3'])
    {'feat_guide_gc': '0.500', 'feat_seed_seq': 'GATCG', 'feat_guide_start3': 'ATC'}
 
    All features require a scaffold **sequence** (not a name string).
    Retrieve it from ``crispio.utils.sequences``:
 
    >>> from crispio.utils import sequences
    >>> scaffold_seq = sequences.scaffolds['Sth1']
    >>> result = featurize(gff_line, scaffold=scaffold_seq)
    >>> sorted(result.keys())   # doctest: +NORMALIZE_WHITESPACE
    ['feat_context_down2', 'feat_context_up2', 'feat_context_up_autocorr',
     'feat_guide_autocorr', 'feat_guide_end3', 'feat_guide_gc', 'feat_guide_purine',
     'feat_guide_scaff_corr', 'feat_guide_start3', 'feat_on_nontemplate_strand',
     'feat_pam_autocorr', 'feat_pam_def', 'feat_pam_gc', 'feat_pam_n',
     'feat_pam_scaff_corr', 'feat_seed_seq']
    >>> result['feat_guide_scaff_corr']
    '9.770'
    >>> result['feat_pam_scaff_corr']
    '2.667'
 
    Calling without a scaffold when computing all features raises
    ``AttributeError``, not ``TypeError``:
 
    >>> featurize(gff_line)
    Traceback (most recent call last):
        ...
    AttributeError: Scaffold must be provided to calculate all features.
 
    Unknown feature name raises ``KeyError``:
 
    >>> featurize(gff_line, 'not_a_feature')
    Traceback (most recent call last):
        ...
    KeyError: 'not_a_feature'
    
    """
    
    if features is None:
        features = get_features()
        if scaffold is None:
            raise AttributeError("Scaffold must be provided to calculate all features.")

    if isinstance(features, str):
        return _FEATURIZERS[features](gff, scaffold)
    elif isinstance(features, Iterable):
        return {f"feat_{feature}": _FEATURIZERS[feature](gff, scaffold)
                for feature in features}
    else:
        raise ValueError(f"Requested feature {features} is not a string or iterable.")


def get_context(
    pam_start: int, 
    pam_end: int,
    guide_start: int, 
    guide_end: int,
    genome: str,
    reverse: bool,
    extra_bases: int = 20
) -> Tuple[str, str]:
    """Get surrounding sequence.
    
    Examples
    ========
    Use a genome with visually distinct regions to make direction clear:
    ``AAAA|CCCC|GGGG|TTTT|ACGT|TGCA``  (blocks of 4, 24 bp total)
 
    >>> genome = 'AAAA' + 'CCCC' + 'GGGG' + 'TTTT' + 'ACGT' + 'TGCA'
 
    Forward strand — guide at [4:8], PAM at [8:12], context window of 4 nt:
    upstream context is the 4 nt *before* the guide; downstream is the 4 nt
    *after* the PAM:
 
    >>> get_context(pam_start=8, pam_end=12,
    ...             guide_start=4, guide_end=8,
    ...             genome=genome, reverse=False, extra_bases=4)
    ('TTTT', 'AAAA')
 
    Reverse strand — PAM at [4:8] (on forward), guide at [8:12]; context is
    reverse-complemented and directions are flipped:
 
    >>> get_context(pam_start=4, pam_end=8,
    ...             guide_start=8, guide_end=12,
    ...             genome=genome, reverse=True, extra_bases=4)
    ('TTTT', 'AAAA')
 
    Context window at the right edge of the genome is truncated gracefully —
    Python slice semantics give an empty string rather than an error:
 
    >>> get_context(pam_start=20, pam_end=24,
    ...             guide_start=16, guide_end=20,
    ...             genome=genome, reverse=False, extra_bases=4)
    ('', 'TTTT')

    """
    if not reverse:
        guide_down = genome[pam_end:(pam_end + extra_bases)] 
        guide_up = genome[(guide_start - extra_bases):guide_start]
    else:
        guide_down = reverse_complement(genome[(pam_start - extra_bases):pam_start])
        guide_up = reverse_complement(genome[guide_end:(guide_end + extra_bases)])
    return guide_down, guide_up
