"""Tools for detecting guide crosstalk."""

from typing import Iterable, Optional, Tuple

from streq import hamming

from bioino import GffLine

def _get_mismatches(gff1: GffLine, 
                    gff2: GffLine, 
                    pairs_checked: Optional[Iterable] = None,
                    maximum: int = 4,
                    seed_length: int = 4) -> Tuple[Tuple[str, str], int]:
    """Check if two guides crosstalk with each other.

    Examples
    ========
    >>> from bioino import GffLine
 
    Two guides sharing seed sequence ``GATC`` with one mismatch in the
    non-seed region should be flagged:
 
    >>> g1 = GffLine(['chr1', 'src', 'protospacer', 1, 20, '.', '+', '.'],
    ...              {'ID': 'g1',
    ...               'guide_sequence': 'ATCGATCGATCGATCGGATC',
    ...               'pam_start': 20})
    >>> g2 = GffLine(['chr1', 'src', 'protospacer', 30, 50, '.', '+', '.'],
    ...              {'ID': 'g2',
    ...               'guide_sequence': 'TTCGATCGATCGATCGGATC',
    ...               'pam_start': 50})
    >>> pair, mismatches = _get_mismatches(g1, g2)
    >>> pair
    ('g1', 'g2')
    >>> mismatches
    {'g2': 1}
 
    Different seed sequence — no crosstalk regardless of non-seed similarity:
 
    >>> g3 = GffLine(['chr1', 'src', 'protospacer', 60, 80, '.', '+', '.'],
    ...              {'ID': 'g3',
    ...               'guide_sequence': 'ATCGATCGATCGATCGTTTG',
    ...               'pam_start': 80})
    >>> _, mismatches_no_seed = _get_mismatches(g1, g3)
    >>> mismatches_no_seed
    {}
 
    Pair is skipped when both guides target the same genomic position
    (``pam_start`` identical — comparing a guide against itself):
 
    >>> _, mismatches_self = _get_mismatches(g1, g1)
    >>> mismatches_self
    {}
 
    Pair already in ``pairs_checked`` is skipped (prevents double-counting
    in the symmetric comparison loop):
 
    >>> _, mismatches_dup = _get_mismatches(g1, g2, pairs_checked={('g1', 'g2')})
    >>> mismatches_dup
    {}
 
    The returned ``pair`` is always sorted so ``(g1, g2) == (g2, g1)``:
 
    >>> pair_reversed, _ = _get_mismatches(g2, g1)
    >>> pair_reversed
    ('g1', 'g2')

    """

    if pairs_checked is None:
        pairs_checked = set()

    query_id, query_seq, query_pam_start = (gff1.attributes['ID'], gff1.attributes['guide_sequence'], 
                                            gff1.attributes['pam_start'])
    ref_id, ref_seq, ref_pam_start = (gff2.attributes['ID'], gff2.attributes['guide_sequence'], 
                                      gff2.attributes['pam_start'])

    query_seed = query_seq[-seed_length:]
    mismatches = {}

    pair = tuple(sorted((query_id, ref_id)))

    if (query_pam_start != ref_pam_start and 
        pair not in pairs_checked and 
        query_id != ref_id): 

        same_seed = ref_seq.endswith(query_seed) 
        is_protospacer = gff2.columns.feature == 'protospacer'

        if is_protospacer and same_seed:

            distance = hamming(query_seq[:-seed_length], 
                               ref_seq[:-seed_length])
            
            close_match = distance <= maximum

            if close_match:

                mismatches = {ref_id: distance}

    return pair, mismatches
