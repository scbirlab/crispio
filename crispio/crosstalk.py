"""Tools for detecting guide crosstalk."""

from typing import Iterable, Optional, Tuple

from streq import hamming

from bioino import GffLine

def _get_mismatches(gff1: GffLine, 
                    gff2: GffLine, 
                    pairs_checked: Optional[Iterable] = None,
                    maximum: int = 4,
                    seed_length: int = 4) -> Tuple[Tuple[str, str], int]:

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
