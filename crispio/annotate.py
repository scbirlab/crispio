"""Tools for annotating guide RNAs from GFF data"""

from typing import Dict, Iterable, Mapping, Optional, Union

from dataclasses import asdict

from bioino import GffFile

Tguide = Dict[str, Union[str, int]]

_TAGS = ('Name', 'locus_tag', 'gene_biotype')
            
def annotate_from_gff(sgRNA: Tguide, 
                      gff_data: GffFile, 
                      tags: Optional[Iterable[str]] = None) -> Mapping:
    
    """Annotate dictionary of guide information with GFF annotations.

    Dictionary must at least have key 'pam_start' and 'pam_end' mapping to 
    numerical values.

    Parameters
    ----------
    sgRNA : dict
        Dictionary containing 'pam_start' and 'pam_end', and optionally other
        information about a guide.
    gff_data : bioino.GffFile
        GffFile object which was loaded with `lookup=True`.
    tags : list of str, optional
        Which GFF tags to extract from attributes of GFF features.

    Returns
    -------
    dict
        Guide RNA dictionary updated with GFF annotations.
    
    """
    
    tags = tags or _TAGS

    pam_loc = sgRNA['pam_start'] + abs(sgRNA['pam_start'] - sgRNA['pam_end']) // 2

    try:

        annotation_matches = gff_data._lookup[pam_loc][0]

    except IndexError:

        raise IndexError(f"Pam loc {pam_loc} is not annotated:\n"
                         f"{gff_data._lookup[pam_loc]}\n{gff_data._lookup[pam_loc - 1]}")

    for tag in tags:

        try:

            sgRNA[f'ann_{tag}'] = annotation_matches.attributes[tag]

        except KeyError:

            pass
        
    sgRNA['pam_offset'] = annotation_matches.attributes['offset']

    sgRNA.update({f'ann_{header}': val for header, val 
                  in asdict(annotation_matches.columns).items()})

    return sgRNA
