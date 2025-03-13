"""Tools for annotating guide RNAs from GFF data"""

from typing import Dict, Iterable, Mapping, Optional, Union

from dataclasses import asdict

from bioino import GffFile
from carabiner import print_err

_TAGS = (
    'Name', 
    'locus_tag', 
    'gene_biotype',
)
            
def annotate_from_gff(
    sgRNA: Mapping[str, Union[str, int]], 
    gff_data: GffFile, 
    tags: Optional[Iterable[str]] = None,
) -> Dict[str, Union[str, int]]:
    
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
    pam_loc = (
        sgRNA['pam_start'] 
        + abs(sgRNA['pam_start'] 
        - sgRNA['pam_end']) // 2
    )

    try:
        annotation_matches = gff_data._lookup[pam_loc][0]
    except KeyError as e:
        max_key = max(map(int, gff_data._lookup))
        print_err(e, "\n", f"Locus {pam_loc} not present in parsed GFF data. Maximum locus is {max_key}. Defaulting to this feature.")
        annotation_matches = gff_data._lookup[max_key][0]
        past_max = True
    else:
        past_max = False
        
    for tag in tags:
        if tag == "locus_tag" and past_max:
            source_tag = "Name"
            if annotation_matches.columns.strand == "+":
                prefix = "_down-"
            else:
                prefix = "_up-"
        else:
            source_tag = tag
            prefix = ""
        try:
            sgRNA[f'ann_{tag}'] = prefix + annotation_matches.attributes[source_tag]
        except KeyError:
            pass
        
    sgRNA['pam_offset'] = annotation_matches.attributes['offset']
    sgRNA.update({
        f'ann_{header}': val for header, val in asdict(annotation_matches.columns).items()
    })

    return sgRNA
