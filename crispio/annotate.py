"""Tools for annotating guide RNAs from GFF data"""

from typing import Dict, Iterable, Mapping, Optional, Union

from dataclasses import asdict

from bioino import GffFile
from carabiner import print_err

_TAGS = (
    "Name", 
    "locus_tag", 
    "gene_biotype",
)
            
def annotate_from_gff(
    sgRNA: Mapping[str, Union[str, int]], 
    gff_data: GffFile,
    seqid: str,
    tags: Optional[Iterable[str]] = None,
) -> Dict[str, Union[str, int]]:
    
    """Annotate dictionary of guide information with GFF annotations.

    Dictionary must at least have key 'pam_start' and 'pam_end' mapping to 
    numerical values.

    Parameters
    ==========
    sgRNA : dict
        Dictionary containing 'pam_start' and 'pam_end', and optionally other
        information about a guide.
    gff_data : bioino.GffFile
        GffFile object which was loaded with `lookup=True`.
    tags : list of str, optional
        Which GFF tags to extract from attributes of GFF features.

    Returns
    =======
    dict
        Guide RNA dictionary updated with GFF annotations.
    
    """
    
    tags = tags or _TAGS
    pam_loc = (
        sgRNA['pam_start'] 
        + abs(sgRNA['pam_start'] 
        - sgRNA['pam_end']) // 2
    )

    results = gff_data.lookup_at(seqid, pam_loc)
    if not results:
        print_err(f"Warning: locus {pam_loc} on {seqid!r} not covered by GFF. Skipping annotation.")
        return sgRNA
    
    annotation_matches = results[0]
    # bioino 0.0.3 already bakes _up-/_down- into locus_tag for intergenic positions
    for tag in tags:
        try:
            sgRNA[f'ann_{tag}'] = annotation_matches.attributes[tag]
        except KeyError:
            pass

    sgRNA["pam_offset"] = annotation_matches.attributes["offset"]
    sgRNA.update({
        f"ann_{header}": val 
        for header, val in asdict(annotation_matches.columns).items()
    })
    return sgRNA
