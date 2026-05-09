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

    Examples
    ========
    Set up a minimal single-gene GFF and build its lookup table:
 
    >>> from io import StringIO
    >>> from bioino import GffFile
    >>> gff_line = '\\t'.join([
    ...     'chr1', 'RefSeq', 'gene', '100', '500', '.', '+', '.',
    ...     'ID=g1;Name=geneA;locus_tag=b0001',
    ... ])
    >>> gff = GffFile.from_file(StringIO(gff_line), lookup=True)
 
    Guide PAM midpoint at position 251, inside the gene body (offset from gene
    start = 251 - 100 = 151):
 
    >>> from crispio.annotate import annotate_from_gff
    >>> result = annotate_from_gff({'pam_start': 250, 'pam_end': 253}, gff, seqid='chr1')
    >>> result['ann_Name']
    'geneA'
    >>> result['ann_locus_tag']
    'b0001'
    >>> result['pam_offset']
    151
    >>> result['ann_strand']
    '+'
    >>> result['ann_start'], result['ann_end']
    (100, 500)
 
    Intergenic guide (PAM midpoint 61, upstream of gene start): bioino 0.0.3
    automatically assigns the ``_up-`` prefix and computes the distance to the
    nearest feature, so no manual prefix logic is needed in crispio:
 
    >>> result2 = annotate_from_gff({'pam_start': 60, 'pam_end': 63}, gff, seqid='chr1')
    >>> result2['ann_locus_tag']
    '_up-geneA'
    >>> result2['pam_offset']
    39
 
    Unknown ``seqid`` (e.g. a plasmid not in the GFF) returns the input dict
    unchanged rather than raising:
 
    >>> result3 = annotate_from_gff({'pam_start': 250, 'pam_end': 253}, gff, seqid='chrX')
    >>> sorted(result3.keys())
    ['pam_end', 'pam_start']
 
    Custom tag set — only extract ``Name``:
 
    >>> result4 = annotate_from_gff({'pam_start': 250, 'pam_end': 253}, gff,
    ...                              seqid='chr1', tags=['Name'])
    >>> 'ann_Name' in result4
    True
    >>> 'ann_locus_tag' in result4
    False
    
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
