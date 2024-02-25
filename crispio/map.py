"""Classes for representing guide RNA libraries."""

from typing import Dict, Iterable, Optional, Union

from dataclasses import asdict, dataclass, field
import math

from bioino import FastaSequence, GffFile, GffLine
from carabiner import pprint_dict
from carabiner.cast import cast
from nemony import encode, hash as nm_hash
from streq import find_iupac, reverse_complement, which_re_sites
from tqdm.auto import tqdm

from .annotate import annotate_from_gff
from .features import get_context

@dataclass
class GuideMatch:

    """Information of guide matching a genome.

    Attributes
    ----------
    pam_search : str
        IUPAC search string for PAM.
    guide_seq : str
        Guide spacer sequence.
    pam_seq : str
        Actual PAM sequence.
    pam_start : int
        Chromosome coordinate of PAM start.
    pam_end : int
        Chromosome coordinate of PAM end.
    length : int
        Length of guide.

    """

    pam_search: str
    guide_seq: str
    pam_seq: str 
    pam_start: int
    pam_end: int
    reverse: bool
    guide_context_up: str = field(init=False, default=None)
    guide_context_down: str = field(init=False, default=None)
    length: int = field(init=False)
    guide_start: int = field(init=False)
    guide_end: int = field(init=False)


    def __post_init__(self):

        self.length = len(self.guide_seq)

        if not self.reverse:
            self.guide_start = self.pam_start - self.length 
            self.guide_end = self.pam_start 
            # guide_seq = guide_seq
        else:
            self.guide_start = self.pam_end
            self.guide_end = self.pam_end + self.length
            self.guide_seq = reverse_complement(self.guide_seq)
        
    def as_dict(self):

        return asdict(self)


@dataclass
class GuideMatchCollection:

    """Set of guides with the same sequence but potentially with multiple 
    matches.

    Attributes
    ----------
    guide_seq : str
        Guide spacer sequence.
    pam_search : str
        IUPAC search string for PAM.
    matches : iterable of GuideMatch
        Objects with matching information.
    guide_name : str, optional
        Name or identifier of guide.

    """

    guide_seq: str
    pam_search: str
    matches: Iterable[GuideMatch]
    guide_name: Optional[str] = field(default=None)

    @staticmethod
    def _from_search(guide_seq: str, 
                     genome: str,
                     pam_search: str = "NGG") -> Iterable[Dict[str, GuideMatch]]:
    
        if (guide_seq not in genome and 
            reverse_complement(guide_seq) not in genome):

            raise ValueError(f'{guide_seq} not in genome')

        pam_len = len(pam_search)

        guide_plus_pam = guide_seq + pam_search

        for reverse in (False, True):

            this_guide_plus_pam = (reverse_complement(guide_plus_pam) if reverse 
                                   else guide_plus_pam)
            
            for ((guide_pam_start, guide_pam_end), 
                 guide_pam_seq) in find_iupac(this_guide_plus_pam, genome):
                
                if not reverse:
                    pam_start, pam_end = guide_pam_end - pam_len, guide_pam_end
                    pam_seq = guide_pam_seq[-pam_len:]
                    _pam_search = pam_search
                    _guide_seq = guide_seq
                else:
                    pam_start, pam_end = guide_pam_start, guide_pam_start + pam_len
                    pam_seq = guide_pam_seq[:pam_len]
                    _pam_search = reverse_complement(pam_search)
                    _guide_seq = reverse_complement(guide_seq)
                
                yield GuideMatch(**dict(pam_search=_pam_search,
                                        pam_seq=pam_seq, 
                                        guide_seq=_guide_seq,
                                        pam_start=pam_start, 
                                        pam_end=pam_end,
                                        reverse=reverse))

    @classmethod
    def from_search(cls,
                    guide_seq: str,
                    genome: str,
                    pam_search: str = "NGG",
                    guide_name: Optional[str] = None):
        
        """Find the location of a guide sequence in a genome.

        Searches the genome in the forward strand then the reverse strand,
        returning the match with an adjacent PAM in the order found.

        Parameters
        ----------
        guide_seq : str
            The sequence of the guide to be found.
        pam_search : str, optional
            The sequence (IUPAC codes allowed) of the PAM to match. Default: "NGG".
        genome : str
            The genome sequence to search.
        guide_name : str
            Name or identifier of guide.

        Raises
        ------
        ValueError
            If guide not found in genome with appropriate PAM.

        Returns
        -------
        GuideMatches
            A iterator of dictionaries of match information.

        Examples
        --------
        >>> gmc = GuideMatchCollection.from_search("AAAAAAAAAAAAAAA", "AAAAAAAAAAAAAAACGG")
        >>> len(list(gmc.matches))
        1
        >>> for match in gmc.matches:
        ...     print(match)
        ... 
        GuideMatch(pam_search='NGG', guide_seq='AAAAAAAAAAAAAAA', pam_seq='CGG', pam_start=15, pam_end=18, reverse=False, guide_context_up=None, guide_context_down=None, length=15, guide_start=0, guide_end=15)

        """

        matches = list(match for match in cls._from_search(guide_seq, genome, pam_search))

        return cls(pam_search=pam_search, 
                   guide_seq=guide_seq, 
                   guide_name=guide_name,
                   matches=matches)
    

@dataclass
class GuideLibrary:

    """Library of guides from a genome.

    Attributes
    ----------
    genome : str
        Genome sequence that guides are matched to.
    guide_matches : list of GuideMatchCollection
        List of matches to the genome.

    """

    genome: str
    guide_matches: Iterable[GuideMatchCollection]

    def __post_init__(self):

        for guide_match_collection in self.guide_matches:

            for match in guide_match_collection.matches:

                guide_down, guide_up = get_context(match.pam_start, match.pam_end,
                                                   match.guide_start, match.guide_end,
                                                   genome=self.genome, 
                                                   reverse=match.reverse)
                match.guide_context_down = guide_down
                match.guide_context_up = guide_up


    def as_gff(self, 
               max: Optional[int] = None,
               annotations_from: Optional[GffFile] = None,
               tags: Optional[Iterable[str]] = None,
               gff_defaults: Optional[Dict[str, Union[str, int]]] = None) -> Iterable[GffLine]:

        """Convert into a iterable of `bioino.GffLine`s.

        Parameters
        ----------
        max : int, optional
            Number of `bioino.GffLine`s to return for each `GuideMatchCollection`. Default: return all.
        annotations_from : bioino.GffFile, optional
            If provided use the `lookup` table to annotate the returned `GffLine`s.
        tags : list of str, optional
            Which tags to take from `annotations_from`.
        gff_defaults : dict
            In case of missing values that are essential for GFF file formats
            (namely columns 1-8), take values from this disctionary.

        Yields
        ------
        bioino.GffLine
            Corresponding to a `GuideMatch`.

        Examples
        --------
        >>> genome = "TTTTTTTTTTAAAAAAAAAATGATCGATCGATCGNGGAAAAAAAAAACCCCCCCCCCC"
        >>> gl = GuideLibrary.from_generating(genome=genome)
        >>> for gff in gl.as_gff(gff_defaults=dict(seqid='my_seq', source='here', feature='protospacer')):  # doctest: +NORMALIZE_WHITESPACE
        ...     print(gff)
        ... 
        my_seq  here    protospacer     15      34      .       +       .       ID=sgr-c68ad8a1;Name=34-dreary_trident;guide_context_down=AAAAAAAAAACCCCCCCCCC;guide_context_up=;guide_length=20;guide_re_sites=;guide_sequence=AAAAAATGATCGATCGATCG;guide_sequence_hash=ab91540e;mnemonic=dreary_trident;pam_end=37;pam_replichore=L;pam_search=NGG;pam_sequence=NGG;pam_start=34;source_name=34-dreary_trident
        my_seq  here    protospacer     51      58      .       -       .       ID=sgr-5021e267;Name=47-vexed_sheriff;guide_context_down=TTTTTTTTTTCCNCGATCGA;guide_context_up=;guide_length=8;guide_re_sites=;guide_sequence=CCCCCCCC;guide_sequence_hash=acea9bbe;mnemonic=vexed_sheriff;pam_end=50;pam_replichore=L;pam_search=CCN;pam_sequence=GGG;pam_start=47;source_name=47-vexed_sheriff
        my_seq  here    protospacer     54      58      .       -       .       ID=sgr-9443d154;Name=50-wistful_pattern;guide_context_down=GGGTTTTTTTTTTCCNCGAT;guide_context_up=;guide_length=5;guide_re_sites=;guide_sequence=CCCCC;guide_sequence_hash=17b80bc7;mnemonic=wistful_pattern;pam_end=53;pam_replichore=L;pam_search=CCN;pam_sequence=GGG;pam_start=50;source_name=50-wistful_pattern
        my_seq  here    protospacer     57      58      .       -       .       ID=sgr-13a7ff38;Name=53-famous_jester;guide_context_down=GGGGGGTTTTTTTTTTCCNC;guide_context_up=;guide_length=2;guide_re_sites=;guide_sequence=CC;guide_sequence_hash=a56362a1;mnemonic=famous_jester;pam_end=56;pam_replichore=L;pam_search=CCN;pam_sequence=GGG;pam_start=53;source_name=53-famous_jester
        """

        genome_length = len(self.genome)
        max = max or math.inf
        gff_defaults = gff_defaults or {}

        for guide_match_collection in self.guide_matches:

            for i, match in enumerate(guide_match_collection.matches):

                if i >= max:
                    break

                sgrna_info = {
                    'ID': 'sgr-' + nm_hash((match.guide_seq, match.pam_search, match.pam_start), 8),
                    'mnemonic': encode((match.guide_seq, match.pam_search, match.pam_start)),
                    'guide_sequence_hash': nm_hash(match.guide_seq, 8),
                    'source_name': guide_match_collection.guide_name,
                    'pam_start': match.pam_start, 
                    'pam_end': match.pam_end,
                    'pam_search': (reverse_complement(match.pam_search) if match.reverse 
                                   else match.pam_search), 
                    'pam_sequence': (match.pam_seq if not match.reverse 
                                    else reverse_complement(match.pam_seq)),
                    'pam_replichore': 'R' if ((match.pam_start / genome_length) < 0.5) else 'L',
                    'strand': ('+' if not match.reverse else '-'),
                    'start': ((match.guide_start + 1) if (match.guide_start + 1) > 0 
                            else match.guide_start + 1 + genome_length), 
                    'end': (match.guide_end if (match.guide_start + 1) > 0 
                            else match.guide_end + genome_length), 
                    'guide_context_up': match.guide_context_up, 
                    'guide_context_down': match.guide_context_down,
                    'guide_length': match.length,
                    'guide_re_sites': ','.join(which_re_sites(match.guide_seq)),
                    'guide_sequence': match.guide_seq
                }
                sgrna_info['Name'] = '{pam_start}-{mnemonic}'.format(**sgrna_info)

                if annotations_from is not None:

                    sgrna_info = annotate_from_gff(sgrna_info, 
                                                   gff_data=annotations_from, 
                                                   tags=tags)
                    sgrna_info['Name'] = '{ann_Name}-{pam_start}-{mnemonic}'.format(**sgrna_info)
                
                sgrna_info.update(gff_defaults)
                sgrna_info['source_name'] = sgrna_info['source_name'] or sgrna_info['Name']

                yield GffLine.from_dict(sgrna_info)

    @staticmethod
    def _from_mapping(guide_seq: Iterable[FastaSequence],
                      genome: str,
                      pam_search: str = "NGG"):
        
        not_found = {}

        with tqdm(guide_seq) as t:  ## run a progress bar
    
            for guide_sequence in t:
                
                t.set_postfix(current=guide_sequence.name[:40], 
                              not_found=len(not_found))

                try:

                    guide_matches = GuideMatchCollection.from_search(guide_seq=guide_sequence.sequence, 
                                                                     guide_name=guide_sequence.name,
                                                                     pam_search=pam_search, 
                                                                     genome=genome)
                
                except ValueError:
                    
                    not_found[guide_sequence.name] = guide_sequence.sequence

                else:

                    yield guide_matches

        pprint_dict(not_found, 
                    f'Not found: {len(not_found)} guides')


    @classmethod
    def from_mapping(cls,
                     guide_seq: Union[str, Iterable[str], FastaSequence, Iterable[FastaSequence]],
                     genome: str,
                     pam_search: str = "NGG"):
        
        """Map a set of known guides to a genome.
        
        Parameters
        ----------
        guide_seq : str or bioino.FastaSequence or list
            Guides to map.
        genome : str
            Genome to map against.
        pam_search : str
            IUPAC PAM sequence to search against.

        Returns
        -------
        GuideLibrary

        Examples
        --------
        >>> genome = "TTTTTTTTTTAAAAAAAAAATGATCGATCGATCGNGGAAAAAAAAAACCCCCCCCCCC"
        >>> guide_seq = "ATGATCGATCGATCG"
        >>> gl = GuideLibrary.from_mapping(guide_seq=guide_seq, genome=genome) 
        >>> gl.genome
        'TTTTTTTTTTAAAAAAAAAATGATCGATCGATCGNGGAAAAAAAAAACCCCCCCCCCC'
        >>> len(gl.guide_matches)
        1
        >>> for collection in gl.guide_matches:
        ...     for match in collection.matches:
        ...             print(match)
        ... 
        GuideMatch(pam_search='NGG', guide_seq='ATGATCGATCGATCG', pam_seq='NGG', pam_start=34, pam_end=37, reverse=False, guide_context_up='', guide_context_down='AAAAAAAAAACCCCCCCCCC', length=15, guide_start=19, guide_end=34)

        """
        
        if isinstance(guide_seq, str):

            guide_seq = [guide_seq]
            
        if isinstance(guide_seq, Iterable):
            
            new_guide_seq = []

            for g in guide_seq:

                if isinstance(g, str):

                    g = FastaSequence(name=g, 
                                      description='query_spacer', 
                                      sequence=g)
                
                new_guide_seq.append(g)

            guide_seq = new_guide_seq
                
        matches = list(match for match in cls._from_mapping(guide_seq, genome, pam_search))
        
        return cls(genome=genome,
                   guide_matches=matches)
    
    @staticmethod
    def _from_generating(genome: str,
                         max_length: int = 20,
                         min_length: Optional[int] = None, 
                         pam_search: str = "NGG") -> Iterable[GuideMatchCollection]:

        min_length = min_length or max_length
        found, guides_created = 0, 0
        
        for reverse in (False, True):

            directionaility = 'reverse' if reverse else 'forward'
        
            _pam_search = (reverse_complement(pam_search) if reverse 
                           else pam_search)

            with tqdm(find_iupac(_pam_search, genome)) as t:  ## run a progress bar
                
                for (pam_start, pam_end), pam_seq in t:

                    found += 1

                    for length in range(min_length, max_length + 1):

                        guides_created += 1
                        
                        guide_start = (pam_start - length if not reverse 
                                       else pam_end)
                        guide_end = (pam_start if not reverse else 
                                     pam_end + length)

                        guide_seq = genome[guide_start:guide_end]
                        _guide_seq = (guide_seq if not reverse 
                                      else reverse_complement(guide_seq))
                        
                        t.set_postfix(direction=directionaility,
                                      at_site=pam_start,
                                      pam_sites_found=found,
                                      guides_created=guides_created)
                        
                        guide_match = GuideMatch(**dict(pam_search=pam_search,
                                                        pam_seq=pam_seq, 
                                                        guide_seq=_guide_seq,
                                                        pam_start=pam_start, 
                                                        pam_end=pam_end,
                                                        reverse=reverse))

                        yield GuideMatchCollection(guide_seq=guide_seq, 
                                                   pam_search=_pam_search,
                                                   matches=[guide_match])
                        
    @classmethod
    def from_generating(cls,
                        genome: str,
                        max_length: int = 20,
                        min_length: Optional[int] = None, 
                        pam_search: str = "NGG"):
        
        """Find all guides matching a PAM sequence in a given genome.

        Parameters
        ----------
        genome : str
            Genome sequence to search.
        max_length : int, optional
            Maximum guide length. Default: 20.
        min_length : int, optional
            Minimum guide length. Default: same as max_length.
        pam_search : str, optional
            IUPAC PAM sequence to search for. Default: "NGG".

        Examples
        --------
        >>> genome = "TTTTTTTTTTAAAAAAAAAATGATCGATCGATCGNGGAAAAAAAAAACCCCCCCCCCC"
        >>> gl = GuideLibrary.from_generating(genome=genome) 
        >>> gl.genome
        'TTTTTTTTTTAAAAAAAAAATGATCGATCGATCGNGGAAAAAAAAAACCCCCCCCCCC'
        >>> len(gl.guide_matches)
        4
        >>> for match_collection in gl.guide_matches:
        ...     for guide in match_collection.matches:
        ...             print(guide)
        ... 
        GuideMatch(pam_search='NGG', guide_seq='AAAAAATGATCGATCGATCG', pam_seq='NGG', pam_start=34, pam_end=37, reverse=False, guide_context_up='', guide_context_down='AAAAAAAAAACCCCCCCCCC', length=20, guide_start=14, guide_end=34)
        GuideMatch(pam_search='NGG', guide_seq='CCCCCCCC', pam_seq='CCC', pam_start=47, pam_end=50, reverse=True, guide_context_up='', guide_context_down='TTTTTTTTTTCCNCGATCGA', length=8, guide_start=50, guide_end=58)
        GuideMatch(pam_search='NGG', guide_seq='CCCCC', pam_seq='CCC', pam_start=50, pam_end=53, reverse=True, guide_context_up='', guide_context_down='GGGTTTTTTTTTTCCNCGAT', length=5, guide_start=53, guide_end=58)
        GuideMatch(pam_search='NGG', guide_seq='CC', pam_seq='CCC', pam_start=53, pam_end=56, reverse=True, guide_context_up='', guide_context_down='GGGGGGTTTTTTTTTTCCNC', length=2, guide_start=56, guide_end=58)
        
        """
        
        matches = list(match for match in cls._from_generating(genome, max_length, min_length, pam_search))

        return cls(genome=genome, guide_matches=matches)
