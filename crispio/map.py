"""Classes for representing guide RNA libraries."""

from typing import Dict, Iterable, Optional, Tuple, Union

from dataclasses import asdict, dataclass, field
import math

from bioino import FastaSequence, GffFile, GffLine
from carabiner import pprint_dict
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

    Examples
    --------
    >>> GuideMatch(pam_search='NGG', guide_seq='ATCGATCG', pam_seq='CGG', pam_start=10, reverse=False)
    GuideMatch(pam_search='NGG', guide_seq='ATCGATCG', pam_seq='CGG', pam_start=10, reverse=False, guide_context_up=None, guide_context_down=None, pam_end=13, length=8, guide_start=2, guide_end=10)
    >>> GuideMatch(pam_search='NGG', guide_seq='ATCGATCG', pam_seq='CCG', pam_start=10, reverse=True)
    GuideMatch(pam_search='NGG', guide_seq='CGATCGAT', pam_seq='CGG', pam_start=10, reverse=True, guide_context_up=None, guide_context_down=None, pam_end=13, length=8, guide_start=13, guide_end=21)
    
    """

    pam_search: str
    guide_seq: str
    pam_seq: str 
    pam_start: int
    reverse: bool
    guide_context_up: Optional[str] = field(init=False, default=None)
    guide_context_down: Optional[str] = field(init=False, default=None)
    pam_end: int = field(init=False)
    length: int = field(init=False)
    guide_start: int = field(init=False)
    guide_end: int = field(init=False)

    def __post_init__(self):

        self.length = len(self.guide_seq)
        self.pam_end = self.pam_start + len(self.pam_seq)

        if not self.reverse:
            self.guide_start = self.pam_start - self.length 
            self.guide_end = self.pam_start 
        else:
            self.guide_start = self.pam_end
            self.guide_end = self.pam_end + self.length
            self.pam_seq = reverse_complement(self.pam_seq)
            self.guide_seq = reverse_complement(self.guide_seq)

    def __len__(self):

        return len(self.guide_seq)

    def __str__(self):

        return self.guide_seq
        
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

    def __iter__(self):

        try:
            for match in self.matches:
                yield match
        except ValueError as e:
            raise e

    def __len__(self):

        """Give the number of matches.
        
        If the matches are not yet instantiated, this will instantiate the matches. 
        Depending on the size of the search, it can take memory and time.
        
        """

        try:
            return len(self.matches)
        except TypeError:
            self.matches = list(self.matches)
            return len(self.matches)            

    @staticmethod
    def _from_search(guide_seq: str, 
                     genome: str,
                     pam_search: str = "NGG") -> Iterable[Tuple[int, Dict[str, GuideMatch]]]:

        pam_len = len(pam_search)

        guide_plus_pam = guide_seq + pam_search

        for reverse in (False, True):

            this_guide_plus_pam = (reverse_complement(guide_plus_pam) if reverse 
                                   else guide_plus_pam)
            
            for i, ((guide_pam_start, guide_pam_end), 
                     guide_pam_seq) in enumerate(find_iupac(this_guide_plus_pam, genome)):
                
                if not reverse:
                    pam_start = guide_pam_end - pam_len
                    pam_seq = guide_pam_seq[-pam_len:]
                    _guide_seq = guide_pam_seq[:-pam_len]
                else:
                    pam_start = guide_pam_start
                    pam_seq = guide_pam_seq[:pam_len]
                    _guide_seq = guide_pam_seq[pam_len:]

                gm = GuideMatch(pam_search=pam_search,
                                pam_seq=pam_seq, 
                                guide_seq=_guide_seq,
                                pam_start=pam_start, 
                                reverse=reverse)

                guide_down, guide_up = get_context(gm.pam_start, gm.pam_end,
                                                   gm.guide_start, gm.guide_end,
                                                   genome=genome, 
                                                   reverse=reverse)
                gm.guide_context_down = guide_down
                gm.guide_context_up = guide_up

                yield gm

    @classmethod
    def from_search(cls,
                    guide_seq: str,
                    genome: str,
                    pam_search: str = "NGG",
                    guide_name: Optional[str] = None,
                    in_memory: bool = False):
        
        """Find the location of a guide sequence in a genome.

        Searches the genome in the forward strand then the reverse strand,
        returning the match with an adjacent PAM in the order found.

        The default behavior is to find matches lazily to save memory and time.

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
        >>> gmc = GuideMatchCollection.from_search("TTTTTTTAAAAAAA", "CCGTTTTTTTAAAAAAACGG")
        >>> len(gmc)
        2
        >>> for match in gmc:
        ...     print(match)
        ... 
        TTTTTTTAAAAAAA
        TTTTTTTAAAAAAA

        """

        if guide_seq not in genome and reverse_complement(guide_seq) not in genome:
            raise ValueError(f'{guide_seq} not in genome')

        matches = cls._from_search(guide_seq, genome, pam_search)

        if in_memory:
            matches = list(matches)

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

    def __iter__(self):
        
        try:
            for match in self.guide_matches:
                yield match
        except ValueError as e:
            raise e
    
    def __len__(self):

        """Give the number of matches.
        
        If the matches are not yet instantiated, this will instantiate the matches. 
        Depending on the size of the search, it can take memory and time.
        
        """

        try:
            return len(self.guide_matches)
        except TypeError:
            self.guide_matches = list(self.guide_matches)
            return len(self.guide_matches)

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
        >>> genome = "ATATATATATATATATATATATATACCGTTTTTTTAAAAAAACGGATATATATATATAATATATATATATAATATATATATATA"
        >>> gl = GuideLibrary.from_generating(genome=genome)
        >>> for gff in gl.as_gff(gff_defaults=dict(seqid='my_seq', source='here', feature='protospacer')):  # doctest: +NORMALIZE_WHITESPACE
        ...     print(gff)
        ... 
        my_seq    here    protospacer     23      42      .       +       .       ID=sgr-06a4ba9b;Name=42-united_exodus;guide_context_down=ATATATATATATAATATATA;guide_context_up=ATATATATATATATATATAT;guide_length=20;guide_re_sites=;guide_sequence=ATACCGTTTTTTTAAAAAAA;guide_sequence_hash=a3987295;mnemonic=united_exodus;pam_end=45;pam_replichore=L;pam_search=NGG;pam_sequence=CGG;pam_start=42;source_name=42-united_exodus
        my_seq    here    protospacer     29      48      .       -       .       ID=sgr-f84d1c6a;Name=25-zigzag_state;guide_context_down=TATATATATATATATATATA;guide_context_up=ATATATATATTATATATATA;guide_length=20;guide_re_sites=;guide_sequence=TATCCGTTTTTTTAAAAAAA;guide_sequence_hash=188c9ee6;mnemonic=zigzag_state;pam_end=28;pam_replichore=R;pam_search=NGG;pam_sequence=CGG;pam_start=25;source_name=25-zigzag_state       
        """

        genome_length = len(self.genome)
        max = max or math.inf
        gff_defaults = gff_defaults or {}

        for guide_match_collection in self:

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
                    'pam_search': match.pam_search, 
                    'pam_sequence': match.pam_seq,
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
                    message=f'Not found: {len(not_found)} guides')

        return not_found

    @classmethod
    def from_mapping(cls,
                     guide_seq: Union[str, Iterable[str], FastaSequence, Iterable[FastaSequence]],
                     genome: str,
                     pam_search: str = "NGG",
                     in_memory: bool = False):
        
        """Map a set of expected guides to a genome.

        The default behavior is to find matches lazily to save memory and time.
        
        Parameters
        ----------
        guide_seq : str or bioino.FastaSequence or list
            Guides to map.
        genome : str
            Genome to map against.
        pam_search : str
            IUPAC PAM sequence to search against.
        in_memory : bool, optional
            Whether to instantiate matches in memory. Default: lazy matching.

        Returns
        -------
        GuideLibrary

        Examples
        --------
        >>> genome = "CCCCCCCCCCCTTTTTTTTTTAAAAAAAAAATGATCGATCGATCGAGGAAAAAAAAAACCCCCCCCCCC"
        >>> guide_seq = ["ATGATCGATCGATCG", "ATGATCGATCGATCGCCC"]
        >>> gl = GuideLibrary.from_mapping(guide_seq=guide_seq, genome=genome) 
        >>> for collection in gl:
        ...     for match in collection:
        ...             print(match.as_dict())
        ...
        {'pam_search': 'NGG', 'guide_seq': 'ATGATCGATCGATCG', 'pam_seq': 'AGG', 'pam_start': 45, 'reverse': False, 'guide_context_up': 'CTTTTTTTTTTAAAAAAAAA', 'guide_context_down': 'AAAAAAAAAACCCCCCCCCC', 'pam_end': 48, 'length': 15, 'guide_start': 30, 'guide_end': 45}

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
                
        matches = cls._from_mapping(guide_seq, genome, pam_search)

        if in_memory:
            matches = list(matches)
        
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

            directionality = 'reverse' if reverse else 'forward'
        
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
                        
                        t.set_postfix(direction=directionality,
                                      at_site=pam_start,
                                      pam_sites_found=found,
                                      guides_created=guides_created)
                        
                        gm = GuideMatch(pam_search=pam_search,
                                                 pam_seq=pam_seq, 
                                                 guide_seq=guide_seq,
                                                 pam_start=pam_start, 
                                                 reverse=reverse)
                        guide_down, guide_up = get_context(gm.pam_start, gm.pam_end,
                                                           gm.guide_start, gm.guide_end,
                                                           genome=genome, 
                                                           reverse=reverse)
                        gm.guide_context_down = guide_down
                        gm.guide_context_up = guide_up

                        # TODO: Actually group by sequence
                        yield GuideMatchCollection(guide_seq=guide_seq, 
                                                   pam_search=_pam_search,
                                                   matches=[gm])
                        
        return guides_created
                        
    @classmethod
    def from_generating(cls,
                        genome: str,
                        max_length: int = 20,
                        min_length: Optional[int] = None, 
                        pam_search: str = "NGG",
                        in_memory: bool = False):
        
        """Find all guides matching a PAM sequence in a given genome.

        The default behavior is to find matches lazily to save memory and time.

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
        in_memory : bool, optional
            Whether to instantiate matches in memory. Default: lazy matching.

        Examples
        --------
        >>> genome = "ATATATATATATATATATATATATACCGTTTTTTTAAAAAAACGGATATATATATATAATATATATATATAATATATATATATA"
        >>> gl = GuideLibrary.from_generating(genome=genome)
        >>> len(gl)
        2
        >>> for match_collection in gl:
        ...     for guide in match_collection:
        ...             print(guide)
        ... 
        ATACCGTTTTTTTAAAAAAA
        TATCCGTTTTTTTAAAAAAA

        """
        
        matches = (match for match in cls._from_generating(genome, max_length, min_length, pam_search))

        if in_memory:
            matches = list(matches)

        return cls(genome=genome, guide_matches=matches)
