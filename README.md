# 🌱 crispio

![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/scbirlab/crispio/python-publish.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/crispio)
![PyPI](https://img.shields.io/pypi/v/crispio)

Command-line utilities and Python API for designing CRISPRi experiments in bacteria.

**crispio** makes it easy to design annotated and systematically named libraries of guide RNAs. 
Alternatively, **crispio** can map a FASTA of existing guides to a genome.

**Hint**: If you have a table of guide RNAs from the literature that you want to annotate
with genomic features, **crispio** is your tool. Use [`bioino table2fasta`](https://github.com/scbirlab/bioino#command-line) 
to convert the table to a FASTA file, then use [`crispio map`](#mapping-known-guide-rnas-to-a-genome).

- [Installation](#installation)
- [Command-line interface](#command-line-interface)
- [Generating new guide RNAs](#generating-new-guide-rnas)
- [Mapping known guide RNAs to a genome](#mapping-known-guide-rnas-to-a-genome)
- [Annotating with extra features](#annotating-with-extra-features)
- [Checking for off-targets](#checking-for-off-targets)
- [Python API](#python-api)
- [Issues, bugs, suggestions]()
- [Documentation]()

## Installation

### The easy way

Install the pre-compiled version from GitHub:

```bash
$ pip install crispio
```

### From source

Clone the repository, then `cd` into it. Then run:

```bash
$ pip install -e .
```

## Command-line interface

The main way to use **crispio** is with its several subcommands. You can get 
help by entering `crispio <subcommand> --help`.

```bash
$ crispio --help
usage: crispio [-h] {generate,map,featurize,offtarget} ...

Design and analysis of bacterial CRISPRi experiments.

optional arguments:
  -h, --help            show this help message and exit

Sub-commands:
  {generate,map,featurize,offtarget}
                        Use these commands to specify the tool you want to use.
    generate            Generate and annotate all guide RNAs for a given genome.
    map                 Map and annotate provided guide RNAs to a given genome.
    featurize           Annotate guide RNAs with additional calculated features.
    offtarget           Compare two sets of guide RNAs for potential cross-target activity.
```

## Generating new guide RNAs

Given a genome in FASTA format and a matching GFF, 
[both available for your favourite bacterium from NCBI](https://www.ncbi.nlm.nih.gov/genbank/ftp/),
along with a PAM sequence or name of a common Cas9 ortholog, you can generate all the possible 
guide RNAs and annotate them from the GFF in one go.

The command `crisio generate` finds the position on the genome, annotates
with genomic features, replichore, and sequence context, detects restriction 
sites, and gives each guide RNA a unique ID and a human-readable
adjective-noun mnemonic.

```bash
$ crispio generate -l 20 --pam Sth1 -g EcoMG1655-NC_000913.3.fasta -a EcoMG1655-NC_000913.3.gff3 | head

🚀 Generating sgRNAs with the following parameters:
        ...

##sequence-region NC_000913.3 1 4641652
##species https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=511145
##genome-sequence       NC_000913.3     1       6930
##genome-description    Escherichia coli str. K-12 substr. MG1655, complete genome
##genome-filename       63
#sgRNA-map      crispio 63      62
NC_000913.3     crispio protospacer     2       21      .       +       .       ID=sgr-e5373243;Name=thrL-21-modest_saddle;ann_Name=thrL;ann_end=255;ann_feature=gene;ann_gene=thrL;ann_gene_biotype=protein_coding;ann_locus_tag=_up-thrL;ann_phase=.;ann_score=.;ann_seqid=NC_000913.3;ann_source=RefSeq;ann_start=190;ann_strand=+;guide_context_down=TATGTCTCTGTGTGGATTAA;guide_context_up=CAGCACCCCAGGAACCCATA;guide_length=20;guide_re_sites=;guide_sequence=GCTTTTCATTCTGACTGCAA;guide_sequence_hash=19d8fdaa;mnemonic=modest_saddle;pam_end=28;pam_offset=-166;pam_replichore=R;pam_search=NNRGVAN;pam_sequence=CGGGCAA;pam_start=21;source_name=thrL-21-modest_saddle

...

NC_000913.3     crispio protospacer     180     199     .       +       .       ID=sgr-b71e7fa7;Name=thrL-199-bouncy_sabine;ann_Name=thrL;ann_end=255;ann_feature=gene;ann_gene=thrL;ann_gene_biotype=protein_coding;ann_locus_tag=b0001;ann_phase=.;ann_score=.;ann_seqid=NC_000913.3;ann_source=RefSeq;ann_start=190;ann_strand=+;guide_context_down=CACCATTACCACCACCATCA;guide_context_up=CAGATAAAAATTACAGAGTA;guide_length=20;guide_re_sites=;guide_sequence=CACAACATCCATGAAACGCA;guide_sequence_hash=3e6eb3a0;mnemonic=bouncy_sabine;pam_end=206;pam_offset=65;pam_replichore=R;pam_search=NNRGVAN;pam_sequence=TTAGCAC;pam_start=199;source_name=thrL-199-bouncy_sabine

```

The guides are output in GFF format, so it can be used directly as an annotation
track in your favourite genome browser. 

It can be dense for human beings to parse, so you can convert to a TSV using 
[`bioino ggf2table`](https://github.com/scbirlab/bioino#command-line):

```bash
$ head guides.gff | bioino gff2table
...

seqid   source  feature start   end     score   strand  phase   ID      Name    ann_Name        ann_end ann_feature     ann_gene     ann_gene_biotype ann_locus_tag   ann_phase       ann_score       ann_seqid       ann_source      ann_start       ann_strand      guide_context_down    guide_context_up        guide_length    guide_re_sites  guide_sequence  guide_sequence_hash     mnemonic        pam_end       pam_offset      pam_replichore  pam_search      pam_sequence    pam_start       source_name
NC_000913.3     crispio protospacer     2       21      .       +       .       sgr-e5373243    thrL-21-modest_saddle   thrL    255  gene     thrL    protein_coding  _up-thrL        .       .       NC_000913.3     RefSeq  190     +       TATGTCTCTGTGTGGATTAA    CAGCACCCCAGGAACCCATA  20              GCTTTTCATTCTGACTGCAA    19d8fdaa        modest_saddle   28      -166    R       NNRGVAN CGGGCAA 21   thrL-21-modest_saddle

...

NC_000913.3     crispio protospacer     180     199     .       +       .       sgr-b71e7fa7    thrL-199-bouncy_sabine  thrL    255  gene     thrL    protein_coding  b0001   .       .       NC_000913.3     RefSeq  190     +       CACCATTACCACCACCATCA    CAGATAAAAATTACAGAGTA  20              CACAACATCCATGAAACGCA    3e6eb3a0        bouncy_sabine   206     65      R       NNRGVAN TTAGCAC 199     thrL-199-bouncy_sabine
```

All the commands are pipeable. All the chatter goes to `stderr`, so
you can pipe your actual data through `stdout`.

## Mapping known guide RNAs to a genome

The command `crispio map` is similar, but takes known guides in FASTA format as input. 

```bash
$ crispio map cv-nar-2020_TableS1.fasta -g EcoMG1655-NC_000913.3.fasta -a EcoMG1655-NC_000913.3.gff3 --pam Spy

🚀 Mapping sgRNAs with the following parameters:
       ...
Finding sgRNA sites matching 21417 sequences from cv-nar-2020_TableS1.fasta and matching PAM Spy (NGGN) in EcoMG1655-NC_000913.3.fasta...

##sequence-region NC_000913.3 1 4641652
##species https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=511145
##genome-sequence       NC_000913.3     1       4641652
##genome-description    Escherichia coli str. K-12 substr. MG1655, complete genome
##genome-filename       EcoMG1655-NC_000913.3.fasta
#sgRNA-map      crispio EcoMG1655-NC_000913.3.fasta     EcoMG1655-NC_000913.3.gff3
NC_000913.3     crispio     2400946 2400965 .       +       .       ID=sgr-7a0a4f43;Name=nuoF-2400965-level_herman;ann_Name=nuoF;ann_end=2401555;ann_feature=gene;ann_gene=nuoF;ann_gene_biotype=protein_coding;ann_locus_tag=b2284;ann_phase=.;ann_score=.;ann_seqid=NC_000913.3;ann_source=RefSeq;ann_start=2400218;ann_strand=-;guide_context_down=TAGCACGACGTCCTTCCAGG;guide_context_up=CCATGCGCCGGAGGTTGCCG;guide_length=20;guide_re_sites=;guide_sequence=GGAAGGGTGGCTTCGAGCGT;guide_sequence_hash=6a359e86;mnemonic=level_herman;pam_end=2400969;pam_offset=0;pam_replichore=L;pam_search=NGGN;pam_sequence=GGGT;pam_start=2400965;source_name=GGAAGGGTGGCTTCGAGCGT

...

NC_000913.3     crispio protospacer     2400933 2400952 .       +       .       ID=sgr-32446b00;Name=nuoF-2400952-telling_austria;ann_Name=nuoF;ann_end=2401555;ann_feature=gene;ann_gene=nuoF;ann_gene_biotype=protein_coding;ann_locus_tag=b2284;ann_phase=.;ann_score=.;ann_seqid=NC_000913.3;ann_source=RefSeq;ann_start=2400218;ann_strand=-;guide_context_down=TTCGAGCGTGGGTTAGCACG;guide_context_up=AGGTCGGTTTACCCCATGCG;guide_length=20;guide_re_sites=;guide_sequence=CCGGAGGTTGCCGGGAAGGG;guide_sequence_hash=1908728f;mnemonic=telling_austria;pam_end=2400956;pam_offset=0;pam_replichore=L;pam_search=NGGN;pam_sequence=TGGC;pam_start=2400952;source_name=CCGGAGGTTGCCGGGAAGGG
```

If you don't have a FASTA to hand, but a table instead, you can pipe it
through [`bioino table2fasta`](https://github.com/scbirlab/bioino#command-line):

```bash
$ cat guide-table.tsv | bioino table2fasta -s sequence -n guide_name | crispio map -g EcoMG1655-NC_000913.3.fasta -a EcoMG1655-NC_000913.3.gff3 --pam Sth1 

##sequence-region NC_000913.3 1 4641652
##species https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=511145
##genome-sequence       NC_000913.3     1       4641652
##genome-description    Escherichia coli str. K-12 substr. MG1655, complete genome
##genome-filename       EcoMG1655-NC_000913.3.fasta
#sgRNA-map      crispio EcoMG1655-NC_000913.3.fasta     EcoMG1655-NC_000913.3.gff3
NC_000913.3     crispio protospacer     1236    1255    .       +       .       ID=sgr-831073da;Name=thrA-1255-honest_brother;ann_Name=thrA;ann_end=2799;ann_feature=gene;ann_gene=thrA;ann_gene_biotype=protein_coding;ann_locus_tag=b0002;ann_phase=.;ann_score=.;ann_seqid=NC_000913.3;ann_source=RefSeq;ann_start=337;ann_strand=+;guide_context_down=TTCCAATCTGAATAACATGG;guide_context_up=CTCATTGGTGCCAGCCGTGA;guide_length=20;guide_re_sites=BbsI;guide_sequence=TGAAGACGAATTACCGGTCA;guide_sequence_hash=55934652;mnemonic=honest_brother;pam_end=1262;pam_offset=2462;pam_replichore=R;pam_search=NNRGVAN;pam_sequence=AGGGCAT;pam_start=1255;source_name=thrA-1255-honest_brother

...

NC_000913.3     crispio protospacer     3999    4018    .       +       .       ID=sgr-83d65199;Name=thrC-4018-jolly_lunar;ann_Name=thrC;ann_end=5020;ann_feature=gene;ann_gene=thrC;ann_gene_biotype=protein_coding;ann_locus_tag=b0004;ann_phase=.;ann_score=.;ann_seqid=NC_000913.3;ann_source=RefSeq;ann_start=3734;ann_strand=+;guide_context_down=TGTTCCACGGGCCAACGCTG;guide_context_up=CCCGGCTCCGGTCGCCAATG;guide_length=20;guide_re_sites=BtgZI;guide_sequence=TTGAAAGCGATGTCGGTTGT;guide_sequence_hash=e539f903;mnemonic=jolly_lunar;pam_end=4025;pam_offset=1286;pam_replichore=R;pam_search=NNRGVAN;pam_sequence=CTGGAAT;pam_start=4018;source_name=thrC-4018-jolly_lunar
```

## Annotating with extra features

It may be useful to calcuate additional guide RNA features for downstream applications like
machine learning. These are the available extra features:

```python
>>> from crispio import *
>>> get_features()
['on_nontemplate_strand', 'context_up2', 'context_down2', 'context_up_autocorr', 'pam_n', 'pam_def', 'pam_gc', 'pam_autocorr', 'pam_scaff_corr', 'guide_purine', 'guide_gc', 'seed_seq', 'guide_start3', 'guide_end3', 'guide_autocorr', 'guide_scaff_corr']
```

Using the command-line these can be easily added to a GFF or piped output 
from `crispio map` or `crispio generate`.

```bash
$ cat mapped-guides.gff | head | crispio featurize --scaffold Spy

🚀 Featurizing sgRNAs with the following parameters:
 ...
> Generating features for guides...
##sequence-region NC_000913.3 1 4641652
##species https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=511145
##genome-sequence       NC_000913.3     1       4641652
##genome-description    Escherichia coli str. K-12 substr. MG1655, complete genome
##genome-filename       EcoMG1655-NC_000913.3.fasta
#sgRNA-map      crispin EcoMG1655-NC_000913.3.fasta     EcoMG1655-NC_000913.3.gff3
NC_000913.3     crispin protospacer     2400946 2400965 .       +       .       ID=sgr-7a0a4f43;Name=nuoF-2400965-level_herman;ann_Name=nuoF;ann_end=2401555;ann_feature=gene;ann_gene=nuoF;ann_gene_biotype=protein_coding;ann_locus_tag=b2284;ann_phase=.;ann_score=.;ann_seqid=NC_000913.3;ann_source=RefSeq;ann_start=2400218;ann_strand=-;guide_context_down=TAGCACGACGTCCTTCCAGG;guide_context_up=CCATGCGCCGGAGGTTGCCG;guide_length=20;guide_re_sites=;guide_sequence=GGAAGGGTGGCTTCGAGCGT;guide_sequence_hash=6a359e86;mnemonic=level_herman;pam_end=2400969;pam_offset=0;pam_replichore=L;pam_search=NGGN;pam_sequence=GGGT;pam_start=2400965;source_name=GGAAGGGTGGCTTCGAGCGT;feat_on_nontemplate_strand=True;feat_context_up2=CG;feat_context_down2=TA;feat_context_up_autocorr=8.928;feat_pam_n=G;feat_pam_def=GGT;feat_pam_gc=0.750;feat_pam_autocorr=2.167;feat_pam_scaff_corr=1.917;feat_guide_purine=0.650;feat_guide_gc=0.650;feat_seed_seq=AGCGT;feat_guide_start3=GGA;feat_guide_end3=CGT;feat_guide_autocorr=8.704;feat_guide_scaff_corr=9.772

...

NC_000913.3     crispin protospacer     1764112 1764131 .       +       .       ID=sgr-f3815635;Name=sufA-1764131-scarce_game;ann_Name=sufA;ann_end=1764386;ann_feature=gene;ann_gene=sufA;ann_gene_biotype=protein_coding;ann_locus_tag=b1684;ann_phase=.;ann_score=.;ann_seqid=NC_000913.3;ann_source=RefSeq;ann_start=1764018;ann_strand=-;guide_context_down=ATCGCTTGCAGCGGGACAAA;guide_context_up=GTCCTTCACGAACGAAATCG;guide_length=20;guide_re_sites=;guide_sequence=ACTTCCGTGCCATCAATAAA;guide_sequence_hash=dc4688e0;mnemonic=scarce_game;pam_end=1764135;pam_offset=0;pam_replichore=R;pam_search=NGGN;pam_sequence=CGGC;pam_start=1764131;source_name=ACTTCCGTGCCATCAATAAA;feat_on_nontemplate_strand=True;feat_context_up2=CG;feat_context_down2=AT;feat_context_up_autocorr=7.159;feat_pam_n=C;feat_pam_def=GGC;feat_pam_gc=1.000;feat_pam_autocorr=2.333;feat_pam_scaff_corr=1.667;feat_guide_purine=0.450;feat_guide_gc=0.400;feat_seed_seq=ATAAA;feat_guide_start3=ACT;feat_guide_end3=AAA;feat_guide_autocorr=7.767;feat_guide_scaff_corr=10.528
```

The attributes starting with `feat_` have been added.

## Checking for off-targets

One downside of CRISPR-based tools is the possibility of off-target effects.
`crispio offtarget` compares two GFF files, or one GFF file against itself, for 
guide RNA sites that share a seed sequence (4 PAM-proximal bases) and have a 
Hamming distance of 4 or less.

```bash
$ cat mapped-guides.gff | head | crispio offtarget -2 <(cat mapped-guides.gff)

##sequence-region NC_000913.3 1 4641652
##species https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=511145
##genome-sequence       NC_000913.3     1       4641652
##genome-description    Escherichia coli str. K-12 substr. MG1655, complete genome
##genome-filename       EcoMG1655-NC_000913.3.fasta
#sgRNA-map      crispin EcoMG1655-NC_000913.3.fasta     EcoMG1655-NC_000913.3.gff3
#crosstalk-comparator   63
NC_000913.3     crispin protospacer     2400946 2400965 .       +       .       ID=sgr-7a0a4f43;Name=nuoF-2400965-level_herman;ann_Name=nuoF;ann_end=2401555;ann_feature=gene;ann_gene=nuoF;ann_gene_biotype=protein_coding;ann_locus_tag=b2284;ann_phase=.;ann_score=.;ann_seqid=NC_000913.3;ann_source=RefSeq;ann_start=2400218;ann_strand=-;guide_context_down=TAGCACGACGTCCTTCCAGG;guide_context_up=CCATGCGCCGGAGGTTGCCG;guide_length=20;guide_re_sites=;guide_sequence=GGAAGGGTGGCTTCGAGCGT;guide_sequence_hash=6a359e86;mnemonic=level_herman;pam_end=2400969;pam_offset=0;pam_replichore=L;pam_search=NGGN;pam_sequence=GGGT;pam_start=2400965;source_name=GGAAGGGTGGCTTCGAGCGT

...

NC_000913.3     crispin protospacer     1764112 1764131 .       +       .       ID=sgr-f3815635;Name=sufA-1764131-scarce_game;ann_Name=sufA;ann_end=1764386;ann_feature=gene;ann_gene=sufA;ann_gene_biotype=protein_coding;ann_locus_tag=b1684;ann_phase=.;ann_score=.;ann_seqid=NC_000913.3;ann_source=RefSeq;ann_start=1764018;ann_strand=-;guide_context_down=ATCGCTTGCAGCGGGACAAA;guide_context_up=GTCCTTCACGAACGAAATCG;guide_length=20;guide_re_sites=;guide_sequence=ACTTCCGTGCCATCAATAAA;guide_sequence_hash=dc4688e0;mnemonic=scarce_game;pam_end=1764135;pam_offset=0;pam_replichore=R;pam_search=NGGN;pam_sequence=CGGC;pam_start=1764131;source_name=ACTTCCGTGCCATCAATAAA
```

## Python API

Some classes and functions are exposed in an API for generating guide RNAs 
in Python scripts.

Guides can be generated *de novo*.

```python 
from crispio import GuideLibrary

genome = "ATATATATATATATATATATATATACCGTTTTTTTAAAAAAACGGATATATATATATAATATATATATATAATATATATATATA"
gl = GuideLibrary.from_generating(genome=genome) 
for match_collection in gl:
    for guide in match_collection:
            print(guide)
```

The above code would return:

```
ATACCGTTTTTTTAAAAAAA
ATACCGTTTTTTTAAAAAAA
```

Or known guide sequences can be mapped to a genome. 

```python
from crispio import GuideLibrary

genome = "CCCCCCCCCCCTTTTTTTTTTAAAAAAAAAATGATCGATCGATCGAGGAAAAAAAAAACCCCCCCCCCC"
guide_seq = "ATGATCGATCGATCG"
gl = GuideLibrary.from_mapping(guide_seq=guide_seq, genome=genome) 

for collection in gl:
    for match in collection:
            print(match.as_dict())
```

This code would return:

```
{'pam_search': 'NGG', 'guide_seq': 'ATGATCGATCGATCG', 'pam_seq': 'AGG', 'pam_start': 45, 'reverse': False, 'guide_context_up': 'CTTTTTTTTTTAAAAAAAAA', 'guide_context_down': 'AAAAAAAAAACCCCCCCCCC', 'pam_end': 48, 'length': 15, 'guide_start': 30, 'guide_end': 45}
```

Check the full API in the [documentation](https://readthedocs.org/crispio).

## Issues, bugs, suggestions

Do not hesitate to add to our [issue tracker](https://github.com/scbirlab/crispio).

## Documentation

Check the documentation and full API [here](https://readthedocs.org/crispio).