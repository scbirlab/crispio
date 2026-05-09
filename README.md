# 🌱 crispio

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/scbirlab/crispio/python-publish.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/crispio)
![PyPI](https://img.shields.io/pypi/v/crispio)

**Design and annotate bacterial CRISPRi guide RNA libraries from any genome.**

CRISPRi uses a catalytically dead Cas9 to silence genes by blocking transcription. Designing a good library means knowing not just *where* a guide targets, but *how far upstream of the TSS* it lands, which replichore it sits on, whether it shares a seed sequence with another guide, and whether it contains a restriction site that would break your cloning. crispio computes all of this in one pass and outputs annotated GFF3 that loads directly into any genome browser.

```
crispio generate --pam Spy -g genome.fasta -a genome.gff3 > guides.gff
```

- [Quick start](#quick-start)
- [What you get](#what-you-get)
- [Generating a new library](#generating-a-new-library)
- [Annotating guides from the literature](#annotating-guides-from-the-literature)
- [Checking for off-targets](#checking-for-off-targets)
- [Adding ML features](#adding-ml-features)
- [Piping commands together](#piping-commands-together)
- [Python API](#python-api)
- [Installation](#installation)
- [PAMs and scaffolds](#pams-and-scaffolds)
- [Issues and documentation](#issues-and-documentation)

---

## Quick start

You need two files, both available for any sequenced bacterium from [NCBI](https://www.ncbi.nlm.nih.gov/genbank/ftp/):

- **FASTA** — the genome sequence (`.fasta` / `.fa`)
- **GFF3** — gene annotations (`.gff` / `.gff3`)

Try crispio on the first 100 guides straight away with `--limit`:

```bash
crispio generate \
  --pam Spy \
  --genome EcoMG1655-NC_000913.3.fasta \
  --annotations EcoMG1655-NC_000913.3.gff3 \
  --limit 100 \
  > first100.gff
```

Convert to a spreadsheet-friendly table with [`bioino`](https://github.com/scbirlab/bioino):

```bash
cat first100.gff | bioino gff2table > first100.tsv
```

Open `first100.tsv` in Excel. Each row is one guide. The most useful columns at a glance:

| Column | Example | What it means |
|--------|---------|---------------|
| `Name` | `thrL-21-modest_saddle` | `gene-position-mnemonic` |
| `guide_sequence` | `GCTTTTCATTCTGACTGCAA` | The 20 nt spacer to synthesise |
| `pam_offset` | `-166` | Distance from PAM to gene start. **Negative = upstream of TSS** — the productive targeting window for CRISPRi |
| `pam_replichore` | `R` | Left or right replichore — matters for efficiency in fast-growing bacteria |
| `ann_locus_tag` | `b0001` | Systematic gene ID for programmatic filtering |
| `guide_re_sites` | `BbsI` | Restriction sites in the spacer that would break Golden Gate cloning |

---

## What you get

Every guide gets a **stable, human-readable mnemonic** — `modest_saddle`, `bouncy_sabine` — that is a deterministic hash of the guide sequence, PAM, and position. The same guide always gets the same mnemonic regardless of when you run crispio or what else is in the library. Use it to refer to guides in lab notebooks and across collaborators without copying 20-character sequences.

The `pam_offset` is signed: **negative means the PAM is upstream of the annotated gene start**, which is the productive targeting window for bacterial CRISPRi. Positive values target inside the coding sequence. Filter on it directly:

```bash
cat guides.gff | bioino gff2table \
  | awk -F'\t' 'NR==1 || ($NF+0 < 0 && $NF+0 > -300)' \
  > upstream_guides.tsv
```

Output is standard **GFF3** and loads as an annotation track in [IGV](https://igv.org) and [Artemis](https://www.sanger.ac.uk/tool/artemis/) — useful for visually checking guide distribution across the chromosome before ordering.

---

## Generating a new library

`crispio generate` finds every PAM site in the genome, extracts the adjacent spacer, and annotates everything in one pass.

```bash
crispio generate \
  --pam Sth1 \
  --max_length 20 \
  --genome EcoMG1655-NC_000913.3.fasta \
  --annotations EcoMG1655-NC_000913.3.gff3 \
  --output guides.gff
```

For multi-chromosome genomes (chromosome + plasmids), pass a FASTA with multiple sequences. Each sequence is processed independently and guides are tagged with the correct chromosome identifier.

Use `--limit N` for quick exploratory runs or to generate a capped sub-library:

```bash
crispio generate --pam Spy -g genome.fasta -a genome.gff3 --limit 500
```

---

## Annotating guides from the literature

This is one of the most useful things crispio does: take a published guide library and fully re-annotate it against your genome. It doesn't require matching coordinates or assemblies — it searches by sequence, so it works across strains.

If you have a TSV with a `sequence` column and a `guide_name` column:

```bash
cat published_library.tsv \
  | bioino table2fasta --sequence sequence --name guide_name \
  | crispio map \
      --pam Spy \
      --genome EcoMG1655-NC_000913.3.fasta \
      --annotations EcoMG1655-NC_000913.3.gff3 \
  > annotated_library.gff
```

Or from an existing FASTA of spacers:

```bash
crispio map \
  published_spacers.fasta \
  --pam Spy \
  --genome EcoMG1655-NC_000913.3.fasta \
  --annotations EcoMG1655-NC_000913.3.gff3 \
  > annotated_library.gff
```

Guides not found in the genome are reported to stderr and skipped — they never appear silently with wrong coordinates.

---

## Checking for off-targets

`crispio offtarget` flags pairs of guides that share a 4 nt PAM-proximal seed sequence and differ by ≤ 4 mismatches elsewhere. These are candidates for unintended cross-silencing.

```bash
# Check a library against itself
crispio offtarget --gff2 guides.gff < guides.gff > checked.gff
```

Flagged guides get a `crosstalk` attribute listing the IDs and distances of matches. Check two libraries against each other — for example, confirming that guides from one experiment won't interfere with another:

```bash
crispio offtarget --gff2 library_b.gff < library_a.gff > crosstalk.gff
```

---

## Adding ML features

`crispio featurize` appends sequence-based features for downstream activity prediction, prefixed `feat_` in the output.

```bash
cat guides.gff | crispio featurize --scaffold Sth1 > guides_featurized.gff
```

Available features:

```python
>>> from crispio import get_features
>>> get_features()
['on_nontemplate_strand', 'context_up2', 'context_down2', 'context_up_autocorr',
 'pam_n', 'pam_def', 'pam_gc', 'pam_autocorr', 'pam_scaff_corr',
 'guide_purine', 'guide_gc', 'seed_seq', 'guide_start3', 'guide_end3',
 'guide_autocorr', 'guide_scaff_corr']
```

`--scaffold` takes a name (`Sth1`, `PerturbSeq`) or a raw scaffold sequence. Use the scaffold for the Cas9 you are working with — the correlation-based features depend on it.

---

## Piping commands together

All subcommands read from stdin and write to stdout. Informational messages go to stderr only, so they never appear in your data stream. Full pipelines with no intermediate files:

```bash
# Generate → featurize → table
crispio generate --pam Spy -g genome.fasta -a genome.gff3 \
  | crispio featurize --scaffold Sth1 \
  | bioino gff2table \
  > full_library.tsv
```

```bash
# Map a published library → off-target check → table
cat published_spacers.fasta \
  | crispio map --pam Spy -g genome.fasta -a genome.gff3 \
  | crispio offtarget -2 <(crispio generate --pam Spy -g genome.fasta -a genome.gff3) \
  | bioino gff2table \
  > mapped_checked.tsv
```

---

## Python API

**Generate guides de novo:**

```python
from crispio import GuideLibrary

genome = "ATATATATATATATATATATATATACCGTTTTTTTAAAAAAACGGATATATATATATAATATATATATATAATATATATATATA"
gl = GuideLibrary.from_generating(genome=genome, pam_search="NGG")

for match_collection in gl:
    for guide in match_collection:
        print(guide)
# ATACCGTTTTTTTAAAAAAA
# TATCCGTTTTTTTAAAAAAA
```

**Map known sequences to a genome:**

```python
from crispio import GuideLibrary

genome = "CCCCCCCCCCCTTTTTTTTTTAAAAAAAAAATGATCGATCGATCGAGGAAAAAAAAAACCCCCCCCCCC"
gl = GuideLibrary.from_mapping(
    guide_seq=["ATGATCGATCGATCG"],
    genome=genome,
    pam_search="NGG",
)

for collection in gl:
    for match in collection:
        print(match.guide_seq, match.pam_start, match.reverse)
```

**Calculate features:**

```python
from crispio import featurize
from crispio.utils import sequences

# gff_line is a bioino.GffLine with guide_sequence, pam_sequence, etc.
scaffold_seq = sequences.scaffolds["Sth1"]
features = featurize(gff_line, scaffold=scaffold_seq)
# {"feat_guide_gc": "0.500", "feat_seed_seq": "GATCG", ...}
```

Pass the scaffold **sequence**, not the name, to `featurize`. Use `sequences.scaffolds["Sth1"]` to retrieve it.

Full API reference: [crispio.readthedocs.io](https://crispio.readthedocs.io/en/latest/)

---

## Installation

Requires Python ≥ 3.10.

```bash
pip install crispio
```

Verify:

```bash
crispio --help
```

**From source:**

```bash
git clone https://github.com/scbirlab/crispio.git
cd crispio
pip install -e .
```

---

## PAMs and scaffolds

Built-in PAM names for `--pam`:

| Name | IUPAC | Cas9 |
|------|-------|------|
| `Spy` | `NGGN` | SpCas9 (*S. pyogenes*) |
| `Sth1` | `NNRGVAN` | StCas9-1 (*S. thermophilus*) |
| `Sau` | `NGRRT` | SaCas9 (*S. aureus*) |
| `Nme` | `NNNNGAT` | NmeCas9 (*N. meningitidis*) |

Built-in scaffold names for `--scaffold`:

| Name | Description |
|------|-------------|
| `Sth1` | StCas9-1 scaffold |
| `PerturbSeq` | Perturb-seq optimised scaffold |

Any IUPAC sequence can be passed directly to either argument.

---

## Issues and documentation

- **Bugs and feature requests**: [issue tracker](https://github.com/scbirlab/crispio/issues)
- **Full API reference**: [crispio.readthedocs.io](https://crispio.readthedocs.io/en/latest/)