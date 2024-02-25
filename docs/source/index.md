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

```{toctree}
:maxdepth: 2
:caption: Contents:

installation
usage
modules
```

## Issues, problems, suggestions

Add to the [issue tracker](https://www.github.com/crispio/issues).

## Source

View source at [GitHub](https://github.com/scbirlab/crispio).
