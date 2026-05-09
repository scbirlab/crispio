#!/usr/bin/env bash

set -euox pipefail
mkdir -p test/outputs

crispio generate \
    --genome test/inputs/EcoMG1655-NC_000913.3.fasta \
    --annotations test/inputs/EcoMG1655-NC_000913.3.gff3 \
    --pam Spy \
| head -n1000 > test/outputs/NC_000913.3.gff

crispio map test/inputs/cv-nar-2020_TableS1.fasta \
    --genome test/inputs/EcoMG1655-NC_000913.3.fasta \
    --annotations test/inputs/EcoMG1655-NC_000913.3.gff3 \
    --pam Spy \
| head -n1000 > test/outputs/cv-nar-2020_TableS1.gff

crispio map test/inputs/hawkins-2020_TableS6.fasta \
    --genome test/inputs/EcoMG1655-NC_000913.3.fasta \
    --annotations test/inputs/EcoMG1655-NC_000913.3.gff3 \
    --pam Spy \
| head -n100 > test/outputs/hawkins-2020_TableS6.gff
