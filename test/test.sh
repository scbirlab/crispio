#!/usr/bin/env bash

set -euox pipefail
mkdir -p test/outputs

crispio generate \
    --genome test/inputs/EcoMG1655-NC_000913.3.fasta \
    --annotations test/inputs/EcoMG1655-NC_000913.3.gff3 \
    --pam Spy \
    --limit 1000 \
    --output test/outputs/NC_000913.3.gff

n_lines=\$(grep -v -c '^#' test/outputs/NC_000913.3.gff)
if [ "\$n_lines" -lt 2 ]
then
    echo "No guides mapped: GFF has \$n_lines lines"
    exit 1
fi

crispio map test/inputs/cv-nar-2020_TableS1.fasta \
    --genome test/inputs/EcoMG1655-NC_000913.3.fasta \
    --annotations test/inputs/EcoMG1655-NC_000913.3.gff3 \
    --pam Spy \
    --limit 1000 \
    --output test/outputs/cv-nar-2020_TableS1.gff

n_lines=\$(grep -v -c '^#' test/outputs/cv-nar-2020_TableS1.gff)
if [ "\$n_lines" -lt 2 ]
then
    echo "No guides mapped: GFF has \$n_lines lines"
    exit 1
fi

crispio map test/inputs/hawkins-2020_TableS6.fasta \
    --genome test/inputs/EcoMG1655-NC_000913.3.fasta \
    --annotations test/inputs/EcoMG1655-NC_000913.3.gff3 \
    --pam Spy \
    --limit 1000 \
    --output test/outputs/hawkins-2020_TableS6.gff

n_lines=\$(grep -v -c '^#' test/outputs/hawkins-2020_TableS6.gff)
if [ "\$n_lines" -lt 2 ]
then
    echo "No guides mapped: GFF has \$n_lines lines"
    exit 1
fi
