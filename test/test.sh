#!/usr/bin/env bash

function check_not_empty () (
    local f="$1"
    n_lines=$(grep -v -c '^#' "$f")
    if [ "$n_lines" -lt 2 ]
    then
        echo "No guides mapped: "$f" has $n_lines lines"
        exit 1
    fi
)

set -euox pipefail
mkdir -p test/outputs

crispio generate \
    --genome test/inputs/EcoMG1655-NC_000913.3.fasta \
    --annotations test/inputs/EcoMG1655-NC_000913.3.gff3 \
    --pam Spy \
    --limit 1000 \
    --output test/outputs/NC_000913.3.gff
check_not_empty test/outputs/NC_000913.3.gff

crispio map test/inputs/cv-nar-2020_TableS1.fasta \
    --genome test/inputs/EcoMG1655-NC_000913.3.fasta \
    --annotations test/inputs/EcoMG1655-NC_000913.3.gff3 \
    --pam Spy \
    --limit 1000 \
    --output test/outputs/cv-nar-2020_TableS1_1000.gff
check_not_empty test/outputs/cv-nar-2020_TableS1_1000.gff

crispio map <(head -n1000 test/inputs/cv-nar-2020_TableS1.fasta) \
    --genome test/inputs/EcoMG1655-NC_000913.3.fasta \
    --annotations test/inputs/EcoMG1655-NC_000913.3.gff3 \
    --pam Spy \
    --output test/outputs/cv-nar-2020_TableS1.gff
check_not_empty test/outputs/cv-nar-2020_TableS1.gff

crispio map test/inputs/hawkins-2020_TableS6.fasta \
    --genome test/inputs/EcoMG1655-NC_000913.3.fasta \
    --annotations test/inputs/EcoMG1655-NC_000913.3.gff3 \
    --pam Spy \
    --limit 1000 \
    --output test/outputs/hawkins-2020_TableS6.gff
check_not_empty test/outputs/hawkins-2020_TableS6.gff
