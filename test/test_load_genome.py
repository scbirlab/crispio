import io
import pytest
from bioino import GffFile
from crispio.cli import _load_genome_and_gff


class NamedStringIO(io.StringIO):
    """StringIO with a .name attribute, matching the interface of a real file handle."""
    def __init__(self, content, name):
        super().__init__(content)
        self.name = name


FASTA_SINGLE = (
    '>chr1 E. coli chromosome\n'
    'ATGCATGCATGCATGCATGC\n'
)
FASTA_MULTI = (
    '>chr1 main chromosome\n'
    'ATGCATGCATGCATGCATGC\n'
    '>plasmid1 resistance plasmid\n'
    'GCATGCATGCAT\n'
)
GFF_CHR1 = '\t'.join([
    'chr1', 'RefSeq', 'gene', '1', '20', '.', '+', '.',
    'ID=g1;Name=geneA;locus_tag=b0001',
]) + '\n'


@pytest.fixture
def single_chrom():
    seqs, gff = _load_genome_and_gff(
        NamedStringIO(FASTA_SINGLE, 'genome.fasta'),
        NamedStringIO(GFF_CHR1,    'genome.gff'),
    )
    return seqs, gff


@pytest.fixture
def multi_chrom():
    seqs, gff = _load_genome_and_gff(
        NamedStringIO(FASTA_MULTI, 'genome.fasta'),
        NamedStringIO(GFF_CHR1,   'genome.gff'),
    )
    return seqs, gff


class TestLoadGenomeAndGffSingle:

    def test_returns_one_sequence(self, single_chrom):
        seqs, _ = single_chrom
        assert len(seqs) == 1

    def test_sequence_name(self, single_chrom):
        seqs, _ = single_chrom
        assert seqs[0].name == 'chr1'

    def test_sequence_length(self, single_chrom):
        seqs, _ = single_chrom
        assert len(seqs[0].sequence) == 20

    def test_lookup_built(self, single_chrom):
        _, gff = single_chrom
        assert 'chr1' in gff._lookup

    def test_lookup_hit(self, single_chrom):
        _, gff = single_chrom
        result = gff.lookup_at('chr1', 10)
        assert result[0].attributes['Name'] == 'geneA'

    def test_metadata_has_genome_sequence(self, single_chrom):
        _, gff = single_chrom
        entries = [m for m in gff.metadata.data if m.name == 'genome-sequence']
        assert len(entries) == 1
        assert str(entries[0].values[0]) == 'chr1'

    def test_metadata_has_genome_description(self, single_chrom):
        _, gff = single_chrom
        entries = [m for m in gff.metadata.data if m.name == 'genome-description']
        assert len(entries) == 1

    def test_metadata_has_genome_filename(self, single_chrom):
        _, gff = single_chrom
        entries = [m for m in gff.metadata.data if m.name == 'genome-filename']
        assert str(entries[0].values[0]) == 'genome.fasta'


class TestLoadGenomeAndGffMulti:

    def test_returns_all_sequences(self, multi_chrom):
        seqs, _ = multi_chrom
        assert len(seqs) == 2

    def test_sequence_names(self, multi_chrom):
        seqs, _ = multi_chrom
        assert [s.name for s in seqs] == ['chr1', 'plasmid1']

    def test_sequence_lengths(self, multi_chrom):
        seqs, _ = multi_chrom
        assert [(s.name, len(s.sequence)) for s in seqs] == [('chr1', 20), ('plasmid1', 12)]

    def test_annotated_chrom_lookup_hit(self, multi_chrom):
        _, gff = multi_chrom
        result = gff.lookup_at('chr1', 10)
        assert result[0].attributes['Name'] == 'geneA'

    def test_unannotated_contig_lookup_miss(self, multi_chrom):
        _, gff = multi_chrom
        assert gff.lookup_at('plasmid1', 5) == ()

    def test_metadata_one_genome_sequence_per_chrom(self, multi_chrom):
        _, gff = multi_chrom
        entries = [m for m in gff.metadata.data if m.name == 'genome-sequence']
        assert len(entries) == 2
        assert [str(m.values[0]) for m in entries] == ['chr1', 'plasmid1']

    def test_metadata_no_genome_description_for_multi(self, multi_chrom):
        # genome-description is ambiguous for multi-sequence files and should be omitted
        _, gff = multi_chrom
        entries = [m for m in gff.metadata.data if m.name == 'genome-description']
        assert len(entries) == 0


class TestLoadGenomeAndGffEdgeCases:

    def test_empty_fasta_raises(self):
        with pytest.raises(OSError, match='No sequences found'):
            _load_genome_and_gff(
                NamedStringIO('', 'empty.fasta'),
                NamedStringIO('', 'empty.gff'),
            )
