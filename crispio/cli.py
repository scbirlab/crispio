"""Command-line interface for crispio."""

from typing import Callable, Dict, Tuple, Union

from argparse import Namespace, FileType
from dataclasses import replace
from functools import wraps
from io import TextIOWrapper
import os
import sys

from bioino import FastaCollection, FastaSequence, GffFile
from carabiner import print_err
from carabiner.cliutils import CLIApp, CLICommand, CLIOption, clicommand
from tqdm.auto import tqdm
from streq import Circular

from . import __version__
from .crosstalk import _get_mismatches
from .features import featurize
from .map import GuideLibrary
from .utils import sequences

def _allow_broken_pipe(f: Callable) -> Callable:

    @wraps(f)
    def _f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except BrokenPipeError:
            sys.exit(0)

    return _f


def _load_genome_and_gff(
    genome: TextIOWrapper, 
    gff: TextIOWrapper
) -> Tuple[FastaSequence, GffFile]:

    for fasta_sequence in FastaCollection.from_file(genome).sequences:
        break

    fasta_sequence = replace(
        fasta_sequence,
        sequence=Circular(fasta_sequence.sequence),
    )

    genome_filename = os.path.basename(genome.name)
    gff_filename = os.path.basename(gff.name)

    gff_data = GffFile.from_file(gff)

    new_metadata = [
        ('genome-sequence', 'constrained', [fasta_sequence.name, 1, len(fasta_sequence.sequence)]), 
        ('genome-description', 'constrained', [fasta_sequence.description]),
        ('genome-filename', 'constrained', [genome_filename]),
        ('sgRNA-map', 'free', [__package__, genome_filename, gff_filename]),
    ]
    old_metadata = list(gff_data.metadata.data)

    gff_data = replace(
        gff_data, 
        lookup=True,
        metadata=old_metadata + new_metadata,
    )

    return fasta_sequence, gff_data


def _prepare_to_search(args: Namespace) -> Tuple[FastaSequence, GffFile, str, Dict[str, Union[str, int]]]:

    fasta_sequence, gff_data = _load_genome_and_gff(
        args.genome,
        args.annotations,
    )

    try:
        pam_search = sequences.pams[args.pam]
    except KeyError:
        pam_search = args.pam
    
    sgRNA_defaults = {
        "seqid": fasta_sequence.name,
        "source": __package__,
        "feature": 'protospacer',
        "score": '.',
        "phase": '.',
    }
    
    return fasta_sequence, gff_data, pam_search, sgRNA_defaults


@clicommand(message="Mapping sgRNAs with the following parameters")
def _map(args: Namespace) -> None:

    fasta_sequence, gff_data, pam_search, sgRNA_defaults = _prepare_to_search(args)

    guide_sequences = list(FastaCollection.from_file(args.input).sequences)
    n_guide_sequences = len(guide_sequences)

    print_err(
        f'\nFinding sgRNA sites matching {n_guide_sequences} sequences',
        f'from {args.input.name} and matching',
        f'PAM {args.pam} ({pam_search})',
        f'in {args.genome.name}...',
    )

    gff_data.metadata.write(file=args.output)
    guide_library = GuideLibrary.from_mapping(
        guide_seq=guide_sequences,
        genome=fasta_sequence.sequence,
        pam_search=pam_search,
    )

    for guide_match in guide_library.as_gff(
        max=1,
        annotations_from=gff_data,
        tags=args.attributes,
        gff_defaults=sgRNA_defaults,
    ):        
        _allow_broken_pipe(guide_match.write)(file=args.output)

    return None


@clicommand(message="Generating sgRNAs with the following parameters")
def _generate(args: Namespace) -> None:
    
    fasta_sequence, gff_data, pam_search, sgRNA_defaults = _prepare_to_search(args)

    gff_data.metadata.write(file=args.output)
    guide_library = GuideLibrary.from_generating(
        max_length=args.max_length,
        min_length=args.min_length,
        genome=fasta_sequence.sequence,
        pam_search=pam_search,
    )
    
    for guide_match in guide_library.as_gff(
        max=1,
        annotations_from=gff_data,
        tags=args.attributes,
        gff_defaults=sgRNA_defaults,
    ):        
        _allow_broken_pipe(guide_match.write)(file=args.output)
    
    return None


@clicommand(message="Featurizing sgRNAs with the following parameters")
def _featurize(args: Namespace) -> None:
    
    try:
        scaffold = sequences.scaffolds[args.scaffold]
    except KeyError:
        scaffold = args.scaffold

    print_err('> Generating features for guides...')

    input_gff = GffFile.from_file(args.input)
    input_gff.metadata.write(file=args.output)

    with tqdm(list(input_gff.lines)) as t:  ## run a progress bar
        for gff_line in t:
            t.set_postfix(current=gff_line.attributes["Name"][:40])
            new_features = featurize(
                gff_line, 
                scaffold=scaffold,
            )
            attr = gff_line.attributes.copy()
            attr.update(new_features)
            gff_line = replace(gff_line, attributes=attr)
            _allow_broken_pipe(gff_line.write)(file=args.output)
    
    return None


@clicommand(message="Detecting off-targets with the following parameters")
def _offtarget(args: Namespace) -> None:

    gff1 = GffFile.from_file(args.input)
    gff2 = GffFile.from_file(args.gff2)

    gff2_lines = tuple(gff2.lines)
    
    n_mismatches = 0 
    pairs_checked = set()

    new_metadata = [
        ('crosstalk-comparator', 'free', [os.path.basename(args.gff2.name)]),
    ]
    old_metadata = list(gff1.metadata.data)
    new_metadata = replace(
        gff1.metadata, 
        data=old_metadata + new_metadata,
    )
    new_metadata.write(file=args.output)

    with tqdm(gff1.lines) as t:
        for gff_line1 in t:
            mm = {}
            for gff_line2 in gff2_lines:
                t.set_postfix(mismatches=n_mismatches)
                pair, added_mm = _get_mismatches(
                    gff_line1, 
                    gff_line2, 
                    maximum=args.mismatches,
                    pairs_checked=pairs_checked,
                )
                pairs_checked.add(pair)
                mm.update(added_mm)
                n_mismatches = len(mm)

            attr = gff_line1.attributes.copy()
            if len(mm) > 0:
                attr['crosstalk'] = ('+'.join(f'{key}~{val}'for key, val in mm.items()))
            gff_line1 = replace(gff_line1, attributes=attr)
            _allow_broken_pipe(gff_line1.write)(args.output)
            
    return None


def main():

    length_max = CLIOption(
        '--max_length', '-l', 
        type=int, 
        default=20,
        help='Maximum length.',
    )
    length_min = CLIOption(
        '--min_length', '-m', 
        type=int, 
        default=None,
        help='Minimum length.',
    )

    # featurize
    scaffold = CLIOption(
        '--scaffold', '-s', 
        dest='scaffold', 
        type=str, 
        required=True,
        help=(
            'Name of a scaffold from "{}" or a scaffold sequence.'
            ).format('", "'.join(sequences.scaffolds)),
    )
    
    # offtarget
    gff2 = CLIOption(
        '--gff2', '-2', 
        type=FileType('r'),
        required=True,
        help='GFF file containing protospacers for comparison.',
    )
    mismatches = CLIOption(
        '--mismatches', '-e', 
        type=int, 
        default=2,
        help='Maximum number of edits between guides to mark a close match.',
    )

    inputs = CLIOption(
        'input', 
        type=FileType('r'),
        default=sys.stdin,
        nargs='?',
        help='Input file. Default: STDIN.',
    )

    pam = CLIOption(
        '--pam', '-p',  
        type=str, 
        required=True,
        help=('Either a PAM sequence or one of "{}".').format('", "'.join(sequences.pams)),
    )
    genome = CLIOption(
        '--genome', '-g', 
        type=FileType('r'),
        required=True,
        help='Genome FASTA file',
    )
    annotations = CLIOption(
        '--annotations', '-a', 
        type=FileType('r'),
        required=True,
        help='GFF file.',
    )
    attributes = CLIOption(
        '--attributes', '-t', 
        type=str, 
        nargs='*',
        default=['Name', 'locus_tag', 'old_locus_tag', 'gene', 'gene_biotype'],
        help='Tag to use in attribute field (column 9) of GFF file.',
    )
    outputs = CLIOption(
        '--output', '-o', 
        type=FileType('w'),
        default=sys.stdout,
        help='Output file. Default: STDOUT',
    )

    generate = CLICommand(
        "generate", 
        description="Generate and annotate all guide RNAs for a given genome.",
        main=_generate,
        options=[
            length_max, 
            length_min, 
            pam, 
            genome, 
            annotations, 
            attributes, 
            outputs,
        ],
    )
    map_guides = CLICommand(
        "map", 
        description="Map and annotate provided guide RNAs to a given genome.",
        main=_map,
        options=[
            inputs, 
            pam, 
            genome, 
            annotations, 
            attributes, 
            outputs,
        ],
    )
    featurize = CLICommand(
        "featurize", 
        description="Annotate guide RNAs with additional calculated features.",
        main=_featurize,
        options=[
            inputs, 
            scaffold, 
            outputs,
        ],
    )
    offtarget = CLICommand(
        "offtarget", 
        description="Compare two sets of guide RNAs for potential cross-target activity.",
        main=_offtarget,
        options=[
            inputs, 
            gff2, 
            mismatches, 
            outputs,
        ],
    )
    
    app = CLIApp(
        'crispio', 
        description='Design and analysis of bacterial CRISPRi experiments.',
        version=__version__,
        commands=[
            generate, 
            map_guides, 
            featurize, 
            offtarget,
        ],
    )

    app.run()

    return None


if __name__ == '__main__':

    main()