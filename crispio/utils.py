"""Utilities for crispio package."""

from collections import namedtuple
import os

import yaml

_data_path = os.path.join(os.path.dirname(__file__), 
                          'sequences.yml')

with open(_data_path, 'r') as f:
    _sequences = yaml.safe_load(f)

_sequence_dict = dict(
    pams = _sequences['PAMs'],
    scaffolds = _sequences['scaffolds']
)

SequenceCollection = namedtuple('SequenceCollection',
                                _sequence_dict)

sequences = SequenceCollection(**_sequence_dict)