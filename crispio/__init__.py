from importlib.metadata import version

from .features import get_features, featurize
from .map import GuideMatch, GuideMatchCollection, GuideLibrary
from .utils import sequences

__version__ = version("crispio")