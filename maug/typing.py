from typing import List, Union

from maug import _itertools

Text = Union[str, List[str]]

MaskingPattern = Union[str, _itertools.ResetableIterator]
