from __future__ import unicode_literals
from builtins import str
from past.builtins import basestring as base_str
import hashlib
import sys


if sys.version_info.major == 2:
    # noinspection PyUnresolvedReferences,PyShadowingBuiltins
    str = str

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


# `past.builtins.basestring` module can't be imported on Python3 in some environments (Ubuntu).
# This code is copy-pasted from it to avoid crashes.
class BaseBaseString(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, (bytes, str))

    def __subclasshook__(cls, thing):
        # TODO: What should go here?
        raise NotImplemented


def with_metaclass(meta, *bases):
    class metaclass(meta):
        __call__ = type.__call__
        __init__ = type.__init__

        def __new__(cls, name, this_bases, d):
            if this_bases is None:
                return type.__new__(cls, name, (), d)
            return meta(name, bases, d)

    return metaclass('temporary_class', None, {})


if sys.version_info.major >= 3:

    class basestring(with_metaclass(BaseBaseString)):
        pass

else:
    # noinspection PyUnresolvedReferences,PyCompatibility
    from builtins import basestring


def _recursive_repr(item):
    """Hack around python `repr` to deterministically represent dictionaries.

    This is able to represent more things than json.dumps, since it does not require
    things to be JSON serializable (e.g. datetimes).
    """
    if isinstance(item, base_str):
        result = str(item)
    elif isinstance(item, list):
        result = '[{}]'.format(', '.join([_recursive_repr(x) for x in item]))
    elif isinstance(item, dict):
        kv_pairs = [
            '{}: {}'.format(_recursive_repr(k), _recursive_repr(item[k]))
            for k in sorted(item)
        ]
        result = '{' + ', '.join(kv_pairs) + '}'
    else:
        result = repr(item)
    return result


def get_hash(item):
    repr_ = _recursive_repr(item).encode('utf-8')
    return hashlib.md5(repr_).hexdigest()


def get_hash_int(item):
    return int(get_hash(item), base=16)


def escape_chars(text, chars):
    """
    Helper function to escape uncomfortable characters.

    This function takes a string and a set of characters that need to be escaped.
    Any occurrence of these characters in the text will be replaced with the character 
    prefixed by a backslash (e.g., 'a' becomes '\a').

    Args:
        text (str): The input text in which characters will be escaped.
        chars (str): A string containing characters to be escaped.

    Returns:
        str: The modified text with specified characters escaped.

    Example:
        >>> escape_chars("Hello, World!", ",!")
        'Hello\, World\!'
        
    Notes:
        - The function first converts the input text to a string to ensure compatibility.
        - The characters to be escaped are converted to a list of unique characters.
        - The backslash itself is treated specially: if it is in the list of characters to escape,
          it will be moved to the beginning of the list to prevent double escaping.
    """
    text = str(text)
    chars = list(set(chars))
    if '\\' in chars:
        chars.remove('\\')
        chars.insert(0, '\\')
    for ch in chars:
        text = text.replace(ch, '\\' + ch)
    return text


def convert_kwargs_to_cmd_line_args(kwargs):
    """
    Helper function to build command line arguments out of dict.

    This function takes a dictionary of keyword arguments and converts it into a list of command line arguments.
    The resulting list will contain the keys and values of the dictionary, prefixed with a hyphen (-).
    If the value is a list, it will be expanded into multiple arguments, one for each element.
    If the value is None, it will not be included in the output.

    Args:
        kwargs (dict): A dictionary of keyword arguments.

    Returns:
        list: A list of command line arguments, including the keys and values of the dictionary.

    Example:
        >>> convert_kwargs_to_cmd_line_args({'a': 1, 'b': [2, 3], 'c': None})
        ['-a', '1', '-b', '2', '-b', '3']
    """
    args = []
    append = args.append
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if isinstance(v, Iterable) and not isinstance(v, str):
            for value in v:
                append('-{}'.format(k))
                if value is not None:
                    append(str(value))
            continue
        append('-{}'.format(k))
        if v is not None:
            append(str(v))
    return args