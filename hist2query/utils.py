from typing import Union

def numerical_or_none(arg: Union[str, int, None],
                      numerical_type: type[int, float]=int):
    """
    Set an argparse arg as either None or a numerical type
    """
    if str(arg).lower() in ("none", "null") or arg is None:
        return None
    return numerical_type(arg)

def str_or_none(arg: Union[str, int, None]=None):
    """
    Set an argparse arg as either None or a numerical type
    """
    if str(arg).lower() in ("none", "null") or arg is None:
        return None
    return str(arg)