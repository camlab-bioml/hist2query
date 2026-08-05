from typing import Union

def numerical_or_none(arg: Union[str, int, None],
                      numerical_type: type[int, float]=int):
    """
    Set an argparse arg as either None or a numerical type
    """
    if str(arg).lower() in ("none", "null"):
        return None
    return numerical_type(arg)