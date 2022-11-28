from typing import TypeVar
NumberVar = float | int

NV = TypeVar('NV', bound=NumberVar)

__all__ = [
    'clamp'
]

def clamp(value: NV, minValue: NV, maxValue: NV) -> NV:
    return min(max(value, minValue), maxValue)

def linked_list_comparison(l0: list, l1: list, minStartIdx: int=0) -> tuple[list[int], list[int]]:
    """returns overlap indices of l0 and l1, respectively"""
    def get_index(vals: list, val) -> int | None:
        try:
            return vals.index(val)
        except ValueError:
            return None
    
    l1_matches: list[int | None] = [get_index(l0, val) for val in l1]
    relevant_l0 = []; relevant_l1 = []
    for i, match in enumerate(l1_matches):
        if match is not None:
            if match > minStartIdx and (len(relevant_l0) == 0 or match > relevant_l0[-1]):
                relevant_l0.append(match)
                relevant_l1.append(i)
            else:
                pass
    return relevant_l0, relevant_l1

def str_llc(s0: str, s1: str, delimiter: str=' '):
    l0 = s0.split(" "); l1 = s1.split(" ")
    m0, m1 = linked_list_comparison(l0, l1)
    print(' '.join([(val.upper() if idx in m0 else val.lower()) for idx, val in enumerate(l0)]))
    print(' '.join([(val.upper() if idx in m1 else val.lower()) for idx, val in enumerate(l1)]))
