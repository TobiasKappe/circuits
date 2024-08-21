import pytest

from circuits.utils import array_to_int, int_to_array, add_fixed_width


@pytest.mark.parametrize(
    'array, expected_int',
    [
        [[True]*32, 2**32 - 1],
        [[False] * 32, 0],
        [[True, False] * 16, 1431655765],
        [[False, True] * 16, 2863311530],
        [[True, False, True] + [False] * 29, 5],
        [[False, True, False, True,
          False, True, False, False] + [False]*24, 42],
        [[False]*32 + [True], 0],
    ]
)
def test_array_to_int(array, expected_int):
    assert array_to_int(array) == expected_int


@pytest.mark.parametrize(
    'int, expected_array',
    [
        [2**32 - 1, [True]*32],
        [0, [False] * 32],
        [1431655765, [True, False] * 16],
        [2863311530, [False, True] * 16],
        [5, [True, False, True] + [False] * 29],
        [42, [False, True, False, True,
              False, True, False, False] + [False]*24],
        [2**32, [False]*32],
    ]
)
def test_int_to_array(int, expected_array):
    assert int_to_array(int) == expected_array


@pytest.mark.parametrize(
    'a, b, expected_sum',
    [
        [0, 0, 0],
        [1, 1, 2],
        [2**32 - 1, 1, 0],
        [0, -1, 2**32-1],
        [17, -13, 4],
        [13, -17, 2**32-4],
    ]
)
def test_add_fixed_width(a, b, expected_sum):
    assert add_fixed_width(a, b) == expected_sum
