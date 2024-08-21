def array_to_int(values: list[bool], length=32) -> int:
    value = 0
    for i in range(length):
        if values[i]:
            value += 1 << i

    return value


def int_to_array(value: int, length=32) -> list[bool]:
    array = []
    for i in range(length):
        array.append(value & 1 != 0)
        value = value >> 1

    return array


def add_fixed_width(a: int, b: int, length=32) -> int:
    a = int_to_array(a, length)
    b = int_to_array(b, length)
    carry = False
    value = []

    for i in range(length):
        value.append((a[i] != b[i]) != carry)
        carry = a[i] and b[i] or a[i] and carry or b[i] and carry

    return array_to_int(value)
