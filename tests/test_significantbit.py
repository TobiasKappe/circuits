from ouca.circuits import sim

import pytest


class TestMSB:
    @pytest.mark.parametrize(
        'bitwidth, input_vals, resolvable',
        [
            [1, None, False],
            [1, [None], False],
            [1, [False], True],
            [1, [True], True],
            [2, [None, None], False],
            [2, [None, True], True],
            [2, [None, False], False],
        ]
    )
    def test_is_resolvable(
        self,
        bitwidth,
        input_vals,
        resolvable,
    ):
        input = sim.Node()
        output = sim.Node()
        enable = sim.Node()
        element = sim.MSBElement(input, output, enable, bitwidth=bitwidth)

        input.value = input_vals

        assert element.is_resolvable() is resolvable

    @pytest.mark.parametrize(
        'bitwidth, input_val, output_val, enable_val',
        [
            [1, [False], [False], False],
            [1, [True], [False], True],
            [2, [None, True], [True, False], True],
            [2, [False, True], [True, False], True],
            [2, [True, True], [True, False], True],
            [2, [True, False], [False, False], True],
        ]
    )
    def test_resolve(
        self,
        bitwidth,
        input_val,
        output_val,
        enable_val,
    ):
        input = sim.Node()
        output = sim.Node()
        enable = sim.Node()
        element = sim.MSBElement(input, output, enable, bitwidth=bitwidth)

        input.value = input_val

        nodes = list(element.resolve())
        assert output in nodes
        assert enable in nodes

        assert output.value == output_val
        assert enable.value == [enable_val]


class TestLSB:
    @pytest.mark.parametrize(
        'bitwidth, input_vals, resolvable',
        [
            [1, None, False],
            [1, [None], False],
            [1, [False], True],
            [1, [True], True],
            [2, [None, None], False],
            [2, [True, None], True],
            [2, [False, None], False],
        ]
    )
    def test_is_resolvable(
        self,
        bitwidth,
        input_vals,
        resolvable,
    ):
        input = sim.Node()
        output = sim.Node()
        enable = sim.Node()
        element = sim.LSBElement(input, output, enable, bitwidth=bitwidth)

        input.value = input_vals

        assert element.is_resolvable() is resolvable

    @pytest.mark.parametrize(
        'bitwidth, input_val, output_val, enable_val',
        [
            [1, [False], [False], False],
            [1, [True], [False], True],
            [2, [True, None], [False, False], True],
            [2, [True, False], [False, False], True],
            [2, [True, True], [False, False], True],
            [2, [False, True], [True, False], True],
        ]
    )
    def test_resolve(
        self,
        bitwidth,
        input_val,
        output_val,
        enable_val,
    ):
        input = sim.Node()
        output = sim.Node()
        enable = sim.Node()
        element = sim.LSBElement(input, output, enable, bitwidth=bitwidth)

        input.value = input_val

        nodes = list(element.resolve())
        assert output in nodes
        assert enable in nodes

        assert output.value == output_val
        assert enable.value == [enable_val]
