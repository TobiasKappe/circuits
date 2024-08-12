import pytest

from circuits import Node
from circuits.mux import DemultiplexerElement


class TestDemultiplexer:
    @pytest.mark.parametrize(
        'signal_size, signal_val, input_val, resolvable',
        [
            [1, None, [False], False],
            [1, [False], None, False],
            [1, [None], [False], False],
            [1, [False], [None], True],
            [1, [False], [True], True],
            [2, [False, None], [True], False],
        ]
    )
    def test_is_resolvable(
        self,
        signal_size,
        signal_val,
        input_val,
        resolvable,
    ):
        control = Node()
        input = Node()
        outputs = [Node() for _ in range(2**signal_size)]
        element = DemultiplexerElement(
            signal_size,
            control,
            input,
            outputs
        )

        control.value = signal_val
        input.value = input_val

        assert element.is_resolvable() is resolvable

    @pytest.mark.parametrize(
        'bitwidth, signal_size, signal_val, input_val, output_vals',
        [
            [1, 1, [False], [True], [[True], [False]]],
            [1, 1, [True], [True], [[False], [True]]],
            [1, 1, [False], [None], [[None], [False]]],
            [1, 2, [False, True], [True], [[False], [False], [True], [False]]],
            [
                2,
                2,
                [False, True],
                [True, True],
                [
                    [False, False],
                    [False, False],
                    [True, True],
                    [False, False]
                ]
            ],
        ]
    )
    def test_resolve(
        self,
        bitwidth,
        signal_size,
        signal_val,
        input_val,
        output_vals,
    ):
        control = Node()
        input = Node()
        outputs = [Node() for _ in range(2**signal_size)]
        element = DemultiplexerElement(
            signal_size,
            control,
            input,
            outputs,
            bitwidth=bitwidth,
        )

        control.value = signal_val
        input.value = input_val

        nodes = list(element.resolve())

        for node in nodes:
            assert node in outputs
        for i, output in enumerate(outputs):
            assert output in nodes
            assert output.value == output_vals[i]
