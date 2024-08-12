import pytest

from circuits import Node
from circuits.mux import MultiplexerElement


class TestMultiplexer:
    @pytest.mark.parametrize(
        'signal_size, signal_val, input_vals, resolvable',
        [
            [1, None, [[False], [True]], False],
            [2, [False, False], [None, [True], [True], [True]], False],
            [2, [False, True], [None, [True], [True], [True]], True],
        ]
    )
    def test_is_resolvable(
        self,
        signal_size,
        signal_val,
        input_vals,
        resolvable,
    ):
        control = Node()
        inputs = [Node() for _ in range(2**signal_size)]
        output = Node()
        element = MultiplexerElement(signal_size, control, inputs, output)

        control.value = signal_val
        for i in range(2**signal_size):
            inputs[i].value = input_vals[i]

        assert element.is_resolvable() is resolvable

    @pytest.mark.parametrize(
        'signal_size, signal_val, input_vals, output_val',
        [
            [1, [False], [[False], [True]], [False]],
            [2, [True, False], [None, [True], [True], [True]], [True]],
            [2, [False, True], [[True], [True], [False], [True]], [False]],
        ]
    )
    def test_resolve(
        self,
        signal_size,
        signal_val,
        input_vals,
        output_val,
    ):
        control = Node()
        inputs = [Node() for _ in range(2**signal_size)]
        output = Node()
        element = MultiplexerElement(signal_size, control, inputs, output)

        control.value = signal_val
        for i in range(2**signal_size):
            inputs[i].value = input_vals[i]

        nodes = list(element.resolve())
        assert nodes == [output]
        assert output.value == output_val
