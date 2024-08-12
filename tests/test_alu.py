import pytest

from ouca.circuits import Node
from ouca.circuits.calculate import ALUElement


class TestALU:
    @pytest.mark.parametrize(
        'bitwidth, inp1, inp2, controlSignalInput, resolvable',
        [
            [1, None, [False], [False]*3, False],
            [1, [None], [False], [False]*3, False],
            [1, [False], None, [False]*3, False],
            [1, [False], [None], [False]*3, False],
            [1, [False], [False], None, False],
            [1, [False], [False], [None, False, False], False],
            [1, [False], [False], [False]*3, True],
        ]
    )
    def test_is_resolvable(
        self,
        bitwidth,
        inp1,
        inp2,
        controlSignalInput,
        resolvable,
    ):
        element = ALUElement(
            Node(bitwidth=bitwidth),
            Node(bitwidth=bitwidth),
            Node(bitwidth=3),
            Node(bitwidth=bitwidth),
            Node(),
            bitwidth=bitwidth,
        )

        element.inp1.value = inp1
        element.inp2.value = inp2
        element.controlSignalInput.value = controlSignalInput

        assert element.is_resolvable() is resolvable

    @pytest.mark.parametrize(
        'bitwidth, inp1, inp2, controlSignalInput, carryOut, output',
        [
            [
                4,
                [False, False, True, True],
                [False, True, False, True],
                [False, False, False],
                [False],
                [False, False, False, True],
            ],
            [
                4,
                [False, False, True, True],
                [False, True, False, True],
                [True, False, False],
                [False],
                [False, True, True, True],
            ],
            [
                4,
                [False, False, False, False],
                [False, False, False, False],
                [False, True, False],
                [False],
                [False, False, False, False],
            ],
            [
                4,
                [True, False, False, False],
                [True, False, False, False],
                [False, True, False],
                [False],
                [False, True, False, False],
            ],
            [
                4,
                [True, True, True, True],
                [True, False, False, False],
                [False, True, False],
                [True],
                [False, False, False, False],
            ],
            [
                4,
                [False, False, True, True],
                [False, True, False, True],
                [False, False, True],
                [False],
                [False, False, True, False],
            ],
            [
                4,
                [False, False, True, True],
                [False, True, False, True],
                [True, False, True],
                [False],
                [True, False, True, True],
            ],
            [
                4,
                [False, True, False, True],
                [True, True, False, False],
                [False, True, True],
                [False],
                [True, True, True, False],
            ],
            [
                4,
                [False, False, False, False],
                [True, False, False, False],
                [False, True, True],
                [False],
                [True, True, True, True],
            ],
            [
                4,
                [False, False, False, False],
                [True, False, False, False],
                [True, True, True],
                [False],
                [True, False, False, False],
            ],
            [
                4,
                [True, True, False, False],
                [True, False, True, False],
                [True, True, True],
                [False],
                [True, False, False, False],
            ],
        ]
    )
    def test_resolve(
        self,
        bitwidth,
        inp1,
        inp2,
        controlSignalInput,
        carryOut,
        output,
    ):
        element = ALUElement(
            Node(bitwidth=bitwidth),
            Node(bitwidth=bitwidth),
            Node(bitwidth=3),
            Node(bitwidth=bitwidth),
            Node(),
            bitwidth=bitwidth,
        )

        element.inp1.value = inp1
        element.inp2.value = inp2
        element.controlSignalInput.value = controlSignalInput

        nodes = list(element.resolve())
        assert element.carryOut in nodes
        assert element.output in nodes

        assert element.carryOut.value == carryOut
        assert element.output.value == output
