from ouca.circuits import sim

import pytest


class TestConstantValElement:
    def test_resolve(self):
        output = sim.Node()
        element = sim.ConstantValElement(
            [True, False, True],
            output,
            bitwidth=3,
        )

        nodes = list(element.resolve())
        assert nodes == [output]
        assert output.value == [True, False, True]


class TestFixedValElement:
    @pytest.mark.parametrize(
        'bitwidth, cls, value',
        [
            [1, sim.GroundElement, [False]],
            [2, sim.GroundElement, [False, False]],
            [1, sim.PowerElement, [True]],
            [2, sim.PowerElement, [True, True]],
        ]
    )
    def test_resolve(self, bitwidth, cls, value):
        output = sim.Node()
        element = cls(output, bitwidth=bitwidth)

        nodes = list(element.resolve())
        assert nodes == [output]
        assert output.value == value
