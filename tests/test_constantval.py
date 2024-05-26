from ouca.circuits import sim


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
