import pytest

from riscv.data import RiscInteger

from ouca.circuits import sim


class TestRegisterFileElement:
    def test_is_resolvable(self):
        R = [sim.Node(bitwidth=5) for _ in range(3)]
        clock = sim.Node()
        dataIn = sim.Node(bitwidth=32)
        dataOut = [sim.Node(bitwidth=32), sim.Node(bitwidth=32)]
        en = sim.Node()

        rf = sim.RegisterFileElement(
            R[0],
            R[1],
            R[2],
            clock,
            dataIn,
            dataOut[0],
            dataOut[1],
            en,
        )
        assert rf.is_resolvable()

    @pytest.mark.parametrize('index', [0, 1])
    def test_read(self, index):
        R = [sim.Node(bitwidth=5) for _ in range(3)]
        clock = sim.Node()
        dataIn = sim.Node(bitwidth=32)
        dataOut = [sim.Node(bitwidth=32), sim.Node(bitwidth=32)]
        en = sim.Node()

        rf = sim.RegisterFileElement(
            R[0],
            R[1],
            R[2],
            clock,
            dataIn,
            dataOut[0],
            dataOut[1],
            en,
            values=[RiscInteger(i) for i in range(32)],
        )

        # All inputs undefined, so no defined output.
        nodes = list(rf.resolve())
        assert not nodes
        assert dataOut[index].value is None

        # A defined input address yields a defined output.
        R[index].value = [True, False]*2 + [False]
        nodes = list(rf.resolve())
        assert nodes == [dataOut[index]]
        assert dataOut[index].value == RiscInteger(5).bits

    def test_write(self):
        R = [sim.Node(bitwidth=5) for _ in range(3)]
        clock = sim.Node()
        dataIn = sim.Node(bitwidth=32)
        dataOut = [sim.Node(bitwidth=32), sim.Node(bitwidth=32)]
        en = sim.Node()

        rf = sim.RegisterFileElement(
            R[0],
            R[1],
            R[2],
            clock,
            dataIn,
            dataOut[0],
            dataOut[1],
            en,
            values=[RiscInteger(i) for i in range(32)],
        )

        # Offering data on an undefined memory address does nothing
        dataIn.value = [True, False]*16
        nodes = list(rf.resolve())
        assert not nodes
        assert all(rf.values[i] == RiscInteger(i) for i in range(32))

        clock.value = [False]
        nodes = list(rf.resolve())
        assert not nodes
        assert all(rf.values[i] == RiscInteger(i) for i in range(32))

        clock.value = [True]
        nodes = list(rf.resolve())
        assert not nodes
        assert all(rf.values[i] == RiscInteger(i) for i in range(32))

        # Defined data on a defined address with enable low does nothing
        dataIn.value = [True, False]*16
        R[2].value = [False, True]*2 + [False]
        clock.value = [False]
        en.value = [False]
        nodes = list(rf.resolve())
        assert not nodes
        assert all(rf.values[i] == RiscInteger(i) for i in range(32))

        clock.value = [True]
        nodes = list(rf.resolve())
        assert not nodes
        assert all(rf.values[i] == RiscInteger(i) for i in range(32))

        # Offering defined data on a defined address with enable high writes
        # that data to the values at that address
        dataIn.value = [True, False]*16
        R[2].value = [False, True]*2 + [False]
        clock.value = [False]
        en.value = [True]
        nodes = list(rf.resolve())
        assert not nodes
        assert all(rf.values[i] == RiscInteger(i) for i in range(16))

        clock.value = [True]
        nodes = list(rf.resolve())
        assert not nodes
        assert all(
            rf.values[i] == RiscInteger(i)
            for i in range(32) if i != 10
        )
        assert rf.values[10] == RiscInteger([True, False]*16)
