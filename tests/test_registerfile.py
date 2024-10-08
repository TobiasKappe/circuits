import pytest

from circuits import Node
from circuits.register import RegisterFileElement
from circuits.utils import array_to_int, int_to_array


class TestRegisterFileElement:
    def test_is_resolvable(self):
        R = [Node(bitwidth=5) for _ in range(3)]
        clock = Node()
        dataIn = Node(bitwidth=32)
        dataOut = [Node(bitwidth=32), Node(bitwidth=32)]
        en = Node()

        rf = RegisterFileElement(
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
        R = [Node(bitwidth=5) for _ in range(3)]
        clock = Node()
        dataIn = Node(bitwidth=32)
        dataOut = [Node(bitwidth=32), Node(bitwidth=32)]
        en = Node()

        rf = RegisterFileElement(
            R[0],
            R[1],
            R[2],
            clock,
            dataIn,
            dataOut[0],
            dataOut[1],
            en,
            values=list(range(32)),
        )

        # All inputs undefined, so no defined output.
        nodes = list(rf.resolve())
        assert not nodes
        assert dataOut[index].value is None

        # A defined input address yields a defined output.
        R[index].value = [True, False]*2 + [False]
        nodes = list(rf.resolve())
        assert nodes == [dataOut[index]]
        assert dataOut[index].value == int_to_array(5)

    def test_write(self):
        R = [Node(bitwidth=5) for _ in range(3)]
        clock = Node()
        dataIn = Node(bitwidth=32)
        dataOut = [Node(bitwidth=32), Node(bitwidth=32)]
        en = Node()

        rf = RegisterFileElement(
            R[0],
            R[1],
            R[2],
            clock,
            dataIn,
            dataOut[0],
            dataOut[1],
            en,
            values=list(range(32)),
        )

        # Offering data on an undefined memory address does nothing
        dataIn.value = [True, False]*16
        nodes = list(rf.resolve())
        assert not nodes
        assert all(rf.values[i] == i for i in range(32))

        clock.value = [False]
        nodes = list(rf.resolve())
        assert not nodes
        assert all(rf.values[i] == i for i in range(32))

        clock.value = [True]
        nodes = list(rf.resolve())
        assert not nodes
        assert all(rf.values[i] == i for i in range(32))

        # Defined data on a defined address with enable low does nothing
        dataIn.value = [True, False]*16
        R[2].value = [False, True]*2 + [False]
        clock.value = [False]
        en.value = [False]
        nodes = list(rf.resolve())
        assert not nodes
        assert all(rf.values[i] == i for i in range(32))

        clock.value = [True]
        nodes = list(rf.resolve())
        assert not nodes
        assert all(rf.values[i] == i for i in range(32))

        # Offering defined data on a defined address with enable high writes
        # that data to the values at that address
        dataIn.value = [True, False]*16
        R[2].value = [False, True]*2 + [False]
        clock.value = [False]
        en.value = [True]
        nodes = list(rf.resolve())
        assert not nodes
        assert all(rf.values[i] == i for i in range(16))

        clock.value = [True]
        nodes = list(rf.resolve())
        assert not nodes
        assert all(
            rf.values[i] == i
            for i in range(32) if i != 10
        )
        assert rf.values[10] == array_to_int([True, False]*16)
