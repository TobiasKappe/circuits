import pytest

from circuits import Node
from circuits.ram import RAMElement
from circuits.utils import int_to_array, array_to_int


class TestRAMElement:
    def test_is_resolvable(self):
        clock = Node()
        dataIn = Node(bitwidth=32)
        dataOut = [Node(bitwidth=32), Node(bitwidth=32)]
        en = Node()
        memAddr = [Node(bitwidth=32), Node(bitwidth=32)]

        ram = RAMElement(
            clock,
            dataIn,
            dataOut[0],
            dataOut[1],
            en,
            memAddr[0],
            memAddr[1]
        )
        assert ram.is_resolvable()

    @pytest.mark.parametrize('index', [0, 1])
    def test_read(self, index):
        clock = Node()
        dataIn = Node(bitwidth=32)
        dataOut = [Node(bitwidth=32), Node(bitwidth=32)]
        en = Node()
        memAddr = [Node(bitwidth=32), Node(bitwidth=32)]

        ram = RAMElement(
            clock,
            dataIn,
            dataOut[0],
            dataOut[1],
            en,
            memAddr[0],
            memAddr[1],
            values=list(range(16))
        )

        # All inputs undefined, so no defined output.
        nodes = list(ram.resolve())
        assert not nodes
        assert dataOut[index].value is None

        # A defined input address yields a defined output.
        memAddr[index].value = [True, False]*3 + [False]*26
        nodes = list(ram.resolve())
        assert nodes == [dataOut[index]]
        assert dataOut[index].value == int_to_array(5)

    def test_write(self):
        clock = Node()
        dataIn = Node(bitwidth=32)
        dataOut = [Node(bitwidth=32), Node(bitwidth=32)]
        en = Node()
        memAddr = [Node(bitwidth=32), Node(bitwidth=32)]

        ram = RAMElement(
            clock,
            dataIn,
            dataOut[0],
            dataOut[1],
            en,
            memAddr[0],
            memAddr[1],
            values=list(range(16))
        )

        # Offering data on an undefined memory address does nothing
        dataIn.value = [True, False]*16
        nodes = list(ram.resolve())
        assert not nodes
        assert dataOut[0].value is None
        assert all(ram.values[i] == i for i in range(16))

        clock.value = [False]
        nodes = list(ram.resolve())
        assert not nodes
        assert dataOut[0].value is None
        assert all(ram.values[i] == i for i in range(16))

        clock.value = [True]
        nodes = list(ram.resolve())
        assert not nodes
        assert dataOut[0].value is None
        assert all(ram.values[i] == i for i in range(16))

        # Offering undefined data on a memory address just outputs the current
        # value at that address
        dataIn.value = None
        nodes = list(ram.resolve())
        assert not nodes
        assert dataOut[0].value is None
        assert all(ram.values[i] == i for i in range(16))

        memAddr[0].value = [False, True]*3 + [False]*26
        nodes = list(ram.resolve())
        assert nodes == [dataOut[0]]
        assert dataOut[0].value == int_to_array(10)
        assert all(ram.values[i] == i for i in range(16))

        clock.value = [False]
        nodes = list(ram.resolve())
        assert nodes == [dataOut[0]]
        assert dataOut[0].value == int_to_array(10)
        assert all(ram.values[i] == i for i in range(16))

        clock.value = [True]
        nodes = list(ram.resolve())
        assert nodes == [dataOut[0]]
        assert dataOut[0].value == int_to_array(10)
        assert all(ram.values[i] == i for i in range(16))

        # Offering defined data on a defined address with enable low does
        # nothing except offer the current value at the output
        dataIn.value = [True, False]*16
        memAddr[0].value = [False, True]*3 + [False]*26
        clock.value = [False]
        en.value = [False]
        nodes = list(ram.resolve())
        assert nodes == [dataOut[0]]
        assert dataOut[0].value == int_to_array(10)
        assert all(ram.values[i] == i for i in range(16))

        clock.value = [True]
        nodes = list(ram.resolve())
        assert nodes == [dataOut[0]]
        assert dataOut[0].value == int_to_array(10)
        assert all(ram.values[i] == i for i in range(16))

        # Offering defined data on a defined address with enable high writes
        # that data to the values at that address, and immediately forwards it
        dataIn.value = [True, False]*16
        memAddr[0].value = [False, True]*3 + [False]*26
        clock.value = [False]
        en.value = [True]
        nodes = list(ram.resolve())
        assert nodes == [dataOut[0]]
        assert dataOut[0].value == int_to_array(10)
        assert all(ram.values[i] == i for i in range(16))

        clock.value = [True]
        nodes = list(ram.resolve())
        assert nodes == [dataOut[0]]
        assert dataOut[0].value == [True, False]*16
        assert all(
            ram.values[i] == i
            for i in range(16) if i != 10
        )
        assert ram.values[10] == array_to_int([True, False]*16)
