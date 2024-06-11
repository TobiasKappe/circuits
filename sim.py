import json
import queue

from functools import singledispatchmethod
from typing import List, Union

from ouca.riscv.data import RiscInteger


class ContentionException(Exception):
    pass


class CircuitSubscopeException(Exception):
    pass


class Node:
    def __init__(
        self,
        value: Union[List[bool], None] = None,
        connections=None,
        bitwidth=1,
        index=None,
    ):
        self.value = value
        self.upstream = None
        self.connections = connections or []
        self.bitwidth = bitwidth
        self.index = index

    @classmethod
    def load(cls, raw_element: dict, index=None):
        return cls(bitwidth=raw_element['bitWidth'], index=index)

    def propagate(self, seen):
        for connection in self.connections:
            if connection in seen:
                continue

            if connection.value != self.value:
                if connection.upstream is None:
                    connection.upstream = self
                elif connection.upstream is not self:
                    raise ContentionException(
                        f'Nodes {self.index} and {connection.upstream.index} '
                        f'are competing for node {connection.index}'
                    )

                connection.value = self.value
                yield connection
                yield from connection.propagate(seen | {self})


class Element:
    @singledispatchmethod
    def __init__(
        self,
        nodes: List[Node],
        *,
        bitwidth: int = 1,
        label: str = '',
        delay: int = 0
    ):
        self.label = label
        self.delay = delay
        self.bitwidth = bitwidth
        self.nodes = nodes

    @__init__.register
    def load(self, raw_element: dict, nodes: List[Node], context: dict):
        self.label = raw_element['label']
        self.delay = raw_element['propagationDelay']
        self.params = raw_element['customData']['constructorParamaters']

        raw_nodes = raw_element['customData'].get('nodes', {})
        self.nodes = set()
        for identifier, ids in raw_nodes.items():
            if isinstance(ids, int):
                setattr(self, identifier, nodes[ids])
                self.nodes |= {nodes[ids]}
            else:
                local_nodes = [nodes[i] for i in ids]
                setattr(self, identifier, local_nodes)
                self.nodes |= set(local_nodes)

    def is_resolvable(self):
        raise NotImplementedError

    def resolve(self):
        raise NotImplementedError


class InputElement(Element):
    @singledispatchmethod
    def __init__(self, node: Node, state=None, **kwargs):
        super().__init__([node], **kwargs)
        self.output1 = node
        self.state = state

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.state = []
        self.bitwidth = self.params[1]
        raw_state = raw_element['customData']['values']['state']
        for i in range(self.bitwidth):
            self.state.append(raw_state & (1 << i) > 0)

    def is_resolvable(self):
        return True

    def resolve(self):
        self.output1.value = self.state
        yield self.output1


class OutputElement(Element):
    @singledispatchmethod
    def __init__(self, inp1: Node, value=None, **kwargs):
        super().__init__([inp1], **kwargs)
        self.inp1 = inp1
        self.value = value

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.value = None

    def is_resolvable(self):
        return True

    def resolve(self):
        self.value = self.inp1.value
        yield from ()


class Circuit:
    IMPL_MAP = {
        'Input': InputElement,
        'Output': OutputElement,
    }

    def __init__(self, elements: List[Element], nodes: List[Node]):
        self.elements = elements
        self.nodes = nodes

    @classmethod
    def add_impl(cls, name):
        def wrapper(impl):
            cls.IMPL_MAP[name] = impl
            return impl

        return wrapper

    @property
    def inputs(self):
        return [e for e in self.elements if isinstance(e, InputElement)]

    def get_input(self, label):
        for input in self.inputs:
            if input.label == label:
                return input

    @property
    def outputs(self):
        return [e for e in self.elements if isinstance(e, OutputElement)]

    def get_output(self, label):
        for output in self.outputs:
            if output.label == label:
                return output

    def simulate(self):
        for node in self.nodes:
            node.upstream = None

        q = queue.Queue()
        for element in self.elements:
            if element.is_resolvable():
                for n in element.resolve():
                    q.put(n)

        while not q.empty():
            node = q.get()
            changed = set(node.propagate(set()))

            for element in self.elements:
                if set(element.nodes) & changed and element.is_resolvable():
                    for n in element.resolve():
                        if n.upstream is None:
                            n.upstream = element
                        elif n.upstream is not element:
                            raise ContentionException(
                                f'Node {n.index} is receiving a value from '
                                'something other than its attached element '
                                f'(of type {type(element)}).'
                            )

                        q.put(n)

    @classmethod
    def load_scope_map(cls, scopes: dict):
        ret = {}
        for scope in scopes:
            ret[scope['id']] = scope

        return ret

    @classmethod
    def load_obj(cls, obj, main_scope_name, context=None):
        context = context or cls.load_scope_map(obj['scopes'])

        for raw_scope in obj['scopes']:
            if raw_scope['name'] == main_scope_name:
                return cls.load(raw_scope, context)

        raise Exception(f'Could not find scope named {main_scope_name}')

    @classmethod
    def load_str(cls, string, main_scope_name, context=None):
        return cls.load_obj(
            json.loads(string),
            main_scope_name,
            context
        )

    @classmethod
    def load_file(cls, filename, main_scope_name, context=None):
        with open(filename, 'r') as handle:
            contents = handle.read()

        return cls.load_str(contents, main_scope_name, context)

    @classmethod
    def load(cls, raw_scope, context):
        nodes = []
        for i, raw_node in enumerate(raw_scope['allNodes']):
            nodes.append(Node.load(raw_node, index=i))

        for i, raw_node in enumerate(raw_scope['allNodes']):
            for j in raw_node['connections']:
                nodes[i].connections.append(nodes[j])

        elements = []
        for name, raw_elements in raw_scope.items():
            if not name[0].isupper():
                # Keys that do not start with an upper case letter are
                # scope metadata, not elements; we can ignore them.
                continue

            if name not in Circuit.IMPL_MAP:
                raise Exception(f'Unknown element: "{name}"')

            impl = Circuit.IMPL_MAP[name]
            for raw_element in raw_elements:
                element = impl(
                    raw_element,
                    nodes=nodes,
                    context=context
                )
                elements.append(element)

        return cls(elements, nodes)

    def input_vectors(self, lengths):
        if len(lengths) == 0:
            yield []
        else:
            for value in self.input_vectors(lengths[1:]):
                for i in range(2 ** lengths[0]):
                    yield [i] + value

    def element_of(self, node):
        for element in self.elements:
            if node in element.nodes:
                return element


class CombinatorialElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        inp: List[Node],
        output1: Node,
        arity=2,
        bitwidth=1,
        **kwargs
    ):
        super().__init__(inp + [output1], **kwargs)
        self.inp = inp
        self.output1 = output1

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.arity = self.params[1]
        self.bitwidth = self.params[2]

    def is_resolvable(self):
        for i in range(self.bitwidth):
            if self.is_resolvable_per_bit(i):
                return True
        return False

    def is_resolvable_per_bit(self):
        raise NotImplementedError

    def resolve(self):
        result = []
        for i in range(self.bitwidth):
            result.append(self.resolve_per_bit(i))

        self.output1.value = result
        yield self.output1

    def resolve_per_bit(self, i):
        raise NotImplementedError


@Circuit.add_impl('AndGate')
class AndGateElement(CombinatorialElement):
    def is_resolvable_per_bit(self, i):
        for inp in self.inp:
            if inp.value is not None and \
             i < len(inp.value) and \
             inp.value[i] is False:
                return True
        for inp in self.inp:
            if inp.value is None or \
             i >= len(inp.value) or \
             inp.value[i] is None:
                return False
        return True

    def resolve_per_bit(self, i):
        for inp in self.inp:
            if inp.value is not None and \
             i < len(inp.value) and \
             inp.value[i] is False:
                return False
        return True


@Circuit.add_impl('NorGate')
class NorGateElement(CombinatorialElement):
    def is_resolvable_per_bit(self, i):
        for inp in self.inp:
            if inp.value is not None and \
             i < len(inp.value) and \
             inp.value[i] is True:
                return True
        for inp in self.inp:
            if inp.value is None or \
             i >= len(inp.value) or \
             inp.value[i] is None:
                return False
        return True

    def resolve_per_bit(self, i):
        for inp in self.inp:
            if inp.value is not None and \
             i < len(inp.value) and \
             inp.value[i] is True:
                return False
        return True


@Circuit.add_impl('OrGate')
class OrGateElement(CombinatorialElement):
    def is_resolvable_per_bit(self, i):
        for inp in self.inp:
            if inp.value is not None and \
             i < len(inp.value) and \
             inp.value[i] is True:
                return True
        for inp in self.inp:
            if inp.value is None or \
             i >= len(inp.value) or \
             inp.value[i] is None:
                return False
        return True

    def resolve_per_bit(self, i):
        for inp in self.inp:
            if inp.value is not None \
             and i < len(inp.value) and \
             inp.value[i] is True:
                return True
        return False


@Circuit.add_impl('NandGate')
class NandGateElement(CombinatorialElement):
    def is_resolvable_per_bit(self, i):
        for inp in self.inp:
            if inp.value is not None \
             and i < len(inp.value) and \
             inp.value[i] is True:
                return True
        for inp in self.inp:
            if inp.value is None or \
             i >= len(inp.value) or \
             inp.value[i] is None:
                return False
        return True

    def resolve_per_bit(self, i):
        for inp in self.inp:
            if inp.value is not None \
             and i < len(inp.value) and \
             inp.value[i] is True:
                return False
        return True


class ParityGateElement(CombinatorialElement):
    start_parity = None

    def is_resolvable_per_bit(self, i):
        for inp in self.inp:
            if inp.value is None or inp.value[i] is None:
                return False
        return True

    def resolve_per_bit(self, i):
        parity = self.start_parity
        for inp in self.inp:
            if inp.value[i] is True:
                parity = not parity
        return parity


@Circuit.add_impl('XorGate')
class XorGateElement(ParityGateElement):
    start_parity = False


@Circuit.add_impl('XnorGate')
class XnorGateElement(ParityGateElement):
    start_parity = True


@Circuit.add_impl('NotGate')
class NotGateElement(Element):
    @singledispatchmethod
    def __init__(self, inp1: Node, output1: Node, **kwargs):
        super().__init__([inp1, output1], **kwargs)
        self.inp1 = inp1
        self.output1 = output1

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)

    def is_resolvable(self):
        if self.inp1.value is None:
            return False
        for val in self.inp1.value:
            if val is None:
                return False
        return True

    def resolve(self):
        self.output1.value = [not v for v in self.inp1.value]
        yield self.output1


@Circuit.add_impl('Splitter')
class SplitterElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        inp1: Node,
        outputs: Node,
        bitwidth_in: int,
        bitwidths_out: List[int],
        **kwargs
    ):
        assert bitwidth_in == sum(bitwidths_out)

        super().__init__([inp1] + outputs, **kwargs)
        self.inp1 = inp1
        self.outputs = outputs
        self.bitwidths_out = bitwidths_out
        self.prev_inp = None
        self.prev_outputs = [None for length in bitwidths_out]

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.bitwidths_out = self.params[2]
        self.prev_inp = None
        self.prev_outputs = [None for length in self.bitwidths_out]

        # The bitwidths on output nodes of splitters are incorrect as exported
        # from CircuitVerse; we patch them up as best we can.
        for i, bitwidth in enumerate(self.bitwidths_out):
            self.outputs[i].bitwidth = bitwidth

    def is_resolvable(self):
        if self.prev_inp != self.inp1.value:
            if self.inp1.value is None:
                return False
            return None not in self.inp1.value
        elif self.prev_outputs != [o.value for o in self.outputs]:
            for output in self.outputs:
                if output.value is None:
                    return False
                if None in output.value:
                    return False
            return True
        return False

    def resolve(self):
        if self.prev_inp != self.inp1.value:
            pos = 0
            for i, width in enumerate(self.bitwidths_out):
                self.outputs[i].value = self.inp1.value[pos:pos+width]
                yield self.outputs[i]
                pos = pos + width
        elif self.prev_outputs != [o.value for o in self.outputs]:
            combined = []
            for output in self.outputs:
                combined += output.value
            self.inp1.value = combined
            yield self.inp1

        self.prev_inp = self.inp1.value
        self.prev_outputs = [o.value for o in self.outputs]


@Circuit.add_impl('SubCircuit')
class SubCircuitElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        circuit: Circuit,
        inputs: List[Node],
        outputs: List[Node],
        **kwargs
    ):
        self.circuit = circuit
        self.inputs = inputs
        self.outputs = outputs
        super().__init__(inputs + outputs, **kwargs)

    @__init__.register
    def load(self, raw_element: dict, nodes: List[Node], context: dict):
        subcircuit_id = int(raw_element['id'])
        if subcircuit_id in context:
            self.circuit = Circuit.load(context[subcircuit_id], context)
        else:
            raise CircuitSubscopeException(
                f'Could not find subscope with id {subcircuit_id}'
            )

        self.inputs = []
        for raw_input_node in raw_element['inputNodes']:
            self.inputs.append(nodes[raw_input_node])

        self.outputs = []
        for raw_output_node in raw_element['outputNodes']:
            self.outputs.append(nodes[raw_output_node])

        self.nodes = set(self.inputs + self.outputs)

    def is_resolvable(self):
        return True

    def resolve(self):
        for i, inp in enumerate(self.circuit.inputs):
            inp.state = self.inputs[i].value
        old_values = [o.value for o in self.circuit.outputs]

        self.circuit.simulate()
        for i, output in enumerate(self.circuit.outputs):
            if old_values[i] != output.value:
                self.outputs[i].value = output.value
                yield self.outputs[i]


@Circuit.add_impl('Multiplexer')
class MultiplexerElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        signal_size: int,
        controlSignalInput: Node,
        inp: List[Node],
        output1: Node,
        **kwargs
    ):
        self.controlSignalInput = controlSignalInput
        self.signal_size = signal_size
        self.inp = inp
        self.output1 = output1
        super().__init__([self.controlSignalInput] + inp + [output1], **kwargs)

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.signal_size = self.params[2]
        self.bitwidth = self.params[1]

    def control_index(self):
        index = 0
        for val in reversed(self.controlSignalInput.value):
            index = index << 1
            if val is True:
                index |= 1
        return index

    def is_resolvable(self):
        if self.controlSignalInput.value is None:
            return False
        if None in self.controlSignalInput.value:
            return False

        index = self.control_index()
        return self.inp[index].value is not None

    def resolve(self):
        index = self.control_index()
        self.output1.value = self.inp[index].value
        yield self.output1


@Circuit.add_impl('Demultiplexer')
class DemultiplexerElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        signal_size: int,
        controlSignalInput: Node,
        input: Node,
        output1: List[Node],
        **kwargs
    ):
        self.signal_size = signal_size
        self.controlSignalInput = controlSignalInput
        self.input = input
        self.output1 = output1
        super().__init__(
            [self.controlSignalInput, self.input] + self.output1,
            **kwargs
        )

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.signal_size = self.params[2]
        self.bitwidth = self.params[1]

    def control_index(self):
        index = 0
        for val in reversed(self.controlSignalInput.value):
            index = index << 1
            if val is True:
                index |= 1
        return index

    def is_resolvable(self):
        if self.controlSignalInput.value is None:
            return False
        if None in self.controlSignalInput.value:
            return False
        if self.input.value is None:
            return False

        return True

    def resolve(self):
        index = self.control_index()

        for i, output in enumerate(self.output1):
            if i == index:
                output.value = self.input.value
            else:
                output.value = [False] * self.bitwidth

            yield output


@Circuit.add_impl('Decoder')
class DecoderElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        signal_size: int,
        input: Node,
        output1: List[Node],
        **kwargs
    ):
        self.signal_size = signal_size
        self.input = input
        self.output1 = output1
        super().__init__([self.input] + self.output1, **kwargs)

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.signal_size = self.params[1]

    def is_resolvable(self):
        if self.input.value is None:
            return False
        if None in self.input.value:
            return False

        return True

    def resolve(self):
        index = 0
        for val in reversed(self.input.value):
            index = index << 1
            if val is True:
                index |= 1

        for i in range(2**self.signal_size):
            self.output1[i].value = [i == index]
            yield self.output1[i]


class SBElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        input: Node,
        output1: List[Node],
        enable: Node,
        **kwargs
    ):
        self.inp1 = input
        self.output1 = output1
        self.enable = enable
        super().__init__([self.inp1, self.output1, self.enable], **kwargs)

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.bitwidth = self.params[1]

    def output_index(self, index):
        if index == -1:
            self.output1.value = [False] * self.bitwidth
            self.enable.value = [False]
        else:
            bits = []
            while index:
                bits.append(index % 2 == 1)
                index = index >> 1

            self.output1.value = bits + [False] * (self.bitwidth - len(bits))
            self.enable.value = [True]

        yield self.enable
        yield self.output1


@Circuit.add_impl('MSB')
class MSBElement(SBElement):
    def is_resolvable(self):
        if self.inp1.value is None:
            return False
        for val in reversed(self.inp1.value):
            if val is True:
                return True
            if val is None:
                return False

        return True

    def resolve(self):
        index = -1
        for i, val in enumerate(self.inp1.value):
            if val:
                index = i

        return self.output_index(index)


@Circuit.add_impl('LSB')
class LSBElement(SBElement):
    def is_resolvable(self):
        if self.inp1.value is None:
            return False
        for val in self.inp1.value:
            if val is True:
                return True
            if val is None:
                return False

        return True

    def resolve(self):
        index = -1
        for i, val in enumerate(self.inp1.value):
            if val:
                index = i
                break

        return self.output_index(index)


@Circuit.add_impl('ConstantVal')
class ConstantValElement(Element):
    @singledispatchmethod
    def __init__(self, value: List[bool], output1: Node, **kwargs):
        super().__init__([output1], **kwargs)
        self.value = value
        self.output1 = output1
        assert self.bitwidth == len(value)

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.value = [v == "1" for v in self.params[2]][::-1]
        self.bitwidth = self.params[1]
        assert self.bitwidth == len(self.value)

    def is_resolvable(self):
        return True

    def resolve(self):
        self.output1.value = self.value
        yield self.output1


class FixedValElement(Element):
    fixed_value = None

    @singledispatchmethod
    def __init__(self, output1: Node, **kwargs):
        super().__init__([output1], **kwargs)
        self.output1 = output1

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.bitwidth = self.params[0]

    def is_resolvable(self):
        return True

    def resolve(self):
        self.output1.value = [self.fixed_value] * self.bitwidth
        yield self.output1


@Circuit.add_impl('Ground')
class GroundElement(FixedValElement):
    fixed_value = False


@Circuit.add_impl('Power')
class PowerElement(FixedValElement):
    fixed_value = True


@Circuit.add_impl('PriorityEncoder')
class PriorityEncoderElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        inp1: List[Node],
        output1: List[Node],
        enable: Node,
        **kwargs
    ):
        super().__init__(inp1 + output1 + [enable], **kwargs)
        self.inp1 = inp1
        self.output1 = output1
        self.enable = enable
        self.sanity_check()

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.bitwidth = self.params[1]
        self.sanity_check()

    def sanity_check(self):
        assert len(self.output1) == self.bitwidth
        assert len(self.inp1) == 2**self.bitwidth

        for node in self.nodes:
            assert node.bitwidth == 1

    def is_resolvable(self):
        for inp in self.inp1:
            if inp.value is None:
                return False
        return True

    def resolve(self):
        index = -1
        for i, inp in enumerate(self.inp1):
            if inp.value == [True]:
                index = i

        if index < 0:
            self.enable.value = [False]
        else:
            for outp in self.output1:
                outp.value = [index & 1 > 0]
                yield outp
                index = index >> 1
            self.enable.value = [True]

        yield self.enable


@Circuit.add_impl('SignExtend')
class SignExtendElement(Element):
    @singledispatchmethod
    def __init__(self, input: Node, output: Node, **kwargs):
        super().__init__([input, output], **kwargs)
        self.input = input
        self.output = output
        assert self.input.bitwidth == self.bitwidth

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.bitwidth = self.params[1]
        assert self.input.bitwidth == self.bitwidth

    def is_resolvable(self):
        return self.input.value is not None

    def resolve(self):
        self.output.value = \
            self.input.value + [self.input.value[-1]] * (32-self.bitwidth)
        yield self.output


@Circuit.add_impl('ProgramCounter')
class ProgramCounterElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        dist: Node,
        jump: Node,
        reset: Node,
        ctr: Node,
        clock: Node,
        **kwargs
    ):
        super().__init__([dist, jump, reset, ctr, clock], **kwargs)
        self.dist = dist
        self.jump = jump
        self.reset = reset
        self.ctr = ctr
        self.clock = clock
        self.prev_clock = None
        self.value = RiscInteger(0)
        self.validate()

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.value = RiscInteger(0)
        self.validate()

    def validate(self):
        assert self.dist.bitwidth == 32
        assert self.jump.bitwidth == 1
        assert self.reset.bitwidth == 1
        assert self.clock.bitwidth == 1
        assert self.ctr.bitwidth == 32

    def is_resolvable(self):
        return True

    def resolve(self):
        if self.reset.value is not None and self.reset.value[0] is True:
            self.value = RiscInteger(0)
        elif (self.clock.value is not None and self.clock.value[0] is True and
              self.prev_clock is False):
            if self.jump.value is not None and self.jump.value[0] is True:
                if self.dist.value is not None:
                    self.value += RiscInteger(self.dist.value)
            else:
                self.value += RiscInteger(4)

        self.ctr.value = self.value.bits
        yield self.ctr

        if self.clock.value is not None:
            self.prev_clock = self.clock.value[0]


@Circuit.add_impl('RAM')
class RAMElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        clock: Node,
        dataIn: Node,
        dataOut1: Node,
        dataOut2: Node,
        en: Node,
        memAddr1: Node,
        memAddr2: Node,
        values: Union[List[RiscInteger], None] = None,
        **kwargs
    ):
        super().__init__(
            [clock, dataIn, dataOut1, dataOut2, en, memAddr1, memAddr2],
            **kwargs
        )
        self.clock = clock
        self.dataIn = dataIn
        self.dataOut1 = dataOut1
        self.dataOut2 = dataOut2
        self.en = en
        self.memAddr1 = memAddr1
        self.memAddr2 = memAddr2
        self.values = values or [RiscInteger(0) for _ in range(16)]
        self.prev_clock = None
        self.validate()

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.values = [
            RiscInteger(v)
            for v in raw_element['customData']['constructorParamaters']
        ]
        self.prev_clock = None
        self.validate()

    def validate(self):
        assert len(self.values) == 16
        assert all(isinstance(v, RiscInteger) for v in self.values)

        assert self.clock.bitwidth == 1
        assert self.dataIn.bitwidth == 32
        assert self.dataOut1.bitwidth == 32
        assert self.dataOut2.bitwidth == 32
        assert self.en.bitwidth == 1
        assert self.memAddr1.bitwidth == 32
        assert self.memAddr2.bitwidth == 32

    def is_resolvable(self):
        return True

    def resolve(self):
        if self.memAddr1.value is not None:
            addr = (RiscInteger(self.memAddr1.value) & RiscInteger(0x3F)) >> 2
            if (self.prev_clock is False and self.clock.value[0] is True and
               self.en.value is not None and self.en.value[0] is True and
               self.dataIn.value is not None):
                self.values[addr.to_int()] = RiscInteger(self.dataIn.value)
            self.dataOut1.value = self.values[addr.to_int()].bits
            yield self.dataOut1

        if self.memAddr2.value is not None:
            addr = (RiscInteger(self.memAddr2.value) & RiscInteger(0x3F)) >> 2
            self.dataOut2.value = self.values[addr.to_int()].bits
            yield self.dataOut2

        if self.clock.value is not None:
            self.prev_clock = self.clock.value[0]


@Circuit.add_impl('RegisterFile')
class RegisterFileElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        R1: Node,
        R2: Node,
        R3: Node,
        clock: Node,
        dataIn: Node,
        dataOut1: Node,
        dataOut2: Node,
        en: Node,
        values: Union[List[RiscInteger], None] = None,
        **kwargs
    ):
        super().__init__(
            [R1, R2, R3, clock, dataIn, dataOut1, dataOut2, en],
            **kwargs
        )
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.clock = clock
        self.dataIn = dataIn
        self.dataOut1 = dataOut1
        self.dataOut2 = dataOut2
        self.en = en
        self.values = values or [RiscInteger(0) for _ in range(32)]
        self.prev_clock = None
        self.validate()

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.values = [
            RiscInteger(v)
            for v in raw_element['customData']['constructorParamaters']
        ]
        self.prev_clock = None
        self.validate()

    def validate(self):
        assert len(self.values) == 32
        assert all(isinstance(v, RiscInteger) for v in self.values)

        assert self.R1.bitwidth == 5
        assert self.R2.bitwidth == 5
        assert self.R3.bitwidth == 5
        assert self.dataIn.bitwidth == 32
        assert self.en.bitwidth == 1
        assert self.clock.bitwidth == 1
        assert self.dataOut1.bitwidth == 32
        assert self.dataOut2.bitwidth == 32

    def is_resolvable(self):
        return True

    def resolve(self):
        if self.R3.value is not None and \
           self.dataIn.value is not None and \
           self.prev_clock is False and \
           self.clock.value is not None and \
           self.clock.value[0] is True and \
           self.en.value is not None and \
           self.en.value[0] is True and \
           any(self.R3.value):
            addr = RiscInteger(self.R3.value)
            self.values[addr.to_int()] = RiscInteger(self.dataIn.value)

        if self.R1.value is not None:
            addr = RiscInteger(self.R1.value)
            self.dataOut1.value = self.values[addr.to_int()].bits
            yield self.dataOut1

        if self.R2.value is not None:
            addr = RiscInteger(self.R2.value)
            self.dataOut2.value = self.values[addr.to_int()].bits
            yield self.dataOut2

        if self.clock.value is not None:
            self.prev_clock = self.clock.value[0]


@Circuit.add_impl('Text')
class TextElement(Element):
    def is_resolvable(self):
        return False


@Circuit.add_impl('Rectangle')
class RectangleElement(Element):
    def is_resolvable(self):
        return False


@Circuit.add_impl('Arrow')
class ArrowElement(Element):
    def is_resolvable(self):
        return False


@Circuit.add_impl('ImageAnnotation')
class ImageAnnotationElement(Element):
    def is_resolvable(self):
        return False


@Circuit.add_impl('SevenSegDisplay')
class SevenSegDispleyElement(Element):
    def is_resolvable(self):
        return False


@Circuit.add_impl('DigitalLed')
class DigitalLedElement(Element):
    @singledispatchmethod
    def __init__(self, inp1: Node, **kwargs):
        super().__init__(input, **kwargs)
        self.state = False
        self.inp1 = input

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)

    def is_resolvable(self):
        if self.inp1.value is None:
            return False
        if self.inp1.value == []:
            return False

        return True

    def resolve(self):
        self.state = self.inp1.value[0]
        yield from ()
