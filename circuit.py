import json
import queue

from typing import List

from ouca.circuits.element import Element
from ouca.circuits.node import Node
from ouca.circuits.exceptions import \
    ContentionException, CircuitUnstableException
from ouca.circuits.io import InputElement, OutputElement
from ouca.circuits.mock import MockElement
from ouca.circuits.registry import ElementRegistry


class Circuit:
    MAX_ITERATIONS = 1000000

    def __init__(self, elements: List[Element], nodes: List[Node]):
        self.elements = elements
        self.nodes = nodes

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

        iterations = 0
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
                                f'(of type {type(element).__name__}).'
                            )

                        q.put(n)

            iterations += 1
            if iterations > self.MAX_ITERATIONS:
                raise CircuitUnstableException(
                    f'Circuit unstable after {self.MAX_ITERATIONS} '
                    f'iterations; is there a circular dependency?'
                )

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

            impl = ElementRegistry.get_impl(name)
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

    def mock(self, replacee, replacement: MockElement):
        for i in range(len(self.elements)):
            if self.elements[i] is replacee:
                self.elements[i] = replacement

        self.nodes += replacement.nodes
        for node in replacee.nodes:
            self.nodes.remove(node)

        for field in replacement.fields:
            replacee_node = getattr(replacee, field)
            replacement_node = getattr(replacement, field)
            for neighbor in replacee_node.connections:
                neighbor.connections.remove(replacee_node)
                neighbor.connections.append(replacement_node)

            replacement_node.connections = replacee_node.connections

    def find_element(self, cls, check=None):
        for element in self.elements:
            if not isinstance(element, cls):
                continue

            if check is None or check(element):
                return element
