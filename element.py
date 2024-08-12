from functools import singledispatchmethod
from typing import List


class Element:
    @singledispatchmethod
    def __init__(
        self,
        nodes: List,
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
    def load(self, raw_element: dict, nodes: List, context: dict):
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
