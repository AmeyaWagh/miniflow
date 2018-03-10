#! /usr/bin/python
import numpy as np

class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Nodes from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Nodes to which this Node passes values
        self.outbound_nodes = []
        # A calculated value
        self.value = None
        # Add this node as an outbound node on its inputs.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    # These will be implemented in a subclass.
    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_nodes` and
        store the result in self.value.
        """
        raise NotImplemented


class Input(Node):
    def __init__(self):
        # an Input node has no inbound nodes,
        # so no need to pass anything to the Node instantiator
        Node.__init__(self)

    # NOTE: Input node is the only node that may
    # receive its value as an argument to forward().
    #
    # All other node implementations should calculate their
    # values from the value of previous nodes, using
    # self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        if value is not None:
            self.value = value


class Add(Node):
    # You may need to change this...
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        """
        For reference, here's the old way from the last
        quiz. You'll want to write code here.
        """
        # x_value = self.inbound_nodes[0].value
        # y_value = self.inbound_nodes[1].value
        # self.value = x_value + y_value

        # print(">>",dir(self.inbound_nodes))
        value=0    
        for node in self.inbound_nodes:
            # print(node.value)
            value+=node.value
        # self.value = reduce(lambda x,y:x.value+y.value,self.inbound_nodes)
        self.value=value

class Multiply(Node):
    def __init__(self, *inputs):
        Node.__init__(self,inputs)
    
    def forward(self):
        value=1    
        for node in self.inbound_nodes:
            print(node.value)
            value*=node.value
        # self.value = reduce(lambda x,y:x.value+y.value,self.inbound_nodes)
        self.value=value


class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

        # NOTE: The weights and bias properties here are not
        # numbers, but rather references to other nodes.
        # The weight and bias values are stored within the
        # respective nodes.

    def forward(self):
        """
        Set self.value to the value of the linear function output.

        Your code goes here!
        """
        # for _input _weight in zip(iter1, iter2)
        # for node in self.inbound_nodes:
        #     print(node.value) 
        inputs = np.array(self.inbound_nodes[0].value)
        weights = np.array(self.inbound_nodes[1].value)
        bias = np.array(self.inbound_nodes[2].value)
        # self.value = 
        # print("inputs",inputs)
        # print("weights",weights)
        # print("bias",bias)
        # print("value",np.matmul(inputs,weights)+bias)
        # self.value = np.matmul(inputs,weights)+bias
        self.value = np.dot(inputs, weights) + bias

class Sigmoid(Node):
    """
    You need to fix the `_sigmoid` and `forward` methods.
    """
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used later with `backward` as well.

        `x`: A numpy array-like object.

        Return the result of the sigmoid function.

        Your code here!
        """
        # sigmoid = lambda n:(1./1.+np.exp(-1*n))
        # meta = Linear(inputs, weights, bias)
        # return map(sigmoid,x)
        return 1./(1.+np.exp(-x))

    def forward(self):
        """
        Set the value of this node to the result of the
        sigmoid function, `_sigmoid`.

        Your code here!
        """
        # This is a dummy value to prevent numpy errors
        # if you test without changing this method.
        # for node in self.inbound_nodes:
        #     print(self._sigmoid(node.value))
        self.value = self._sigmoid(self.inbound_nodes[0].value)

class MSE(Node):
    def __init__(self, y, a):
        """
        The mean squared error cost function.
        Should be used as the last node for a network.
        """
        # Call the base class' constructor.
        Node.__init__(self, [y, a])

    def forward(self):
        """
        Calculates the mean squared error.
        """
        # NOTE: We reshape these to avoid possible matrix/vector broadcast
        # errors.
        #
        # For example, if we subtract an array of shape (3,) from an array of shape
        # (3,1) we get an array of shape(3,3) as the result when we want
        # an array of shape (3,1) instead.
        #
        # Making both arrays (3,1) insures the result is (3,1) and does
        # an elementwise subtraction as expected.
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        # TODO: your code here
        # print(y)
        # print(a)
        self.diff = y-a 
        self.value = np.mean(self.diff**2)

def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

def forward_pass(graph):
    """
    Performs a forward pass through a list of sorted Nodes.

    Arguments:

        `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    for n in graph:
        n.forward()

# def forward_pass(output_node, sorted_nodes):
#     """
#     Performs a forward pass through a list of sorted nodes.

#     Arguments:

#         `output_node`: A node in the graph, should be the output node (have no outgoing edges).
#         `sorted_nodes`: A topologically sorted list of nodes.

#     Returns the output Node's value
#     """

#     for n in sorted_nodes:
#         n.forward()

#     return output_node.value
