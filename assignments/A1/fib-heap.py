import math


# from https://github.com/danielborowski/fibonacci-heap-python/blob/master/fib-heap.py
# TODO: Test

# iterate through a doubly linked list
def iterate(head):
    node = stop = head
    flag = False
    while True:
        if node == stop and flag is True:
            break
        elif node == stop:
            flag = True
        # `yield` is weird, but useful:
        # https://www.simplilearn.com/tutorials/python-tutorial/yield-in-python
        yield node
        node = node.right


# merge a node with the doubly linked child list of a root node
def merge_with_child_list(parent, node):
    if parent.child is None:
        parent.child = node
    else:
        node.right = parent.child.right
        node.left = parent.child
        parent.child.right.left = node
        parent.child.right = node


# remove a node from the doubly linked child list
def remove_from_child_list(parent, node):
    if parent.child == parent.child.right:
        parent.child = None
    elif parent.child == node:
        parent.child = node.right
        node.right.parent = parent
    node.left.right = node.right
    node.right.left = node.left


class FibonacciHeap:
    # internal node class
    class Node:
        # Only key & value are required parameters for the constructor
        def __init__(self, key, value, parent=None, child=None, left=None, right=None, degree=0, mark=False):
            self.key = key
            self.value = value
            self.parent = parent
            self.child = child
            self.left = left
            self.right = right
            self.degree = degree
            self.mark = mark

    # pointers to the head node and minimum node in the root list
    root_node, min_node = None, None

    # maintain total node count in full fibonacci heap
    total_nodes = 0

    # return min node in O(1) time
    def find_min(self):
        return self.min_node

    # extract (delete) the min node from the heap in O(log n) time
    # amortized cost analysis can be found here (http://bit.ly/1ow1Clm)
    def extract_min(self):
        z = self.min_node
        if z is not None:
            if z.child is not None:
                # attach child nodes to root list
                children = [x for x in iterate(z.child)]
                for i in range(0, len(children)):
                    self.merge_with_root_node(children[i])
                    children[i].parent = None
            self.remove_from_root_node(z)
            # set new min node in heap
            if z == z.right:
                self.min_node = self.root_node = None
            else:
                self.min_node = z.right
                self.consolidate()
            self.total_nodes -= 1
        return z

    # insert new node into the unordered root list in O(1) time
    # returns the node so that it can be used for decrease_key later
    def insert(self, key, value=None):
        n = self.Node(key, value)
        n.left = n.right = n
        self.merge_with_root_node(n)
        if self.min_node is None or n.key < self.min_node.key:
            self.min_node = n
        self.total_nodes += 1
        return n

    # modify the key of some node in the heap in O(1) time
    def decrease_key(self, x, k):
        if k > x.key:
            return None
        x.key = k
        y = x.parent
        if y is not None and x.key < y.key:
            self.cut(x, y)
            self.cascading_cut(y)
        if x.key < self.min_node.key:
            self.min_node = x

    # merge two fibonacci heaps in O(1) time by concatenating the root lists
    # the root of the new root list becomes equal to the first list and the second
    # list is simply appended to the end (then the proper min node is determined)
    def merge(self, new_heap):
        my_heap = FibonacciHeap()
        my_heap.root_node, my_heap.min_node = self.root_node, self.min_node
        # fix pointers when merging the two heaps
        last = new_heap.root_node.left
        new_heap.root_node.left = my_heap.root_node.left
        my_heap.root_node.left.right = new_heap.root_node
        my_heap.root_node.left = last
        my_heap.root_node.left.right = my_heap.root_node
        # update min node if needed
        if new_heap.min_node.key < my_heap.min_node.key:
            my_heap.min_node = new_heap.min_node
        # update total nodes
        my_heap.total_nodes = self.total_nodes + new_heap.total_nodes
        return my_heap

    # if a child node becomes smaller than its parent node we
    # cut this child node off and bring it up to the root list
    def cut(self, x, y):
        remove_from_child_list(y, x)
        y.degree -= 1
        self.merge_with_root_node(x)
        x.parent = None
        x.mark = False

    # cascading cut of parent node to obtain good time bounds
    def cascading_cut(self, y):
        z = y.parent
        if z is not None:
            if y.mark is False:
                y.mark = True
            else:
                self.cut(y, z)
                self.cascading_cut(z)

    # combine root nodes of equal degree to consolidate the heap
    # by creating a list of unordered binomial trees
    def consolidate(self):
        array = [None] * int(math.log(self.total_nodes) * 2)
        nodes = [w for w in iterate(self.root_node)]
        for w in range(0, len(nodes)):
            x = nodes[w]
            degree = x.degree
            while array[degree] is not None:
                y = array[degree]
                if x.key > y.key:
                    temp = x
                    x, y = y, temp
                self.heap_link(y, x)
                array[degree] = None
                degree += 1
            array[degree] = x
        # find new min node - no need to reconstruct new root list below
        # because root list was iteratively changing as we were moving
        # nodes around in the above loop
        for index in range(0, len(array)):
            if array[index] is not None:
                if array[index].key < self.min_node.key:
                    self.min_node = array[index]

    # actual linking of one node to another in the root list
    # while also updating the child linked list
    def heap_link(self, y, x):
        self.remove_from_root_node(y)
        y.left = y.right = y
        merge_with_child_list(x, y)
        x.degree += 1
        y.parent = x
        y.mark = False

    # merge a node with the doubly linked root list
    def merge_with_root_node(self, node):
        if self.root_node is None:
            self.root_node = node
        else:
            node.right = self.root_node.right
            node.left = self.root_node
            self.root_node.right.left = node
            self.root_node.right = node

    # remove a node from the doubly linked root list
    def remove_from_root_node(self, node):
        if node == self.root_node:
            self.root_node = node.right
        node.left.right = node.right
        node.right.left = node.left
