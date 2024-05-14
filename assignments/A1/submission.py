# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

import heapq
import os
import pickle
import math


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        if self.size() == 0:
            return None
        index = 0

        value = self.queue[0][0]
        for x in range(len(self.queue)):
            if self.queue[x][0] < value:
                index = x
                value = self.queue[x][0]
        item = self.queue[index]
        del self.queue[index]
        return item

    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """

        # if self.__contains__(node):
        self.queue.remove(node)

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        self.queue.append(node)
        
    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in the queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if start == goal or start not in graph.nodes or goal not in graph.nodes:
        return []

    # Would've been nice to know sooner that we're not supposed to reset the graph -_-
    # graph.reset_search()

    # where we've looked
    explored = []
    # where we're looking everywhere
    frontier = PriorityQueue()
    frontier.append(start)
    while frontier.size() > 0:
        # where we're looking now
        path = frontier.pop()
        # last node in the path
        node = path[-1]
        if node not in explored:
            # neighbors sorted alphabetically
            neighbors = sorted(graph.neighbors(node))
            for neighbor in neighbors:
                final_path = list(path)
                final_path.append(neighbor)
                frontier.append(final_path)
                if neighbor == goal:
                    return final_path
            explored.append(node)
    # not found
    return []


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if start == goal or start not in graph.nodes or goal not in graph.nodes:
        return []

    # Would've been nice to know sooner that we're not supposed to reset the graph -_-
    # graph.reset_search()

    node = (0, start, None)
    frontier = PriorityQueue()
    frontier.append(node)
    explored = set()

    while frontier.size() > 0:
        node = frontier.pop()
        if node[1] == goal:
            path_list = [node[1]]
            temp_node = node[2]
            while temp_node:
                path_list.append(temp_node[1])
                temp_node = temp_node[2]
            path_list.reverse()
            return path_list
        explored.add(node[1])
        children = graph[node[1]]
        for child in children:
            if child not in explored:
                h = node[0] + children[child]['weight']
                child_node = (h, child, node)
                frontier.append(child_node)
    # not found
    return []


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    if v == goal:
        return 0

    if v not in graph or goal not in graph:
        return -1

    keys = graph.nodes()[v].keys()

    if 'pos' in keys:
        key = 'pos'
    elif 'position' in keys:
        key = 'position'
    else:
        return 0

    node_1 = graph.nodes()[v][key]
    node_2 = graph.nodes()[goal][key]
    dx = node_2[0] - node_1[0]
    dy = node_2[1] - node_1[1]
    dx = math.pow(dx, 2)
    dy = math.pow(dy, 2)
    return int(math.sqrt(dx + dy))


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if start == goal or start not in graph.nodes or goal not in graph.nodes:
        return []

    # Would've been nice to know sooner that we're not supposed to reset the graph -_-
    # graph.reset_search()

    node = (heuristic(graph, start, goal), 0, start, None)

    frontier = PriorityQueue()
    explored = set()

    frontier.append(node)

    while frontier.size() > 0:
        node = frontier.pop()
        if node[2] == goal:
            path_list = [node[2]]
            temp_node = node[3]
            while temp_node:
                path_list.append(temp_node[2])
                temp_node = temp_node[3]
            path_list.reverse()
            return path_list

        children = graph[node[2]]
        explored.add(node[2])

        for child in children:
            if child not in explored:
                f = (node[1] + children[child]['weight']) + heuristic(graph, child, goal)
                child_node = (f, (node[1] + children[child]['weight']), child, node)
                frontier.append(child_node)
    # not found
    return []


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if start == goal or start not in graph.nodes or goal not in graph.nodes:
        return []

    # Would've been nice to know sooner that we're not supposed to reset the graph -_-
    # graph.reset_search()

    explored_start = {}
    explored_goal = {}
    node_cost_start = {}
    node_cost_goal = {}
    best_path = []
    best_met = []

    frontier_start = PriorityQueue()
    frontier_goal = PriorityQueue()

    frontier_start.append((0, start))
    frontier_goal.append((0, goal))

    node_cost_start[start] = (0, None)
    node_cost_goal[goal] = (0, None)

    while frontier_start.size() > 0 and frontier_goal.size() > 0:
        current_cost, current_node = frontier_start.pop()
        if current_node in explored_goal:
            while frontier_goal.size() > 0:
                node = frontier_goal.pop()
                temp_node = node[1]
                if temp_node not in explored_goal:
                    explored_goal[temp_node] = node_cost_goal[temp_node]
            explored_start[current_node] = node_cost_start.pop(current_node)
            best_met = (current_node, current_cost + explored_goal[current_node][0])
            for key in explored_start:
                if key in explored_goal:
                    my_cost = explored_start[key][0] + explored_goal[key][0]
                    if my_cost < best_met[1]:
                        best_met = (key, my_cost)
            break
        elif current_node not in explored_start:
            children = graph[current_node]
            explored_start[current_node] = node_cost_start.pop(current_node)
            parent_cost = explored_start[current_node][0]
            for child in children:
                temp_cost = children[child]['weight'] + parent_cost
                if child in node_cost_start and temp_cost < node_cost_start[child][0]:
                    del node_cost_start[child]
                if child not in explored_start and child not in node_cost_start:
                    node_cost_start[child] = (temp_cost, current_node)
                    frontier_start.append((temp_cost, child))

        current_cost, current_node = frontier_goal.pop()

        if current_node in explored_start:
            while frontier_start.size() > 0:
                node = frontier_start.pop()
                temp_node = node[1]
                if temp_node not in explored_start:
                    explored_start[temp_node] = node_cost_start[temp_node]
            explored_goal[current_node] = node_cost_goal.pop(current_node)
            best_met = (current_node, current_cost + explored_start[current_node][0])
            for key in explored_goal:
                if key in explored_start:
                    my_cost = explored_goal[key][0] + explored_start[key][0]
                    if my_cost < best_met[1]:
                        best_met = (key, my_cost)
            break
        elif current_node not in explored_goal:
            children = graph[current_node]
            explored_goal[current_node] = node_cost_goal.pop(current_node)

            parent_cost = explored_goal[current_node][0]
            for child in children:
                temp_cost = children[child]['weight'] + parent_cost
                if child in node_cost_goal and temp_cost < node_cost_goal[child][0]:
                    del node_cost_goal[child]
                if child not in explored_goal and child not in node_cost_goal:
                    node_cost_goal[child] = (temp_cost, current_node)
                    frontier_goal.append((temp_cost, child))

    parent_node = best_met[0]

    while parent_node:
        best_path.insert(0, parent_node)
        parent_node = explored_start[parent_node][1]

    parent_node = explored_goal[best_met[0]][1]

    while parent_node:
        best_path.append(parent_node)
        parent_node = explored_goal[parent_node][1]

    return best_path


def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if start == goal or start not in graph.nodes or goal not in graph.nodes:
        return []

    # Would've been nice to know sooner that we're not supposed to reset the graph -_-
    # graph.reset_search()

    explored_start = {}
    explored_goal = {}
    node_cost_start = {}
    node_cost_goal = {}
    best_path = []

    frontier_start = PriorityQueue()
    frontier_goal = PriorityQueue()

    frontier_start.append((0, start))
    frontier_goal.append((0, goal))

    node_cost_start[start] = (0, None)
    node_cost_goal[goal] = (0, None)

    while frontier_start.size() > 0 and frontier_goal.size() > 0:
        current_cost, current_node = frontier_start.pop()
        if current_node in explored_goal:
            while frontier_goal.size() > 0:
                node = frontier_goal.pop()
                temp_node = node[1]
                if temp_node not in explored_goal:
                    explored_goal[temp_node] = node_cost_goal[temp_node]
            explored_start[current_node] = node_cost_start.pop(current_node)
            best_met = (current_node, current_cost + explored_goal[current_node][0])
            for key in explored_start:
                if key in explored_goal:
                    my_cost = explored_start[key][0] + explored_goal[key][0]
                    if my_cost < best_met[1]:
                        best_met = (key, my_cost)
            break
        elif current_node not in explored_start:
            children = graph[current_node]
            explored_start[current_node] = node_cost_start.pop(current_node)
            parent_cost = explored_start[current_node][0]
            for child in children:
                g = children[child]['weight'] + parent_cost
                h = heuristic(graph, child, goal)
                if child in node_cost_start and g < node_cost_start[child][0]:
                    del node_cost_start[child]
                if child not in node_cost_start and child not in explored_start:
                    node_cost_start[child] = (g, current_node)
                    frontier_start.append((g + h, child))

        current_cost, current_node = frontier_goal.pop()

        if current_node in explored_start:
            while frontier_start.size() > 0:
                node = frontier_start.pop()
                temp_node = node[1]
                if temp_node not in explored_start:
                    explored_start[temp_node] = node_cost_start[temp_node]
            explored_goal[current_node] = node_cost_goal.pop(current_node)
            best_met = (current_node, current_cost + explored_start[current_node][0])
            for key in explored_goal:
                if key in explored_start:
                    my_cost = explored_goal[key][0] + explored_start[key][0]
                    if my_cost < best_met[1]:
                        best_met = (key, my_cost)
            break
        elif current_node not in explored_goal:
            children = graph[current_node]
            explored_goal[current_node] = node_cost_goal.pop(current_node)
            parent_cost = explored_goal[current_node][0]
            for child in children:
                g = children[child]['weight'] + parent_cost
                h = heuristic(graph, child, start)
                if child in node_cost_goal and g < node_cost_goal[child][0]:
                    del node_cost_goal[child]
                if child not in node_cost_goal and child not in explored_goal:
                    node_cost_goal[child] = (g, current_node)
                    frontier_goal.append((g + h, child))

    parent_node = best_met[0]

    while parent_node:
        best_path.insert(0, parent_node)
        parent_node = explored_start[parent_node][1]

    parent_node = explored_goal[best_met[0]][1]

    while parent_node:
        best_path.append(parent_node)
        parent_node = explored_goal[parent_node][1]

    return best_path


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    if goals[0] == goals[1] and goals[1] == goals[2]:
        return []
    """
    Expressly forbidden, but otherwise a good idea:
    if goals[0] == goals[1]:
        return bidirectional_ucs(graph, goals[0], goals[1])
    if goals[1] == goals[2]:
        return bidirectional_ucs(graph, goals[1], goals[2])
    if goals[0] == goals[2]:
        return bidirectional_ucs(graph, goals[0], goals[2])
    """

    # Would've been nice to know sooner that we're not supposed to reset the graph -_-
    # graph.reset_search()

    frontier = []
    explored = []
    node_data = []
    path = []
    count = 0

    for goal in goals:
        if goal not in graph.nodes:
            return []
        frontier.append(PriorityQueue())
        explored.append({})
        node_data.append({})
        if count == 0:
            count += 1
        else:
            path.append([])

    # Couldn't figure this out
    return []


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic, landmarks=None):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
        landmarks: Iterable containing landmarks pre-computed in compute_landmarks()
            Default: None

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    if goals[0] == goals[1] and goals[1] == goals[2]:
        return []

    # Would've been nice to know sooner that we're not supposed to reset the graph -_-
    # graph.reset_search()

    frontier = []
    explored = []
    node_data = []
    path = []
    count = 0
    goal_sorted = []

    for goal in goals:
        if goal not in graph.nodes:
            return []
        frontier.append(PriorityQueue())
        explored.append({})
        node_data.append({})
        goal_sorted.append(None)
        if count == 0:
            count += 1
        else:
            path.append([])

    # Couldn't figure this out
    return []


def return_your_name():
    """Return your name from this function"""
    return "David Strube"


def compute_landmarks(graph):
    """
    Feel free to implement this method for computing landmarks. We will call
    tridirectional_upgraded() with the object returned from this function.

    Args:
        graph (ExplorableGraph): Undirected graph to search.

    Returns:
    List with not more than 4 computed landmarks. 
    """
    return None


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """
    pass


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to Gradescope, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    return breadth_first_search(graph, start, goal)


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None
 
 
def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    # Load latitude and longitude coordinates in radians:
    v_lat_long = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goal_lat_long = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    # Now we want to execute portions of the formula:
    # Radius of Earth is 6,371 kilometers
    const_out_front = 2*6371
    # First term inside sqrt
    term1_in_sqrt = (math.sin((goal_lat_long[0]-v_lat_long[0])/2))**2
    # Second term
    term2_in_sqrt = math.cos(v_lat_long[0]) * math.cos(goal_lat_long[0]) * \
                    ((math.sin((goal_lat_long[1]-v_lat_long[1])/2))**2)
    # Straight application of formula
    return const_out_front*math.asin(math.sqrt(term1_in_sqrt+term2_in_sqrt))
