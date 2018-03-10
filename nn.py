#! /usr/bin/python

from MiniFlow import * 

if __name__ == '__main__':
    x, y, z = Input(), Input(), Input()

    # f = Add(x, y, z)
    f = Multiply(x, y, z)

    feed_dict = {x: 4, y: 5, z: 10}

    graph = topological_sort(feed_dict)
    # for gf in graph:
    #     print("gf >> ",gf)

    output = forward_pass(f, graph)

    # should output 19
    # print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))
    print("{} * {} * {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))
