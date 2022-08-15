import numpy as np
import matplotlib.pyplot as plt
from node_detection import NodeDetection
from bowyer_watson import BowyerWatson


def plot(points, triangulation):
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    ax.set_aspect('equal')
    for i, t in enumerate(triangulation):
        cords = np.array([[t.v[j].y, t.v[j].x] for j in range(3)])
        t1 = plt.Polygon(cords, ec=t.color, fc=t.color)  # ec='black',
        ax.add_patch(t1)
    plt.scatter(points[:, 1], points[:, 0], s=0)
    plt.show()


if __name__ == '__main__':
    nodeDetection = NodeDetection("wave.jpg", N=100, S=5, B=10, M=10)
    nodes = nodeDetection.get_nodes()
    triangulation = BowyerWatson(nodes)
    plot(nodes, triangulation)
