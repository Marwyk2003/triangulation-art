import numpy as np
import matplotlib.pyplot as plt


def dot(a, b):
    return a.x*b.x+a.y*b.y


def cross(a, b):
    return a.x*b.y-a.y*b.x


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, p):
        return Point(self.x+p.x, self.y+p.y)

    def __sub__(self, p):
        return Point(self.x-p.x, self.y-p.y)

    def __mul__(self, c):
        return Point(self.x*c, self.y*c)

    def __truediv__(self, c):
        return Point(self.x/c, self.y/c)

    def __repr__(self):
        return f'({self.x}, {self.y})'

    def __eq__(self, p):
        return self.x == p.x and self.y == p.y

    def __lt__(self, p):
        if self.x == p.x:
            return self.y < p.y
        return self.x < p.x

    def __hash__(self):
        return hash((self.x, self.y))


class Triangle:
    def __init__(self, v):
        assert len(v) == 3
        self._v = v

    @property
    def v(self):
        return sorted(list(self._v))

    @property
    def circum_circle(self):
        # A = (0, 0)
        V = self.v
        B = V[1]-V[0]
        C = V[2]-V[0]
        D = 2*cross(B, C)
        Ux = (C.y*dot(B, B)-B.y*dot(C, C))
        Uy = (B.x*dot(C, C)-C.x*dot(B, B))
        U = Point(Ux, Uy)/D
        r = dot(U, U)**0.5
        return U+V[0], r

    @property
    def edges(self):
        return (Edge([self.v[0], self.v[1]]), Edge([self.v[1], self.v[2]]), Edge([self.v[2], self.v[0]]))

    def in_cc(self, p):
        u, r = self.circum_circle
        return dot(u-p, u-p) <= r**2

    def __repr__(self):
        return str(self.v)

    def __eq__(self, t):
        return self.v == t.v

    def __hash__(self):
        return hash(tuple(self.v))


class Edge:
    def __init__(self, v):
        assert len(v) == 2
        self.a, self.b = sorted(v)

    def __eq__(self, e):
        return self.a == e.a and self.b == e.b

    def __hash__(self):
        return hash((self.a, self.b))

    def __repr__(self):
        return str((self.a, self.b))


def BowyerWatson(points):
    super_t = Triangle(
        {Point(-1000, -1000), Point(1000, -1000), Point(0, 1000)})
    triangulation = [super_t]
    history = []
    history += [triangulation.copy()]
    for p in points:
        bad_t = set()
        for t in triangulation:
            if t.in_cc(p):
                bad_t.add(t)
        polygon = set()
        for t in bad_t:
            for e in t.edges:
                shared = False
                for t1 in bad_t:
                    if t == t1:
                        continue
                    for e1 in t1.edges:
                        if e == e1:
                            shared = True
                            break
                if not shared:
                    polygon.add(e)
        for t in bad_t:
            triangulation.remove(t)
        for e in polygon:
            new_t = Triangle({e.a, e.b, p})
            triangulation += [new_t]
        history += [triangulation.copy()]
    for t in set(triangulation):
        if len(set(t.v) & set(super_t.v)) > 0:
            triangulation.remove(t)
    return triangulation, history


def plot(points, triangulation):
    X = np.array(list(map(lambda p: [p.x, p.y], points)))
    fig, ax = plt.subplots()
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])

    Xt = []
    colors = np.random.rand(len(triangulation), 3)
    for i, t in enumerate(triangulation):
        cords = np.array([[t.v[j].x, t.v[j].y] for j in range(3)])
        t1 = plt.Polygon(cords, ec='black', fc=colors[i])
        ax.add_patch(t1)
    plt.scatter(X[:, 0], X[:, 1], s=25)
    plt.show()


if __name__ == '__main__':
    N = 100
    random = np.random.randint(-100, 10, (N, 2))
    unique = np.unique(random, axis=0)
    points = list(map(lambda cords: Point(*cords), unique)
                  )
    triangulation, history = BowyerWatson(points)
    plot(points, triangulation)
