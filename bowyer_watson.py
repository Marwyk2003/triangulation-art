from __future__ import annotations
import numpy as np
import numpy.typing as npt


def dot(a: float, b: float) -> float:
    return a.x*b.x+a.y*b.y


def cross(a: float, b: float) -> float:
    return a.x*b.y-a.y*b.x


class Point:
    def __init__(self, x: float, y: float, c=[150, 150, 150]):
        self.x = x
        self.y = y
        self.c = c

    def __repr__(self):
        return f'({self.x}, {self.y})'

    def __add__(self, p: Point):
        return Point(self.x+p.x, self.y+p.y)

    def __sub__(self, p: Point):
        return Point(self.x-p.x, self.y-p.y)

    def __mul__(self, c: float):
        return Point(self.x*c, self.y*c)

    def __truediv__(self, c: float):
        return Point(self.x/c, self.y/c)

    def __eq__(self, p: Point):
        return self.x == p.x and self.y == p.y

    def __lt__(self, p: Point):
        if self.x == p.x:
            return self.y < p.y
        return self.x < p.x

    def __hash__(self):
        return hash((self.x, self.y))


class Triangle:
    def __init__(self, v: list[Point], color=[150, 150, 150]):
        assert len(v) == 3
        self._v = v

    @property
    def color(self) -> Array[int]:
        v = self.v
        return [((v[0].c[x]**2+v[1].c[x]**2+v[2].c[x]**2)/3)**0.5/255 for x in range(3)]

    @property
    def v(self) -> list[Point]:
        return sorted(self._v)

    @property
    def circum_circle(self) -> Tuple[Point, float]:
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
    def edges(self) -> list[Edge]:
        return [Edge([self.v[0], self.v[1]]), Edge([self.v[1], self.v[2]]), Edge([self.v[2], self.v[0]])]

    def in_cc(self, p: Point) -> bool:
        u, r = self.circum_circle
        return dot(u-p, u-p) <= r**2

    def __repr__(self):
        return str(self.v)

    def __eq__(self, t):
        return self.v == t.v

    def __hash__(self):
        return hash(tuple(self.v))


class Edge:
    def __init__(self, v: list[Point]):
        assert len(v) == 2
        self.a, self.b = sorted(v)

    def __repr__(self):
        return str((self.a, self.b))

    def __eq__(self, e):
        return self.a == e.a and self.b == e.b

    def __hash__(self):
        return hash((self.a, self.b))


def BowyerWatson(points_arr: npt.ArrayLike, border: bool = True) -> list[Triangle]:
    points = list(map(lambda cords: Point(
        cords[0], cords[1], cords[2:]), points_arr))
    min_x, min_y = min(points_arr[:, 0])-1, min(points_arr[:, 1])-1
    max_x, max_y = max(points_arr[:, 0])+1, max(points_arr[:, 1])+1
    super_p = [Point(min_x, max_y), Point(max_x, max_y),
               Point(min_x, min_y), Point(max_x, min_y)]
    super_t = [Triangle(super_p[:3]), Triangle(super_p[1:])]
    triangulation = set(super_t)
    for p in points:
        bad_t = set()
        bad_e = {}  # edge hash -> no occurences
        for t in triangulation:
            if t.in_cc(p):
                bad_t.add(t)
                for e in t.edges:
                    bad_e[e] = bad_e.get(e, 0) + 1
        polygon = set()
        for t in bad_t:
            for e in t.edges:
                assert e in bad_e
                if bad_e[e] == 1:
                    polygon.add(e)
        for t in bad_t:
            triangulation.remove(t)
        for e in polygon:
            new_t = Triangle([e.a, e.b, p])
            triangulation.add(new_t)
    if not border:
        for t in set(triangulation):
            if len(set(t.v) & set(super_p)) > 0:
                triangulation.remove(t)
    return list(triangulation)


if __name__ == '__main__':
    N = 100
    MIN, MAX = -1000, 1000
    random = np.random.randint(MIN, MAX, (N, 2))
    unique = np.unique(random, axis=0)
    triangulation = BowyerWatson(unique)
    print(triangulation)
