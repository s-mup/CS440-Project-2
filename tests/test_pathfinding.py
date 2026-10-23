import unittest

import numpy as np

from where_is_it.pathfinding import (UnreachableError, all_pairs_bfs,
                                     bfs_distances, bfs_path, shortest_path)
from where_is_it.ship import Ship, generate_ship

# The wall forces a detour: (0,0) -> (2,0) is 10 moves, not 2.
DETOUR = Ship.from_ascii("""
    .....
    ####.
    .....
""")


class TestBFS(unittest.TestCase):
    def test_blocked_cells_force_detour(self):
        a, b = DETOUR.index_of((0, 0)), DETOUR.index_of((2, 0))
        self.assertEqual(bfs_distances(DETOUR.neighbors, a)[b], 10)
        self.assertEqual(DETOUR.bfs_dist[a, b], 10)

    def test_shortest_path_is_valid_and_optimal(self):
        a, b = DETOUR.index_of((0, 0)), DETOUR.index_of((2, 0))
        path = shortest_path(DETOUR.neighbors, DETOUR.bfs_dist, a, b)
        self.assertEqual(len(path), 10)
        self.assertEqual(path[-1], b)
        prev = a
        for step in path:  # every step is a legal move to an open neighbour
            self.assertIn(step, DETOUR.neighbors[prev])
            prev = step
        self.assertEqual(path, bfs_path(DETOUR.neighbors, a, b))

    def test_path_to_self_is_empty(self):
        self.assertEqual(shortest_path(DETOUR.neighbors, DETOUR.bfs_dist, 3, 3), [])

    def test_unreachable(self):
        ship = Ship.from_ascii("..#..")
        a, b = ship.index_of((0, 0)), ship.index_of((0, 4))
        self.assertEqual(ship.bfs_dist[a, b], -1)
        self.assertIsNone(bfs_path(ship.neighbors, a, b))
        with self.assertRaises(UnreachableError):
            shortest_path(ship.neighbors, ship.bfs_dist, a, b)
        self.assertFalse(ship.is_connected())

    def test_all_pairs_on_generated_ship(self):
        ship = generate_ship(15, np.random.default_rng(0))
        d = all_pairs_bfs(ship.neighbors)
        np.testing.assert_array_equal(d, d.T)
        manhattan = np.abs(ship.cells[:, None] - ship.cells[None]).sum(axis=2)
        self.assertTrue((d >= manhattan).all())  # walls only lengthen paths
        rng = np.random.default_rng(1)
        for _ in range(50):
            s, g = rng.integers(ship.n, size=2)
            path = shortest_path(ship.neighbors, d, s, g)
            self.assertEqual(len(path), d[s, g])
            self.assertEqual(len(bfs_path(ship.neighbors, s, g)), d[s, g])


if __name__ == "__main__":
    unittest.main()
