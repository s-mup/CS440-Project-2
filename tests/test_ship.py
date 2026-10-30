import unittest

import numpy as np

from where_is_it.ship import Ship, generate_ship


class TestShipRepresentation(unittest.TestCase):
    def setUp(self):
        self.ship = Ship.from_ascii("""
            ..#
            .#.
            ...
        """)

    def test_indexing_round_trip(self):
        self.assertEqual(self.ship.n, 7)
        for i in range(self.ship.n):
            self.assertEqual(self.ship.index_of(self.ship.cell_of(i)), i)

    def test_blocked_and_out_of_bounds_cells_rejected(self):
        for cell in [(0, 2), (1, 1), (-1, 0), (3, 0)]:
            with self.assertRaises(ValueError):
                self.ship.index_of(cell)

    def test_neighbours_are_orthogonal_and_open(self):
        for i, nbrs in enumerate(self.ship.neighbors):
            r, c = self.ship.cell_of(i)
            for j in nbrs:
                nr, nc = self.ship.cell_of(j)
                self.assertEqual(abs(nr - r) + abs(nc - c), 1)
        # (0,1) touches (0,0) only: (0,2) and (1,1) are blocked.
        self.assertEqual(self.ship.neighbors[self.ship.index_of((0, 1))],
                         (self.ship.index_of((0, 0)),))

    def test_pairwise_geometry(self):
        a, b = self.ship.index_of((0, 0)), self.ship.index_of((2, 2))
        self.assertEqual(self.ship.cheb_dist[a, b], 2)
        self.assertAlmostEqual(self.ship.euclid_dist[a, b], np.sqrt(8))
        self.assertEqual(self.ship.bfs_dist[a, b], 4)

    def test_bad_ascii_rejected(self):
        with self.assertRaises(ValueError):
            Ship.from_ascii("..\n...")
        with self.assertRaises(ValueError):
            Ship.from_ascii("..x")
        with self.assertRaises(ValueError):
            Ship(np.zeros((3, 3), dtype=bool))


class TestShipGenerator(unittest.TestCase):
    def test_reproducible_with_same_seed(self):
        a = generate_ship(25, np.random.default_rng(123))
        b = generate_ship(25, np.random.default_rng(123))
        c = generate_ship(25, np.random.default_rng(124))
        np.testing.assert_array_equal(a.grid, b.grid)
        self.assertFalse(np.array_equal(a.grid, c.grid))

    def test_connected_and_reasonable_density(self):
        for seed in range(5):
            ship = generate_ship(41, np.random.default_rng(seed))
            self.assertTrue(ship.is_connected())
            frac = ship.n / 41 ** 2
            self.assertTrue(0.4 < frac < 0.9, frac)

    def test_braiding_removes_dead_ends(self):
        rng_seed = 3
        no_braid = generate_ship(31, np.random.default_rng(rng_seed), 0.0)
        braided = generate_ship(31, np.random.default_rng(rng_seed), 1.0)

        def dead_ends(ship):
            return sum(len(nb) == 1 for nb in ship.neighbors)

        self.assertLess(dead_ends(braided), dead_ends(no_braid))

    def test_dead_ends_reduced_to_at_most_half(self):
        # Team Project 1 rule: keep opening a neighbour of a random current
        # dead end until at most half of the original dead ends remain. With
        # the same seed, fraction 0 stops right after the growth phase, so it
        # gives the original dead-end count.
        def dead_ends(ship):
            return sum(len(nb) == 1 for nb in ship.neighbors)

        for seed in range(5):
            before = dead_ends(generate_ship(41, np.random.default_rng(seed), 0.0))
            after = dead_ends(generate_ship(41, np.random.default_rng(seed)))
            self.assertLessEqual(after, before // 2)

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            generate_ship(2, np.random.default_rng(0))
        with self.assertRaises(ValueError):
            generate_ship(10, np.random.default_rng(0), dead_end_fraction=1.5)


if __name__ == "__main__":
    unittest.main()
