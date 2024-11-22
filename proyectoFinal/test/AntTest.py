import unittest

from Map import *
from Ant import *


class MyTestCase(unittest.TestCase):
    def test__init__(self):
        a = Map( np.array([[0,1],[2,0]]), 0.5, 1, 1,0.8,0.7,0.2)
        ant = Ant(1, a, 1)
        self.assertEqual(ant.id, 1)
        self.assertEqual(ant.map, a)
        self.assertEqual(ant.current_distance, 0)
        self.assertEqual(ant.current_city, 1)
        self.assertEqual(ant.tour, [1])

    def test_update_tour_dist(self):
        a = Map( np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]]), 0.5, 1, 1,0.8,0.7,0.2)
        ant = Ant(1, a, 1)
        ant2 = Ant(2, a, 2)
        ant.update_tour_dist(0)
        self.assertEqual(ant.current_distance, 1)
        self.assertEqual(ant.tour, [1,0])
        self.assertEqual(ant.current_city, 0)
        ant2.update_tour_dist(1)
        self.assertEqual(ant2.current_distance, 3)
        self.assertEqual(ant2.tour, [2,1])
        self.assertEqual(ant2.current_city, 1)

    def test_choice_city(self):
        a = Map( np.array([[0,1],[2,0]]), 0.5, 1, 1,0.8,0.7,0.2)
        ant = Ant(1, a, 1)
        ant2 = Ant(2, a, 0)
        self.assertEqual(ant.choice_city(),0)
        self.assertEqual(ant2.choice_city(),1)
        ant.update_tour_dist(0)
        ant2.update_tour_dist(1)
        self.assertEqual(ant.choice_city(),1)
        self.assertEqual(ant2.choice_city(),0)




if __name__ == '__main__':
    unittest.main()
