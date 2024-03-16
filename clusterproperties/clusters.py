from typing import Any

class ClusterObject:
    def __init__(self, cluster_id, data_points):
        self.cluster_id = cluster_id
        self.data_points = data_points
        self._convex_hull_area = None  # Use private variable to store calculated value


        def get_number_of_data_points(self):
            return len(self.data_points)
        
        def convex_hull_area(self):
            if self._convex_hull_area is None:
                self._convex_hull_area = calculate_convex_hull_area(self.data_points)
            return self._convex_hull_area

class DBSCANClusterProperties(ClusterObject):
    def __init__(self) -> None:
        super().__init__(cluster_id, data_points)