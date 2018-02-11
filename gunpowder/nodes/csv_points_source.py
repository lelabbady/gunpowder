import numpy as np
from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.nodes.batch_provider import BatchProvider
from gunpowder.points import Point, Points
from gunpowder.points_spec import PointsSpec
from gunpowder.profiling import Timing
from gunpowder.roi import Roi

class CsvPointsSource(BatchProvider):
    '''Read a set of points from a comma-separated-values text file. Each line
    in the file represents one point.

    Args:

        filename (string): The file to read from.

        points (:class:`PointsKey`): The key of the points set to create.

        points_spec (PointsSpec, optional): An optional :class:`PointsSpec` to
            overwrite the points specs automatically determined from the CSV
            file. This is useful to set the :class:`Roi` manually.
    '''

    def __init__(self, filename, points, points_spec=None):

        self.filename = filename
        self.points = points
        self.points_spec = points_spec
        self.ndims = None
        self.data = None

    def setup(self):

        self.data = self.read_points(self.filename)
        self.ndims = self.data.shape[1]

        if self.points_spec is not None:

            self.provides(self.points, self.points_spec)
            return

        min_bb = Coordinate(np.floor(np.amin(self.data, 0)))
        max_bb = Coordinate(np.ceil(np.amax(self.data, 0)))

        roi = Roi(min_bb, max_bb - min_bb)

        self.provides(self.points, PointsSpec(roi=roi))

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        min_bb = request[self.points].roi.get_begin()
        max_bb = request[self.points].roi.get_end()

        point_filter = np.ones((self.data.shape[0],), dtype=np.bool)
        for d in range(self.ndims):
            point_filter = np.logical_and(point_filter, self.data[:,d] >= min_bb[d])
            point_filter = np.logical_and(point_filter, self.data[:,d] < max_bb[d])

        filtered = self.data[point_filter]

        points_data = {

            i: Point(Coordinate(p))
            for i, p in enumerate(filtered)
        }
        points_spec = PointsSpec(roi=request[self.points].roi.copy())

        batch = Batch()
        batch.points[self.points] = Points(points_data, points_spec)

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def read_points(self, filename):

        return np.array(
            [
                [ float(t.strip(',')) for t in line.split() ]
                for line in open(filename, 'r')
            ])
