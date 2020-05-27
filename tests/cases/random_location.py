from .provider_test import ProviderTest
from gunpowder import (
    ArrayKeys,
    ArrayKey,
    ArraySpec,
    Array,
    Roi,
    Coordinate,
    Batch,
    BatchRequest,
    BatchProvider,
    RandomLocation,
    MergeProvider,
    build,
)
import numpy as np


class TestSourceRandomLocation(BatchProvider):
    def __init__(self, array):
        self.array = array
        self.roi = Roi((-200, -20, -20), (1000, 100, 100))
        self.data_shape = (60, 60, 60)
        self.voxel_size = (20, 2, 2)
        x = np.linspace(-10, 49, 60).reshape((-1, 1, 1))
        self.data = x + x.transpose([1, 2, 0]) + x.transpose([2, 0, 1])

    def setup(self):
        self.provides(self.array, ArraySpec(roi=self.roi, voxel_size=self.voxel_size))

    def provide(self, request):

        batch = Batch()

        spec = request[ArrayKeys.RAW].copy()
        spec.voxel_size = Coordinate((20, 2, 2))

        start = (request[ArrayKeys.RAW].roi.get_begin() / self.voxel_size) + (
            10,
            10,
            10,
        )
        end = (request[ArrayKeys.RAW].roi.get_end() / self.voxel_size) + (10, 10, 10)
        data_slices = tuple(map(slice, start, end))

        data = self.data[data_slices]

        batch.arrays[self.array] = Array(data=data, spec=spec)

        return batch


class CustomRandomLocation(RandomLocation):
    def __init__(self, array, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.array = array

    # only accept random locations that contain (0, 0, 0)
    def accepts(self, request):
        return request.array_specs[self.array].roi.contains((0, 0, 0))


class TestRandomLocation(ProviderTest):
    def test_output(self):

        raw = ArrayKey("RAW")
        source = TestSourceRandomLocation(raw)
        pipeline = source + CustomRandomLocation(raw)

        with build(pipeline):

            for i in range(10):
                batch = pipeline.request_batch(
                    BatchRequest({raw: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20)))})
                )

                self.assertTrue(0 in batch.arrays[raw].data)

                # Request a ROI with the same shape as the entire ROI
                full_roi = Roi((0, 0, 0), source.roi.get_shape())
                batch = pipeline.request_batch(
                    BatchRequest({raw: ArraySpec(roi=full_roi)})
                )

    def test_random_seed(self):
        raw = ArrayKey("RAW")
        pipeline = TestSourceRandomLocation(raw) + CustomRandomLocation(raw)

        with build(pipeline):
            seeded_sums = []
            unseeded_sums = []
            for i in range(10):
                batch_seeded = pipeline.request_batch(
                    BatchRequest(
                        {raw: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20)))},
                        random_seed=10,
                    )
                )
                seeded_sums.append(batch_seeded[raw].data.sum())
                batch_unseeded = pipeline.request_batch(
                    BatchRequest({raw: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20)))})
                )
                unseeded_sums.append(batch_unseeded[raw].data.sum())

            self.assertEqual(len(set(seeded_sums)), 1)
            self.assertGreater(len(set(unseeded_sums)), 1)

    def test_impossible(self):
        a = ArrayKey("A")
        b = ArrayKey("B")
        null_key = ArrayKey("NULL")
        source_a = TestSourceRandomLocation(a)
        source_b = TestSourceRandomLocation(b)

        pipeline = (
            (source_a, source_b) + MergeProvider() + CustomRandomLocation(null_key)
        )

        with build(pipeline):
            with self.assertRaises(AssertionError):
                batch = pipeline.request_batch(
                    BatchRequest(
                        {
                            a: ArraySpec(roi=Roi((0, 0, 0), (200, 20, 20))),
                            b: ArraySpec(roi=Roi((1000, 100, 100), (220, 22, 22))),
                        }
                    )
                )
