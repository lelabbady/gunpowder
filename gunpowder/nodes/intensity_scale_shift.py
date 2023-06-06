from .batch_filter import BatchFilter
import numpy as np
import random

class IntensityScaleShift(BatchFilter):
    '''Scales the intensities of a batch by ``scale``, then adds ``shift``.

    Args:

        array (:class:`ArrayKey`):

            The key of the array to modify.

        scale (``float``):
        shift (``float``):

            The shift and scale to apply to ``array``.
    '''

    def __init__(self, array, scale, shift):
        self.array = array
        self.scale = scale
        self.shift = shift

    def process(self, batch, request):

        if self.array not in batch.arrays:
            return

        raw = batch.arrays[self.array]
        raw.data = raw.data*self.scale + self.shift
        rand_int = random.randint(0,5000)
        if rand_int % 10 == 0:
            np.save('raw_data_example_%d'%(rand_int),raw.data)
