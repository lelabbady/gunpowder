from .batch_filter import BatchFilter
import numpy as np

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
        print(type(raw.data))
        np.save('raw_data_example_%d'%(raw.data.shape[0]),raw.data)
