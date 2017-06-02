import logging
import os

from .batch_filter import BatchFilter
from gunpowder.ext import h5py
from gunpowder.volume import VolumeType

logger = logging.getLogger(__name__)

class Snapshot(BatchFilter):
    '''Save a passing batch in an HDF file.'''

    def __init__(self, output_dir='snapshots', output_filename='{id}.hdf', every=1):
        '''
        output_dir: string

            The directory to save the snapshots. Will be created, if it does not exist.

        output_filename: string

            Template for output filenames. '{id}' in the string will be replaced 
            with the ID of the batch.

        every:

            How often to save a batch. 'every=1' indicates that every batch will 
            be stored, 'every=2' every second and so on. By default, every batch 
            will be stored.
        '''
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.every = max(1,every)

    def process(self, batch, request):

        id = batch.id

        if id%self.every == 0:

            try:
                os.makedirs(self.output_dir)
            except:
                pass

            snapshot_name = os.path.join(self.output_dir, self.output_filename.format(id=str(id).zfill(8)))
            logger.info("saving to " + snapshot_name)
            with h5py.File(snapshot_name, 'w') as f:

                for (volume_type, volume) in batch.volumes.items():

                    ds_name = {
                            VolumeType.RAW: 'volumes/raw',
                            VolumeType.ALPHA_MASK: 'volumes/alpha_mask',
                            VolumeType.GT_LABELS: 'volumes/labels/neuron_ids',
                            VolumeType.GT_AFFINITIES: 'volumes/labels/affs',
                            VolumeType.GT_MASK: 'volumes/labels/mask',
                            VolumeType.GT_IGNORE: 'volumes/labels/ignore',
                            VolumeType.PRED_AFFINITIES: 'volumes/predicted_affs',
                    }[volume_type]

                    offset = volume.roi.get_offset()
                    offset*= volume.resolution
                    dataset = f.create_dataset(name=ds_name, data=volume.data)
                    dataset.attrs['offset'] = offset
                    dataset.attrs['resolution'] = volume.resolution

                if batch.loss is not None:
                    f['/'].attrs['loss'] = batch.loss
