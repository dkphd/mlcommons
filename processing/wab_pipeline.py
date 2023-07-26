from sklearn.pipeline import Pipeline
from ..tracking.wab import WABArtifact, ArtifactType

from typing import Union, Dict, Optional, Iterable
from pathlib import Path

from gems.io import Pickle


class WABPipeline(Pipeline, WABArtifact):

    def __init__(self, steps, name: str, local_path: Union[Path, str], *, memory=None, verbose=False):
        WABArtifact.__init__(self, name, local_path)
        Pipeline.__init__(self, steps, memory=memory, verbose=verbose)

    def save(self):
        Pickle.save(self.local_path / f'pipeline.pkl', self)
        self.local_path_used = True

    @staticmethod
    def load_wab(project_name, artifact_name, tag='latest', job_type='download-data_processing_pipeline'):
        artifact, download_path = WABArtifact.load_wab(project_name, artifact_name, tag, job_type=job_type)
        pipeline = Pickle.load(download_path / f'pipeline.pkl')
        pipeline.artifact = artifact
        return pipeline

    def save_wab(self, project_name, tags=('latest',), metadata: Dict = None,
                 depends_on:  Union[ArtifactType, Iterable[ArtifactType]] = None,
                 job_type='upload-data_processing_pipeline'):

        pipeline_config = {'pipeline_steps_config': self._get_pipeline_config()}

        if metadata is not None:
            pipeline_config.update(metadata)

        WABArtifact.save_wab(self, project_name, tags, pipeline_config, depends_on, job_type=job_type)

    @staticmethod
    def load(path: Union[str, Path]):
        pipeline = Pickle.load(path / f'pipeline.pkl')
        return pipeline

    def _get_pipeline_config(self):
        config = {}
        for step in self.steps:
            config[step[0]] = step[1].get_params()
        return config

    @property
    def type(self):
        return "Pipeline"
