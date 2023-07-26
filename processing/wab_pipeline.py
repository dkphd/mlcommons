from sklearn.pipeline import Pipeline
from tracking.wab import WABArtifact, ArtifactManifestEntry

from typing import Union, Dict
from pathlib import Path

from gems.io import Pickle

import wandb


class WABPipeline(WABArtifact, Pipeline):

    def __init__(self, steps, name: str, local_path: Union[Path, str], *, memory=None, verbose=False):
        Pipeline.__init__(self, steps, memory=memory, verbose=verbose)
        WABArtifact.__init__(self, name, local_path)

    def save(self):
        Pickle.save(self.local_path / f'pipeline.pkl', self)
        self.local_path_used = True

    @staticmethod
    def load_wab(project_name, artifact_name, tag='latest'):
        artifact, download_path = WABArtifact.load_wab(project_name, artifact_name, tag)
        pipeline = Pickle.load(download_path / f'pipeline.pkl')
        pipeline.artifact = artifact
        return pipeline

    def save_wab(self, project_name, tags=('latest',), metadata: Dict = None,
                 depends_on: Union[str, ArtifactManifestEntry, None] = None):

        pipeline_config = {'pipeline_steps_config': self._get_pipeline_config()}

        if metadata is not None:
            pipeline_config.update(metadata)

        WABArtifact.save_wab(self, project_name, tags, pipeline_config, depends_on)

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