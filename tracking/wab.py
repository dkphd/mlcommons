from abc import abstractmethod, ABC
from data.local_artifact import LocalArtifact
import wandb
from pathlib import Path
from typing import Union, Dict
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry


class WABArtifact(LocalArtifact, ABC):

    def __init__(self, name: str, local_path: Union[Path, str]):
        LocalArtifact.__init__(self, local_path=local_path)
        self.name: str = name
        self.artifact = None
        self.tag = None

    @abstractmethod
    def save_wab(self, project_name, tags=('latest',), metadata: Dict = None,
                 depends_on: Union[str, ArtifactManifestEntry, None] = None):

        if not self.local_path_used:
            print("Warning: local path was not used to load or save artifact locally")

        if metadata is None:
            metadata = {}

        wab_save_run = wandb.init(
            project=project_name,
            job_type='upload-artifact'
        )

        self.artifact = wandb.Artifact(
            name=self.name,
            type=self.type,
            metadata=metadata
        )

        self.artifact.add_dir(str(self.local_path))

        if depends_on is not None:
            if not isinstance(depends_on, (str, wandb.Artifact, ArtifactManifestEntry)):
                raise ValueError("depends_on object has to be wandb ArtifactManifestEntry")
            else:
                wab_save_run.use_artifact(depends_on)

        wab_save_run.log_artifact(self.artifact, aliases=tags)
        self.artifact.wait()
        wab_save_run.finish()

        return self.artifact

    @staticmethod
    @abstractmethod
    def load_wab(project_name, artifact_name, tag='latest'):
        run = wandb.init(
            project=project_name,
            job_type='download-artifact'
        )

        artifact = run.use_artifact(f'{project_name}/{artifact_name}:{tag}')
        download_path = Path(artifact.download())

        run.finish()

        return artifact, download_path

    @property
    @abstractmethod
    def type(self):
        return ""
