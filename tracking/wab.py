from abc import abstractmethod, ABC
from ..data.local_artifact import LocalArtifact
import wandb
from pathlib import Path
from typing import Union, Dict, Iterable

from wandb.sdk.artifacts.public_artifact import Artifact as PublicWandbArtifact
from wandb.sdk.artifacts.local_artifact import Artifact as LocalWandbArtifact

ArtifactType = Union[PublicWandbArtifact, LocalWandbArtifact]

class WABArtifact(LocalArtifact, ABC):

    def __init__(self, name: str, local_path: Union[Path, str]):
        LocalArtifact.__init__(self, local_path=local_path)
        self.name: str = name
        self.artifact = None
        self.tag = None

    @abstractmethod
    def save_wab(self, project_name, tags=('latest',), metadata: Dict = None,
                 depends_on: Union[ArtifactType, Iterable[ArtifactType]] = None, job_type="upload-artifact"):


        if self.name == "LOADED_FROM_LOCAL_STORAGE":
            raise ValueError("Name is automatic (LOADED_FROM_LOCAL_STORAGE), "
                             "which means that the data was loaded from local storage and"
                             " the name of the artifact is not known, set the name first")

        if depends_on is not None:
            depends_on = tuple(depends_on)
            for depend in depends_on:
                if not isinstance(depend, (PublicWandbArtifact, LocalWandbArtifact)):
                    raise ValueError(f"each dependant object has to be wandb Artifact but it is {type(depend)}")

        if not self.local_path_used:
            print("Warning: local path was not used to load or save artifact locally")

        if metadata is None:
            metadata = {}

        wab_save_run = wandb.init(
            project=project_name,
            job_type=job_type
        )

        self.artifact = wandb.Artifact(
            name=self.name,
            type=self.type,
            metadata=metadata
        )

        self.artifact.add_dir(str(self.local_path))

        if depends_on is not None:
            for depend in depends_on:
                wab_save_run.use_artifact(depend)

        wab_save_run.log_artifact(self.artifact, aliases=tags)
        print("Waiting for artifact to finish uploading")
        wab_save_run.log({'status':"Waiting for artifact to finish uploading"})
        self.artifact.wait()
        wab_save_run.finish()

        return self.artifact

    @staticmethod
    @abstractmethod
    def load_wab(project_name, artifact_name, tag='latest', job_type="download-artifact"):
        run = wandb.init(
            project=project_name,
            job_type=job_type
        )

        artifact = run.use_artifact(f'{project_name}/{artifact_name}:{tag}')
        download_path = Path(artifact.download())

        run.finish()

        return artifact, download_path

    @property
    @abstractmethod
    def type(self):
        return ""
