"""Experiment-specific dataset adapters."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Sequence

from .core import InferenceJob, VALID_LABELS

LOGGER = logging.getLogger("slurm_experiment")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class FaceSpectrogramDatasetAdapter:
    """Build File 4 jobs containing one face and one spectrogram image."""

    def __init__(
        self,
        face_dir: Path,
        spectrogram_dir: Path,
        *,
        strict_pairs: bool = False,
        allow_positional_fallback: bool = False,
    ) -> None:
        """Remember where the face images and spectrogram images live.

        Args:
            face_dir: Folder that contains one sub-folder per label
                (e.g. Positive/Negative/Neutral) full of face images.
            spectrogram_dir: Same idea, but for spectrogram images.
            strict_pairs: If True, raise an error whenever a face image
                has no matching spectrogram (or vice versa) instead of
                silently skipping it.
            allow_positional_fallback: If True and no images can be
                matched by filename, fall back to pairing them purely
                by their position/order in each folder.
        """
        self.face_dir = face_dir
        self.spectrogram_dir = spectrogram_dir
        self.strict_pairs = strict_pairs
        self.allow_positional_fallback = allow_positional_fallback

    @staticmethod
    def _normalise_stem(stem: str) -> str:
        """Strip helper words like "img"/"spec" out of a filename stem.

        This turns filenames like "happy-img-001" and "happy-spec-001"
        into the same key ("happy-001") so a face image and its matching
        spectrogram image can be recognised as a pair.

        Args:
            stem: A filename without its extension (e.g. "happy-img-001").

        Returns:
            The same stem with the words "img"/"spec" removed.
        """
        return "-".join(
            part for part in stem.split("-") if part.lower() not in {"img", "spec"}
        )

    @staticmethod
    def _list_images(root: Path, label: str) -> list[Path]:
        """Find all usable image files for one label/class folder.

        Looks inside `root/label`, keeps only real image files
        (.jpg/.jpeg/.png) and skips hidden/system files (like macOS
        "._" files), then returns them sorted for reproducibility.

        Args:
            root: The base folder (e.g. face_dir or spectrogram_dir).
            label: The class/emotion sub-folder name to look inside.

        Returns:
            A sorted list of image file paths for that label.

        Raises:
            FileNotFoundError: If `root/label` does not exist.
        """
        class_dir = root / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Required class directory not found: {class_dir}")

        images: list[Path] = []
        for path in class_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative_parts = path.relative_to(class_dir).parts
            if path.name.startswith("._") or any(part.startswith(".") for part in relative_parts):
                continue
            images.append(path)
        return sorted(images, key=lambda item: str(item.relative_to(class_dir)))

    def _index(self, paths: Sequence[Path], modality: str) -> dict[str, Path]:
        """Turn a list of image paths into a lookup dict by normalised stem.

        This lets us quickly find "does this face image have a matching
        spectrogram image?" by comparing normalised filenames.

        Args:
            paths: Image file paths to index (all faces, or all specs).
            modality: Human-readable name used only in error messages
                (e.g. "face" or "spectrogram").

        Returns:
            A dict mapping normalised filename stem -> image path.

        Raises:
            ValueError: If two different files normalise to the same
                stem, since that would make the pairing ambiguous.
        """
        index: dict[str, Path] = {}
        for path in paths:
            key = self._normalise_stem(path.stem)
            if key in index:
                raise ValueError(
                    f"Duplicate normalised {modality} stem {key!r}: "
                    f"{index[key]} and {path}"
                )
            index[key] = path
        return index

    def build(self) -> list[InferenceJob]:
        """Create one inference job per matched face+spectrogram pair.

        For every label (Positive/Negative/Neutral), this lists the face
        images and spectrogram images, matches them up by normalised
        filename, logs how many matched vs. didn't, and turns each
        matched pair into an `InferenceJob` with a unique, reproducible
        job_id (a hash of the label + both file paths).

        Returns:
            A list of InferenceJob objects, one per matched pair, across
            all labels.

        Raises:
            ValueError: If `strict_pairs` is True and some files could
                not be matched, if a label has no matching pairs and
                positional fallback is disabled, or if no jobs at all
                could be built.
        """
        jobs: list[InferenceJob] = []
        for label in VALID_LABELS:
            faces = self._list_images(self.face_dir, label)
            specs = self._list_images(self.spectrogram_dir, label)
            face_index = self._index(faces, "face")
            spec_index = self._index(specs, "spectrogram")

            common = sorted(set(face_index) & set(spec_index))
            face_only = sorted(set(face_index) - set(spec_index))
            spec_only = sorted(set(spec_index) - set(face_index))
            LOGGER.info(
                "%s: face=%d spec=%d matched=%d face-only=%d spec-only=%d",
                label,
                len(faces),
                len(specs),
                len(common),
                len(face_only),
                len(spec_only),
            )

            if self.strict_pairs and (face_only or spec_only):
                raise ValueError(f"Unmatched files found for {label}")

            pairs = [(face_index[key], spec_index[key]) for key in common]
            if not pairs:
                if not self.allow_positional_fallback:
                    raise ValueError(
                        f"No matching face/spectrogram stems for {label}; "
                        "positional fallback is disabled."
                    )
                pairs = list(zip(faces, specs))

            for face_path, spec_path in pairs:
                identity = f"{label}|{face_path.resolve()}|{spec_path.resolve()}"
                job_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
                jobs.append(
                    InferenceJob(
                        job_id=job_id,
                        label=label,
                        image_paths=(face_path, spec_path),
                        metadata={
                            "face_path": str(face_path),
                            "spec_path": str(spec_path),
                        },
                    )
                )

        if not jobs:
            raise ValueError("No face/spectrogram pairs were found")
        return jobs

    @staticmethod
    def split(
        jobs: Sequence[InferenceJob], test_size: float, seed: int
    ) -> tuple[list[InferenceJob], list[InferenceJob]]:
        """Split jobs into train/test sets, keeping label proportions equal.

        Args:
            jobs: All the jobs to split.
            test_size: Fraction of jobs to put in the test set (e.g. 0.2
                for 20%).
            seed: Random seed so the split is the same every time it's run.

        Returns:
            A (train_jobs, test_jobs) tuple.

        Raises:
            RuntimeError: If scikit-learn is not installed.
        """
        try:
            from sklearn.model_selection import train_test_split
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required") from exc

        train, test = train_test_split(
            list(jobs),
            test_size=test_size,
            shuffle=True,
            stratify=[job.label for job in jobs],
            random_state=seed,
        )
        return list(train), list(test)


class FaceImageDatasetAdapter:
    """Build File 5 jobs containing one face image."""

    def __init__(self, face_dir: Path) -> None:
        """Remember where the face images live.

        Args:
            face_dir: Folder that contains one sub-folder per label
                (e.g. Positive/Negative/Neutral) full of face images.
                It is expanded (e.g. "~") and resolved to an absolute path.
        """
        self.face_dir = face_dir.expanduser().resolve()

    @staticmethod
    def _list_images(root: Path, label: str) -> list[Path]:
        """Return supported, non-hidden images for one class."""
        class_dir = root / label

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Required class directory not found: {class_dir}"
            )

        images: list[Path] = []

        for path in class_dir.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            relative_parts = path.relative_to(class_dir).parts

            if path.name.startswith("._"):
                continue

            if any(part.startswith(".") for part in relative_parts):
                continue

            images.append(path)

        return sorted(
            images,
            key=lambda item: str(item.relative_to(class_dir)),
        )

    def build(self) -> list[InferenceJob]:
        """Build one inference job for every face image."""
        jobs: list[InferenceJob] = []

        for label in VALID_LABELS:
            images = self._list_images(self.face_dir, label)

            LOGGER.info(
                "%s: face images=%d",
                label,
                len(images),
            )

            for image_path in images:
                resolved_path = image_path.resolve()
                identity = f"{label}|{resolved_path}"
                job_id = hashlib.sha256(
                    identity.encode("utf-8")
                ).hexdigest()[:20]

                jobs.append(
                    InferenceJob(
                        job_id=job_id,
                        label=label,
                        image_paths=(resolved_path,),
                        metadata={
                            "face_path": str(resolved_path),
                        },
                    )
                )

        if not jobs:
            raise ValueError(
                f"No face images were found under: {self.face_dir}"
            )

        return jobs

    @staticmethod
    def split(
        jobs: Sequence[InferenceJob],
        test_size: float,
        seed: int,
    ) -> tuple[list[InferenceJob], list[InferenceJob]]:
        """Create a reproducible stratified train/test split."""
        try:
            from sklearn.model_selection import train_test_split
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn is required"
            ) from exc

        train, test = train_test_split(
            list(jobs),
            test_size=test_size,
            shuffle=True,
            stratify=[job.label for job in jobs],
            random_state=seed,
        )

        return list(train), list(test)

class SpectrogramImageDatasetAdapter:
    """Build File 5 jobs containing one spectrogram image."""

    def __init__(self, spectrogram_dir: Path) -> None:
        """Remember where the spectrogram images live.

        Args:
            spectrogram_dir: Folder that contains one sub-folder per
                label (e.g. Positive/Negative/Neutral) full of
                spectrogram images. It is expanded (e.g. "~") and
                resolved to an absolute path.
        """
        self.spectrogram_dir = spectrogram_dir.expanduser().resolve()

    @staticmethod
    def _list_images(root: Path, label: str) -> list[Path]:
        """Return supported, non-hidden images for one class."""
        class_dir = root / label

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Required class directory not found: {class_dir}"
            )

        images: list[Path] = []

        for path in class_dir.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            relative_parts = path.relative_to(class_dir).parts

            if path.name.startswith("._"):
                continue

            if any(part.startswith(".") for part in relative_parts):
                continue

            images.append(path)

        return sorted(
            images,
            key=lambda item: str(item.relative_to(class_dir)),
        )

    def build(self) -> list[InferenceJob]:
        """Build one inference job for every spectrogram image."""
        jobs: list[InferenceJob] = []

        for label in VALID_LABELS:
            images = self._list_images(self.spectrogram_dir, label)

            LOGGER.info(
                "%s: spectrogram images=%d",
                label,
                len(images),
            )

            for image_path in images:
                resolved_path = image_path.resolve()
                identity = f"{label}|{resolved_path}"
                job_id = hashlib.sha256(
                    identity.encode("utf-8")
                ).hexdigest()[:20]

                jobs.append(
                    InferenceJob(
                        job_id=job_id,
                        label=label,
                        image_paths=(resolved_path,),
                        metadata={
                            "spec_path": str(resolved_path),
                        },
                    )
                )

        if not jobs:
            raise ValueError(
                f"No spectrogram images were found under: {self.spectrogram_dir}"
            )

        return jobs

    @staticmethod
    def split(
        jobs: Sequence[InferenceJob],
        test_size: float,
        seed: int,
    ) -> tuple[list[InferenceJob], list[InferenceJob]]:
        """Create a reproducible stratified train/test split."""
        try:
            from sklearn.model_selection import train_test_split
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn is required"
            ) from exc

        train, test = train_test_split(
            list(jobs),
            test_size=test_size,
            shuffle=True,
            stratify=[job.label for job in jobs],
            random_state=seed,
        )

        return list(train), list(test)
