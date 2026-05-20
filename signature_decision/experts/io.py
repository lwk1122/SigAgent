from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import uuid

import pandas as pd

from .schema import ExpertRequest

NORMALIZED_MUTATION_TYPES = {
    "96": "SBS96",
    "SBS": "SBS96",
    "SBS96": "SBS96",
    "78": "DBS78",
    "DBS": "DBS78",
    "DBS78": "DBS78",
    "83": "ID83",
    "ID": "ID83",
    "ID83": "ID83",
}

KNOWN_METADATA_COLUMNS = {
    "SBS96": ["Mutation.type", "Trinucleotide"],
    "DBS78": ["Ref", "Var"],
    "ID83": ["Type", "Subtype", "Indel_size", "Repeat_MH_size"],
}


@dataclass(slots=True)
class PreparedMatrix:
    numeric_frame: pd.DataFrame
    metadata_frame: pd.DataFrame
    metadata_columns: list[str]
    source: str | None


def normalize_mutation_type(mutation_type: str) -> str:
    normalized = NORMALIZED_MUTATION_TYPES.get(str(mutation_type).upper())
    if normalized is None:
        raise ValueError(f"Unsupported mutation type: {mutation_type}")
    return normalized


def _load_frame(source: str | Path | pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    if isinstance(source, pd.DataFrame):
        return source.copy(), None
    path = Path(source)
    return pd.read_csv(path), str(path)


def _is_strictly_numeric(series: pd.Series) -> bool:
    coerced = pd.to_numeric(series, errors="coerce")
    return bool(coerced.notna().all())


def _split_matrix_frame(frame: pd.DataFrame, mutation_type: str, source: str | None) -> PreparedMatrix:
    known_metadata = set(KNOWN_METADATA_COLUMNS[mutation_type])
    metadata_columns: list[str] = []
    for column in frame.columns:
        if column in known_metadata or not _is_strictly_numeric(frame[column]):
            metadata_columns.append(column)
    numeric_columns = [column for column in frame.columns if column not in metadata_columns]
    if not numeric_columns:
        raise ValueError(f"No numeric columns found in {source or 'dataframe input'}.")
    numeric_frame = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="raise")
    metadata_frame = frame.loc[:, metadata_columns].copy() if metadata_columns else pd.DataFrame(index=frame.index)
    return PreparedMatrix(
        numeric_frame=numeric_frame,
        metadata_frame=metadata_frame,
        metadata_columns=metadata_columns,
        source=source,
    )


def _values_to_key(frame: pd.DataFrame, columns: list[str]) -> pd.Index:
    if not columns:
        return pd.Index([f"row_{i:03d}" for i in range(len(frame))])
    return pd.Index(frame.loc[:, columns].astype(str).agg("|".join, axis=1))


def _unique(values: list[str] | pd.Index) -> bool:
    values = list(values)
    return len(set(values)) == len(values)


def _preferred_channel_ids(metadata: pd.DataFrame, mutation_type: str) -> pd.Index:
    if mutation_type == "SBS96" and {"Mutation.type", "Trinucleotide"}.issubset(metadata.columns):
        values = [
            f"{str(tri)[0]}[{str(mut)}]{str(tri)[2]}" if len(str(tri)) >= 3 else f"{mut}|{tri}"
            for mut, tri in metadata.loc[:, ["Mutation.type", "Trinucleotide"]].itertuples(index=False)
        ]
        if _unique(values):
            return pd.Index(values)
    if mutation_type == "DBS78" and {"Ref", "Var"}.issubset(metadata.columns):
        values = [f"{ref}>{var}" for ref, var in metadata.loc[:, ["Ref", "Var"]].itertuples(index=False)]
        if _unique(values):
            return pd.Index(values)
    if mutation_type == "ID83" and {"Type", "Subtype", "Indel_size", "Repeat_MH_size"}.issubset(metadata.columns):
        values = [
            f"{indel_type}:{subtype}:{indel_size}:{repeat_mh_size}"
            for indel_type, subtype, indel_size, repeat_mh_size in metadata.loc[
                :, ["Type", "Subtype", "Indel_size", "Repeat_MH_size"]
            ].itertuples(index=False)
        ]
        if _unique(values):
            return pd.Index(values)
    if len(metadata.columns) > 0:
        values = metadata.astype(str).agg("|".join, axis=1).tolist()
        if _unique(values):
            return pd.Index(values)
    return pd.Index([f"channel_{i:03d}" for i in range(len(metadata))])


def _align_frames(
    sample_prepared: PreparedMatrix,
    signature_prepared: PreparedMatrix,
    mutation_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, str]:
    shared_columns = [
        column
        for column in sample_prepared.metadata_columns
        if column in signature_prepared.metadata_columns
    ]
    sample_matrix = sample_prepared.numeric_frame.copy()
    signature_matrix = signature_prepared.numeric_frame.copy()
    channel_metadata = sample_prepared.metadata_frame.copy()

    if shared_columns:
        sample_keys = _values_to_key(sample_prepared.metadata_frame, shared_columns)
        signature_keys = _values_to_key(signature_prepared.metadata_frame, shared_columns)
        if _unique(sample_keys) and _unique(signature_keys) and set(sample_keys) == set(signature_keys):
            sample_matrix.index = sample_keys
            signature_matrix.index = signature_keys
            signature_matrix = signature_matrix.loc[sample_keys]
            channel_metadata.index = sample_keys
            preferred_ids = _preferred_channel_ids(sample_prepared.metadata_frame, mutation_type)
            sample_matrix.index = preferred_ids
            signature_matrix.index = preferred_ids
            channel_metadata.index = preferred_ids
            return sample_matrix, signature_matrix, channel_metadata, "shared_metadata"

    if sample_matrix.shape[0] != signature_matrix.shape[0]:
        raise ValueError("Could not align sample and signature matrices by metadata or row order.")

    preferred_ids = _preferred_channel_ids(sample_prepared.metadata_frame, mutation_type)
    sample_matrix.index = preferred_ids
    signature_matrix.index = preferred_ids
    channel_metadata.index = preferred_ids
    return sample_matrix, signature_matrix, channel_metadata, "row_order"


def load_expert_request(
    *,
    sample_source: str | Path | pd.DataFrame,
    signature_source: str | Path | pd.DataFrame,
    mutation_type: str,
    sample_ids: list[str] | None = None,
    signature_names: list[str] | None = None,
    request_id: str | None = None,
    reference_name: str | None = None,
) -> ExpertRequest:
    normalized_type = normalize_mutation_type(mutation_type)
    sample_frame, sample_path = _load_frame(sample_source)
    signature_frame, signature_path = _load_frame(signature_source)

    sample_prepared = _split_matrix_frame(sample_frame, normalized_type, sample_path)
    signature_prepared = _split_matrix_frame(signature_frame, normalized_type, signature_path)

    sample_matrix, signature_matrix, channel_metadata, alignment_strategy = _align_frames(
        sample_prepared,
        signature_prepared,
        normalized_type,
    )

    if sample_ids is not None:
        missing = [sample_id for sample_id in sample_ids if sample_id not in sample_matrix.columns]
        if missing:
            raise KeyError(f"Unknown sample ids in request: {missing}")
        sample_matrix = sample_matrix.loc[:, sample_ids]

    if signature_names is not None:
        missing = [signature_name for signature_name in signature_names if signature_name not in signature_matrix.columns]
        if missing:
            raise KeyError(f"Unknown signature names in request: {missing}")
        signature_matrix = signature_matrix.loc[:, signature_names]

    return ExpertRequest(
        mutation_type=normalized_type,
        sample_matrix=sample_matrix,
        signature_matrix=signature_matrix,
        channel_metadata=channel_metadata,
        sample_source=sample_path,
        signature_source=signature_path,
        reference_name=reference_name,
        request_id=request_id or str(uuid.uuid4()),
        alignment_strategy=alignment_strategy,
    )
