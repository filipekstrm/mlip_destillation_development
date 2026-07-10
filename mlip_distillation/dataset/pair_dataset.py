from __future__ import annotations

import bisect
import logging

import ase
from fairchem.core.common.registry import registry
from fairchem.core.datasets._utils import rename_data_object_keys
from fairchem.core.datasets.ase_datasets import AseDBDataset


class AsePairDBDataset(AseDBDataset):
    """
    Dataset that returns (initial, relaxed) structure pairs, backed by the same ASE DB
    machinery as AseDBDataset. Pairs are derived from "sid" values that share a common
    base and end in "_N": N=0 denotes the initial structure, and the other N denotes the
    relaxed structure (e.g. "mp-12345_0" is initial, "mp-12345_4" is relaxed).

    Assumes exactly two entries per base sid (one "_0" and one other). If your underlying
    DB contains more than two entries for a given base sid, pre-filter it yourself before
    passing it to this dataset.

    args:
        config (dict):
            src / connect_args / select_args: same as AseDBDataset - describes the underlying
                    DB(s) containing exactly the initial + relaxed structures to be paired.

            a2g_args / atoms_transform_args / transforms / key_mapping: same meaning as in
                    AseDBDataset, applied independently to each structure in the pair.
    """

    def _get_sid(self, idx: int) -> str | None:
        """Cheaply fetch just the sid for a flat underlying index, without a full get_atoms()."""
        db_idx = bisect.bisect(self._idlen_cumulative, idx)
        el_idx = idx if db_idx == 0 else idx - self._idlen_cumulative[db_idx - 1]
        row = self.dbs[db_idx]._get_row(self.db_ids[db_idx][el_idx])
        data = row.data if isinstance(row.data, dict) else {}
        return data.get("sid", None)

    def _load_dataset_get_ids(self, config: dict) -> list[tuple[int, int]]:
        # Reuse AseDBDataset entirely to connect to the db(s) and build the flat id list.
        base_ids = super()._load_dataset_get_ids(config)

        # group flat indices by base sid -> {"initial": idx, "relaxed": idx}
        groups: dict[str, dict] = {}
        skipped_no_sid = 0
        for idx in base_ids:
            sid = self._get_sid(idx)
            if sid is None or "_" not in sid:
                skipped_no_sid += 1
                continue

            base_sid, _, suffix = sid.rpartition("_")
            try:
                n = int(suffix)
            except ValueError:
                skipped_no_sid += 1
                continue

            group = groups.setdefault(base_sid, {"initial": None, "relaxed": None})
            if n == 0:
                if group["initial"] is not None:
                    raise ValueError(
                        f"Found more than one '_0' entry for base sid {base_sid!r}. "
                        "AsePairDBDataset expects exactly one initial structure per base sid "
                        "- pre-filter the underlying DB first."
                    )
                group["initial"] = idx
            else:
                if group["relaxed"] is not None:
                    raise ValueError(
                        f"Found more than one non-'_0' entry for base sid {base_sid!r}. "
                        "AsePairDBDataset expects exactly one relaxed structure per base sid "
                        "- pre-filter the underlying DB first."
                    )
                group["relaxed"] = idx

        if skipped_no_sid:
            logging.warning(
                f"AsePairDBDataset: skipped {skipped_no_sid} entries with no sid or "
                "no trailing '_N' suffix; they cannot be paired."
            )

        pairs = []
        n_missing_initial = 0
        n_missing_relaxed = 0
        for base_sid, group in groups.items():
            if group["initial"] is None:
                n_missing_initial += 1
                continue
            if group["relaxed"] is None:
                n_missing_relaxed += 1
                continue
            pairs.append((group["initial"], group["relaxed"]))

        if n_missing_initial:
            logging.warning(
                f"AsePairDBDataset: {n_missing_initial} sid groups had a relaxed structure "
                "but no initial (_0) structure; they were skipped."
            )
        if n_missing_relaxed:
            logging.warning(
                f"AsePairDBDataset: {n_missing_relaxed} sid groups had an initial (_0) "
                "structure but no relaxed structure; they were skipped."
            )

        return pairs

    def get_atoms(self, idx: int) -> tuple[ase.Atoms, ase.Atoms]:
        idx_initial, idx_relaxed = self.ids[idx]
        atoms_initial = super().get_atoms(idx_initial)
        atoms_relaxed = super().get_atoms(idx_relaxed)
        return atoms_initial, atoms_relaxed

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]

        atoms_initial, atoms_relaxed = self.get_atoms(idx)

        if self.atoms_transform is not None:
            transform_args = self.config.get("atoms_transform_args", {})
            atoms_initial = self.atoms_transform(atoms_initial, **transform_args)
            atoms_relaxed = self.atoms_transform(atoms_relaxed, **transform_args)

        sid_initial = atoms_initial.info.get("sid", self.ids[idx][0])
        sid_relaxed = atoms_relaxed.info.get("sid", self.ids[idx][1])

        data_initial = self.transforms(self.a2g(atoms_initial, sid=sid_initial))
        data_relaxed = self.transforms(self.a2g(atoms_relaxed, sid=sid_relaxed))

        if self.key_mapping is not None:
            data_initial = rename_data_object_keys(data_initial, self.key_mapping)
            data_relaxed = rename_data_object_keys(data_relaxed, self.key_mapping)

        return data_initial, data_relaxed

    def get_relaxed_energy(self, identifier):
        raise NotImplementedError(
            "get_relaxed_energy doesn't apply here directly — the relaxed structure's "
            "energy is already available on 'structure_relaxed' after a2g conversion "
            "(e.g. via r_energy=True in a2g_args)."
        )

    def get_metadata(self, attr, idx):
        if attr == "natoms":
            idx_initial, idx_relaxed = self.ids[idx]
            return (len(self.get_atoms(idx_initial)), len(self.get_atoms(idx_relaxed)))
        return super().get_metadata(attr, idx)
