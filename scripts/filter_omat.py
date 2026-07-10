"""
Filter an ASE DB down to (initial, relaxed) pairs: for each base sid, keep the
single "_0" entry as initial, and the "_N" entry with the highest N as relaxed.
Writes the resulting pairs to a new DB.

Avoids reading full Atoms objects (positions/cell/calculator) until the
second pass, when we already know which indices survived filtering.
"""
import bisect
import logging
import os
import shutil
from itertools import groupby

import ase.db
from fairchem.core.datasets import AseDBDataset


def get_prefix(sid: str) -> str:
    return sid.rsplit("_", 1)[0]


def get_sid_cheap(dataset: AseDBDataset, idx: int) -> str | None:
    """Fetch just the sid for a flat index without materializing a full Atoms object."""
    db_idx = bisect.bisect(dataset._idlen_cumulative, idx)
    el_idx = idx if db_idx == 0 else idx - dataset._idlen_cumulative[db_idx - 1]
    row = dataset.dbs[db_idx]._get_row(dataset.db_ids[db_idx][el_idx])
    data = row.data if isinstance(row.data, dict) else {}
    return data.get("sid", None)


def filter_pairs_to_new_db(src_config: dict, dst_path: str) -> None:
    dataset = AseDBDataset(config=src_config)

    # Pass 1 (cheap): sid only, no atoms conversion.
    sid_idx_pairs = []
    for idx in range(len(dataset)):
        sid = get_sid_cheap(dataset, idx)
        if sid is None:
            logging.warning(f"Entry {idx} has no sid, skipping.")
            continue
        sid_idx_pairs.append((sid, idx))

    sid_idx_pairs.sort(key=lambda pair: get_prefix(pair[0]))

    survivors = []  # flat indices to actually materialize + write
    n_skipped_groups = 0

    for prefix, group_iter in groupby(
        sid_idx_pairs, key=lambda pair: get_prefix(pair[0])
    ):
        group = list(group_iter)

        initial_idx = None
        relaxed_candidates = []  # (n, idx)
        valid = True

        for sid, idx in group:
            suffix = sid.rsplit("_", 1)[-1]
            try:
                n = int(suffix)
            except ValueError:
                valid = False
                break

            if n == 0:
                if initial_idx is not None:
                    valid = (
                        False  # duplicate _0 - real data problem, not just "pick one"
                    )
                    break
                initial_idx = idx
            else:
                relaxed_candidates.append((n, idx))

        if not valid or initial_idx is None or not relaxed_candidates:
            n_skipped_groups += 1
            logging.warning(
                f"Skipping base sid {prefix!r}: needs exactly one _0 and at least one "
                "relaxed entry (or found a duplicate _0)."
            )
            continue

        _, relaxed_idx = max(relaxed_candidates, key=lambda t: t[0])  # highest N wins

        survivors.append(initial_idx)
        survivors.append(relaxed_idx)

    # Pass 2 (expensive): only now do we materialize full Atoms objects, and only
    # for the indices that actually survived filtering.
    new_db = ase.db.connect(dst_path)
    for idx in survivors:
        atoms = dataset.get_atoms(idx)
        new_db.write(atoms, data=atoms.info)
    new_db.close()

    print(
        f"Wrote {len(survivors)} entries ({len(survivors) // 2} pairs); "
        f"skipped {n_skipped_groups} invalid groups."
    )


if __name__ == "__main__":
    dataset_path = os.path.join("data", "omat_rattled_relax")
    config_kwargs = {}
    config = dict(src=dataset_path, **config_kwargs)
    destination_folder = os.path.join("data", "omat_init_and_final_relaxed")
    if os.path.isdir(destination_folder):
        shutil.rmtree(destination_folder)
    os.mkdir(destination_folder)
    filter_pairs_to_new_db(
        config, os.path.join(destination_folder, "filtered_pairs.aselmdb")
    )
