"""
Filter an ASE DB down to (initial, relaxed) pairs: for each base sid, keep the
single "_0" entry as initial, and the "_N" entry with the highest N as relaxed.
Writes the resulting pairs to a new DB.

The sid lives in each row's data, so pass 1 has to decode every row. It does
so in parallel, one shard per task, and skips building Atoms objects. Full
Atoms objects are only materialized in pass 2, for the indices that survived.
"""
import multiprocessing as mp
import os
import shutil
from collections import Counter
from itertools import groupby

import ase.db
from fairchem.core.datasets import AseDBDataset

NUM_WORKERS = os.cpu_count()


def get_prefix(sid: str) -> str:
    return sid.rsplit("_", 1)[0]


def get_shard_sids(args: tuple[str, list[int]]) -> list[str | None]:
    """Read the sid of every row id in one shard, in the order given."""
    path, ids = args
    db = AseDBDataset.connect_db(path)
    sids = []
    for db_id in ids:
        row = db._get_row(db_id)
        data = row.data if isinstance(row.data, dict) else {}
        sids.append(data.get("sid", None))
    db.close()
    return sids


def filter_pairs_to_new_db(src_config: dict, dst_path: str) -> None:
    dataset = AseDBDataset(config=src_config)

    # Pass 1 (parallel over shards): sid only, no atoms conversion.
    tasks = [(str(db.filename), ids) for db, ids in zip(dataset.dbs, dataset.db_ids)]
    sid_idx_pairs = []
    n_no_sid = 0
    with mp.get_context("spawn").Pool(NUM_WORKERS) as pool:
        offset = 0
        for i, shard_sids in enumerate(pool.imap(get_shard_sids, tasks)):
            for el_idx, sid in enumerate(shard_sids):
                if sid is None:
                    n_no_sid += 1
                    continue
                sid_idx_pairs.append((sid, offset + el_idx))
            offset += len(shard_sids)
            print(f"\rPass 1: read {i + 1}/{len(tasks)} shards", end="", flush=True)
    print()
    assert offset == len(dataset)

    sid_idx_pairs.sort(key=lambda pair: get_prefix(pair[0]))

    survivors = []  # flat indices to actually materialize + write
    skip_reasons = Counter()

    for prefix, group_iter in groupby(
        sid_idx_pairs, key=lambda pair: get_prefix(pair[0])
    ):
        group = list(group_iter)

        initial_idx = None
        relaxed_candidates = []  # (n, idx)
        skip_reason = None

        for sid, idx in group:
            suffix = sid.rsplit("_", 1)[-1]
            try:
                n = int(suffix)
            except ValueError:
                skip_reason = "non-integer sid suffix"
                break

            if n == 0:
                if initial_idx is not None:
                    # Duplicate _0 - real data problem, not just "pick one"
                    skip_reason = "duplicate _0"
                    break
                initial_idx = idx
            else:
                relaxed_candidates.append((n, idx))

        if skip_reason is None and initial_idx is None:
            skip_reason = "no _0"
        if skip_reason is None and not relaxed_candidates:
            skip_reason = "no relaxed (_N, N > 0) entry"
        if skip_reason is not None:
            skip_reasons[skip_reason] += 1
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
        f"Read {len(dataset)} entries ({n_no_sid} without sid, ignored). "
        f"Wrote {len(survivors)} entries ({len(survivors) // 2} pairs); "
        f"skipped {sum(skip_reasons.values())} invalid groups."
    )
    for reason, count in skip_reasons.most_common():
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    dataset_path = os.path.join("data", "omat_rattled_relax")
    config_kwargs = {}
    config = dict(src=dataset_path, **config_kwargs)
    # Lives on the large disk; data/omat_init_and_final_relaxed is a symlink to it.
    destination_folder = "/mnt/data0/filipeks/omat_init_and_final_relaxed"
    if os.path.isdir(destination_folder):
        shutil.rmtree(destination_folder)
    os.mkdir(destination_folder)
    filter_pairs_to_new_db(
        config, os.path.join(destination_folder, "filtered_pairs.aselmdb")
    )
