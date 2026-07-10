from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch


def pair_atomicdata_list_to_batch(
    data_list: list[AtomicData], exclude_keys: list | None = None
) -> AtomicData:
    initial_list, relaxed_list = zip(*data_list)
    return atomicdata_list_to_batch(
        list(initial_list), exclude_keys
    ), atomicdata_list_to_batch(list(relaxed_list), exclude_keys)
