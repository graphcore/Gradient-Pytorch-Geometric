from typing import Callable, Dict, List, Optional, Tuple, Union

import poptorch
import torch
from poptorch_geometric.collate import FixedSizeCollater
from poptorch_geometric.collate import CombinedBatchingCollater
from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader.utils import get_input_nodes
from torch_geometric.sampler import NeighborSampler
from torch_geometric.typing import EdgeType, InputNodes, OptTensor


# TODO: Tidy
# TODO: Fix below issues
# TODO: Heuristic for setting num_neighbours?
# TODO: Verify pruning doesn't happen on the wrong bits - that the IDs still align correctly
# TODO: Add padded node and edge to original graph and pad the e_id and n_id with ids of that padded node / edge

"""
Known issues:
 - `batch_size` must be added to `exclude_keys` as it the FixedSizeCollator doesn't handle it correctly
 - `e_id` and `n_id` not padded correctly - requires this change to upstream:https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/storage.py#L653
    '''
    if key in E_KEYS:
        self._cached_attr[AttrType.EDGE].add(key)
        return True
    else:
        self._cached_attr[AttrType.NODE].add(key)
        return False
    '''
 - Trimming needs to be input_id aware - shouldn't prune the batch ids
 - Input ids, get doubled when the call to the default collater happens, caused by: https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/loader/dataloader.py#L12
    `batch = super().__call__(data_list)`
 - Edge indices are converted to floats? - this needs to return the correct types
    @_create_structure_dict.register(HeteroData)
    def _(self, data: HeteroData) -> Dict[Union[NodeType, EdgeType], Any]:
        out = dict()
        for key, attr in data._global_store.to_dict().items():  # pylint: disable=protected-access
            out[key] = _reset_attr(attr)
        for key, attr in chain(data.node_items(), data.edge_items()):
            out[key] = {
                k: torch.zeros(_reset_dim(v.shape, k))
                for k, v in attr.to_dict().items()
                if isinstance(v, torch.Tensor)
            }
        return out

        , dtype=data[key][k].dtype)
"""


class PyGFixedSizeNeighborLoader(torch.utils.data.DataLoader):

    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        input_nodes: InputNodes = None,
        input_time: OptTensor = None,
        replace: bool = False,
        directed: bool = True,
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
        time_attr: Optional[str] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        is_sorted: bool = False,
        filter_per_worker: bool = False,
        batch_size: int = 1,
        neighbor_sampler: Optional[NeighborSampler] = None,
        collater_args: Optional[Dict[str, Union[int, float]]] = None,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.neighbour_loader = NeighborLoader(
            data,
            num_neighbors,
            input_nodes=input_nodes,
            input_time=input_time,
            replace=replace,
            directed=directed,
            disjoint=disjoint,
            temporal_strategy=temporal_strategy,
            time_attr=time_attr,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            is_sorted=is_sorted,
            filter_per_worker=filter_per_worker,
            neighbor_sampler=neighbor_sampler,
        )
        _, input_nodes = get_input_nodes(data, input_nodes)

        collater_args = collater_args if collater_args else {}
        collater = self._create_collater(**collater_args)
        super().__init__(dataset=range(input_nodes.size(0)),
                         batch_size=self.batch_size,
                         collate_fn=collater,
                         **kwargs)

    def __collate__(self, index):
        out = self.neighbour_loader.collate_fn(index)
        out = self.fixed_size_collater([out])
        return out

    def _create_collater(self, **collater_args):
        self.fixed_size_collater = FixedSizeCollater(**collater_args)
        return self.__collate__


class FixedSizeNeighborLoader(PyGFixedSizeNeighborLoader, poptorch.DataLoader):

    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        input_nodes: InputNodes = None,
        input_time: OptTensor = None,
        replace: bool = False,
        directed: bool = True,
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
        time_attr: Optional[str] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        is_sorted: bool = False,
        filter_per_worker: bool = False,
        batch_size: int = 1,
        neighbor_sampler: Optional[NeighborSampler] = None,
        collater_args: Optional[Dict[str, Union[int, float]]] = None,
        options: Optional[poptorch.Options] = None,
        **kwargs,
    ):
        self.batch_size = batch_size

        if options is None:
            # Create IPU default options
            options = poptorch.Options()

        super().__init__(
            data,
            num_neighbors,
            input_nodes=input_nodes,
            input_time=input_time,
            replace=replace,
            directed=directed,
            disjoint=disjoint,
            temporal_strategy=temporal_strategy,
            time_attr=time_attr,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            is_sorted=is_sorted,
            filter_per_worker=True,
            batch_size=batch_size,
            neighbor_sampler=neighbor_sampler,
            collater_args=collater_args,
            options=options,
            **kwargs,
        )

    def _create_collater(self, **collater_args):
        collater = super()._create_collater(**collater_args)
        return CombinedBatchingCollater(mini_batch_size=self.batch_size,
                                        collater=collater)
