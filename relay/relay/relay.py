#!/usr/bin/env python3
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
from typing import List
import torch
import bagua.torch_api as bagua
from torch.optim import Optimizer
import sys


class RelayAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        hierarchical: bool = True,
        communication_interval: int = 1,
        optimizer: Optimizer = None,
        topology: str = "binary_tree"
    ):
        """
        Implementation of the `RelaySGD` algorithm.

        Args:
            process_group (BaguaProcessGroup): The process group to work on.
            hierarchical (bool): Enable hierarchical communication.
            communication_interval (int): Number of iterations between two communication steps.
            optimizer (Optimizer): A torch Optimizer initialized with model parameters.
            topology (str): Can be ``"binary_tree"`` or ``"chain"``.
        """
        super(RelayAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.communication_interval = communication_interval
        self.cuda_event = torch.cuda.Event()
        self.m_recv = {}
        self.c_recv = {}
        self.ones = torch.ones(1, dtype=torch.float32).cuda()
        self.c_temp = torch.zeros(1, dtype=torch.float32).cuda()
        self.n = torch.zeros(1, dtype=torch.float32).cuda()
        self.x_buffered = 0
        self.optimizer = optimizer
        self.param_size = 0
        for layer in optimizer.param_groups[0]['params']:
            self.param_size += layer.numel()
        # get current rank
        self.rank = bagua.get_local_rank()

        # create neighbour list
        neighbours = []
        if topology == "binary_tree":
            neighbours = [(self.rank - 1) // 2, 2 * self.rank + 1, 2 * self.rank + 2]
        elif topology == "chain":
            neighbours = [self.rank - 1, self.rank + 1]
        else:
            raise NotImplementedError
        
        self.neighbours_filtered = []
        for nb in neighbours:
            if nb >= 0 and nb < bagua.get_world_size():
                self.neighbours_filtered.append(nb)
                self.m_recv[nb] = torch.zeros(self.param_size, dtype=torch.float32).cuda()
                self.c_recv[nb] = torch.ones(1, dtype=torch.float32).cuda()
        self.m_send = torch.zeros(self.param_size, dtype=torch.float32).cuda()
        self.c_send = torch.ones(1, dtype=torch.float32).cuda()


    def _should_communicate(self, bagua_ddp: BaguaDistributedDataParallel) -> bool:
        cur_step = bagua_ddp.bagua_train_step_counter - 1
        return cur_step % self.communication_interval == 0

    def init_tensors(self, bagua_ddp: BaguaDistributedDataParallel) -> List[BaguaTensor]:
        parameters = bagua_ddp.bagua_build_params()
        self.tensors = [
            param.ensure_bagua_tensor(name, bagua_ddp.bagua_module_name)
            for name, param in parameters.__reversed__()
        ]
        return self.tensors

    def tensors_to_buckets(
        self, tensors: List[List[BaguaTensor]], do_flatten: bool
    ) -> List[BaguaBucket]:
        all_tensors = []
        for idx, bucket in enumerate(tensors):
            all_tensors.extend(bucket)

        bagua_bucket = BaguaBucket(all_tensors, flatten=do_flatten, name=str(0))

        return [bagua_bucket]

    def init_forward_pre_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(input):
            return

        return hook

    def init_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(parameter_name, parameter):
            return

        return hook

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook():
            return

        return hook
    
    def init_post_optimizer_step_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(optimizer: torch.optim.Optimizer):
            if not self._should_communicate(bagua_ddp):
                return

            def pack(tensors):
                """Packs a list of tensors into one buffer for sending to other workers"""
                buffer = torch.cat([t.view(-1) for t in tensors])  # copies
                shapes = [tensor.shape for tensor in tensors]
                return buffer, shapes

            def unpack(buffer, shapes):
                """Provides pointers to tensors of original `shapes` in a flat-packed buffer."""
                idx = 0
                entries = []
                for tensor_shape in shapes:
                    end = idx + tensor_shape.numel()
                    entries.append(buffer[idx:end].view(size=tensor_shape))
                    idx = end

                return entries
            
            def sum_wo(dict, wo_key):
                """Sums up values of a given dictionary, excluding the values of wo_key."""
                return sum(value for (key, value) in dict.items() if key != wo_key)

            # init X_i^(t + 1/2)
            x_i = [layer for layer in optimizer.param_groups[0]['params']]
            x_i_buffered, shapes = pack(x_i)
            self.x_buffered = torch.clone(x_i_buffered)

            def send_messages(neighbour):
                # send messages
                self.m_send.copy_(sum_wo(self.m_recv, neighbour) + x_i_buffered)
                bagua.send(self.m_send, neighbour)

                # send corresponding counters
                self.c_send.copy_(sum_wo((self.c_recv), neighbour) + self.ones)
                bagua.send(self.c_send, neighbour)
            
            def recv_messages(neighbour):
                # # recieve messages
                bagua.recv(self.m_recv[neighbour], neighbour)
                bagua.recv(self.c_temp, neighbour)
                self.c_recv[neighbour] = self.c_temp.clone().detach()

            # iterate over neighbours
            for neighbour in self.neighbours_filtered:
                # Deadlock avoidance
                if neighbour < self.rank:
                    send_messages(neighbour)
                    recv_messages(neighbour)
                else:
                    recv_messages(neighbour)
                    send_messages(neighbour)

            # update n and x_i
            self.n = 1 + sum(self.c_recv.values())
            self.x_buffered.add_(sum(self.m_recv.values())).div_(self.n)

            # unpack x_buffered
            x_i_2 = unpack(self.x_buffered, shapes)

            # overwrite current weights
            for idx, layer in enumerate(optimizer.param_groups[0]['params']):
                layer.data.copy_(x_i_2[idx])

        return hook

    def _init_states(self, bucket: BaguaBucket):
        weight_tensor = bucket.flattened_tensor()
        bucket._peer_weight = weight_tensor.ensure_bagua_tensor("peer_weight")

    def init_operations(
        self,
        bagua_ddp: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        self._init_states(bucket)
        torch.cuda.synchronize()
        bucket.clear_ops()


class RelayAlgorithm(Algorithm):
    def __init__(
        self,
        hierarchical: bool = True,
        communication_interval: int = 1,
        optimizer: Optimizer = None,
        topology: str = "binary_tree"
    ):
        """
        Create an instance of the RelaySGD algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            communication_interval (int): Number of iterations between two communication steps.
            optimizer (Optimizer): A torch Optimizer initialized with model parameters.
            topology (str): Can be ``"binary_tree"`` or ``"chain"``.
        """
        self.hierarchical = hierarchical
        self.communication_interval = communication_interval
        self.optimizer = optimizer
        self.topology = topology

    def reify(self, process_group: BaguaProcessGroup) -> RelayAlgorithmImpl:
        return RelayAlgorithmImpl(
            process_group,
            hierarchical=self.hierarchical,
            communication_interval=self.communication_interval,
            optimizer=self.optimizer,
            topology=self.topology
        )
