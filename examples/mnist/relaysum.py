#!/usr/bin/env python3
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
import bagua.torch_api as bagua
from bagua.torch_api.algorithms import gradient_allreduce
import logging
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from typing import List
import torch

USE_RELAY = True
DEBUG = True


class RelayAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        hierarchical: bool = True,
        peer_selection_mode: str = "all",
        communication_interval: int = 1,
    ):
        """
        Implementation of the
        `Decentralized SGD <https://tutorials.baguasys.com/algorithms/decentralized>`_
        algorithm.

        Args:
            process_group (BaguaProcessGroup): The process group to work on.
            hierarchical (bool): Enable hierarchical communication.
            peer_selection_mode (str): Can be ``"all"`` or ``"shift_one"``. ``"all"`` means all workers'
                weights are averaged in each communication step. ``"shift_one"`` means each worker
                selects a different peer to do weights average in each communication step.
            communication_interval (int): Number of iterations between two communication steps.

        """
        super(RelayAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.peer_selection_mode = peer_selection_mode
        self.communication_interval = communication_interval
        self.cuda_event = torch.cuda.Event()
        self.m = []
        self.c = torch.zeros(1, dtype=torch.float32).cuda()
        self.recv_c = torch.zeros(1, dtype=torch.float32).cuda()
        self.recv_c_agg = torch.zeros(1, dtype=torch.float32).cuda()
        self.n = torch.zeros(1, dtype=torch.float32).cuda()

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
            if self._should_communicate(bagua_ddp):
                for tensor in self.tensors:
                    tensor.bagua_mark_communication_ready()

        return hook

    def init_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(parameter_name, parameter):
            return

        return hook

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook():
            if self._should_communicate(bagua_ddp):
                bagua_ddp._bagua_backend.wait_pending_comm_ops()

                torch.cuda.current_stream().record_event(self.cuda_event)
                self.cuda_event.synchronize()
                for bucket in bagua_ddp.bagua_buckets:
                    bucket._decentralized_op.copy_back_peer_weight(
                        bucket.backend_bucket
                    )

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

        def relay_mechanism(*args):
            # get current rank
            rank = bagua.get_local_rank()
            if DEBUG: print('Local Rank: {}'.format(rank))

            # create neighbour list
            neighbours = [(rank - 1) // 2, 2 * rank + 1, 2 * rank + 2]
            neighbours_filtered = []
            for nb in neighbours:
                if nb >= 0 and nb < bagua.get_world_size():
                    neighbours_filtered.append(nb)

            # precalc of count
            self.c = 1 + self.recv_c_agg
            self.recv_c_agg = torch.zeros(1, dtype=torch.float32).cuda()

            # iterate over neighbours
            for neighbour in neighbours_filtered:
                PRINT_COND = DEBUG and rank == 3 or neighbour == 3

                if rank > neighbour:
                    # send messages: TODO relay gradient messages
                    send_tensor = bucket.flattened_tensor()
                    if PRINT_COND: print('Sending M from {} to {}'.format(rank, neighbour))
                    bagua.send(send_tensor, neighbour)
                    if PRINT_COND: print('Sent M from {} to {}'.format(rank, neighbour))

                    # recieve messages: TODO aggregate gradient messages
                    recv_tensor_bagua = bucket.flattened_tensor()
                    if PRINT_COND: print('Recieving M from {} to {}'.format(neighbour, rank))
                    bagua.recv(recv_tensor_bagua, neighbour)
                    if PRINT_COND: print('Recieved M from {} to {}'.format(neighbour, rank))

                    # send precalculated relayed counts
                    if PRINT_COND: print('Sending c from {} to {}'.format(rank, neighbour))
                    bagua.send(self.c, neighbour)
                    if PRINT_COND: print('Sent c from {} to {}'.format(rank, neighbour))

                    # recieve and aggregate counts
                    if PRINT_COND: print('Recieving c from {} to {}'.format(neighbour, rank))
                    bagua.recv(self.recv_c, neighbour)
                    if PRINT_COND: print('Recieved c from {} to {}'.format(neighbour, rank))
                    self.recv_c_agg += self.recv_c
                else:
                    # recieve messages: TODO aggregate gradient messages
                    recv_tensor_bagua = bucket.flattened_tensor()
                    if PRINT_COND: print('Recieving M from {} to {}'.format(neighbour, rank))
                    bagua.recv(recv_tensor_bagua, neighbour)
                    if PRINT_COND: print('Recieved M from {} to {}'.format(neighbour, rank))

                    # send messages: TODO relay gradient messages
                    send_tensor = bucket.flattened_tensor()
                    if PRINT_COND: print('Sending M from {} to {}'.format(rank, neighbour))
                    bagua.send(send_tensor, neighbour)
                    if PRINT_COND: print('Sent M from {} to {}'.format(rank, neighbour))

                    # recieve and aggregate counts
                    if PRINT_COND: print('Recieving c from {} to {}'.format(neighbour, rank))
                    bagua.recv(self.recv_c, neighbour)
                    if PRINT_COND: print('Recieved c from {} to {}'.format(neighbour, rank))
                    self.recv_c_agg += self.recv_c

                    # send precalculated relayed counts
                    if PRINT_COND: print('Sending c from {} to {}'.format(rank, neighbour))
                    bagua.send(self.c, neighbour)
                    if PRINT_COND: print('Sent c from {} to {}'.format(rank, neighbour))
            
            # update n
            self.n = 1 + self.recv_c_agg

            # TODO update gradient

            print(bagua.get_local_rank())
            print(neighbours_filtered)
            print(bagua.get_world_size())

        bucket.append_python_op(relay_mechanism, group=self.process_group)
        decentralized_op = bucket.append_decentralized_synchronous_op(
            peer_weight=bucket._peer_weight,
            hierarchical=self.hierarchical,
            peer_selection_mode=self.peer_selection_mode,
            group=self.process_group,
        )
        bucket._decentralized_op = decentralized_op


class RelayAlgorithm(Algorithm):
    def __init__(
        self,
        hierarchical: bool = True,
        peer_selection_mode: str = "all",
        communication_interval: int = 1,
    ):
        """
        Create an instance of the
        `Decentralized SGD <https://tutorials.baguasys.com/algorithms/decentralized>`_
        algorithm.

        Args:
            hierarchical (bool): Enable hierarchical communication.
            peer_selection_mode (str): Can be ``"all"`` or ``"shift_one"``. ``"all"`` means all workers'
                weights are averaged in each communication step. ``"shift_one"`` means each worker
                selects a different peer to do weights average in each communication step.
            communication_interval (int): Number of iterations between two communication steps.

        """
        self.hierarchical = hierarchical
        self.peer_selection_mode = peer_selection_mode
        self.communication_interval = communication_interval

    def reify(self, process_group: BaguaProcessGroup) -> RelayAlgorithmImpl:
        return RelayAlgorithmImpl(
            process_group,
            hierarchical=self.hierarchical,
            peer_selection_mode=self.peer_selection_mode,
            communication_interval=self.communication_interval,
        )

def main():
    torch.manual_seed(42)
    torch.cuda.set_device(bagua.get_local_rank())
    bagua.init_process_group()
    logging.getLogger().setLevel(logging.INFO)

    model = torch.nn.Sequential(torch.nn.Linear(200, 1)).cuda()

    if USE_RELAY:
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        algorithm = RelayAlgorithm()
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        algorithm = gradient_allreduce.GradientAllReduceAlgorithm()


    model = model.with_bagua(
        [optimizer],
        algorithm,
        do_flatten=True,
    )

    model.train()
    X = torch.randn(1000, 200).cuda()
    y = torch.zeros(1000, 1).cuda()

    for epoch in range(1, 101):
        optimizer.zero_grad()
        output = model(X)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        logging.info(f"it {epoch}, loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()