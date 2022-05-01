#!/usr/bin/env python3
import bagua.torch_api as bagua
from bagua.torch_api.algorithms import gradient_allreduce
import logging
import torch.nn.functional as F
import torch.optim as optim
import torch

USE_RELAY = True

from relay import RelayAlgorithm

def main():
    #torch.manual_seed(42)
    torch.cuda.set_device(bagua.get_local_rank())
    bagua.init_process_group()
    logging.getLogger().setLevel(logging.INFO)

    model = torch.nn.Sequential(torch.nn.Linear(10000, 10000),torch.nn.Linear(10000, 1)).cuda()

    if USE_RELAY:
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        algorithm = RelayAlgorithm(optimizer=optimizer)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        algorithm = gradient_allreduce.GradientAllReduceAlgorithm()


    model = model.with_bagua(
        [optimizer],
        algorithm,
        do_flatten=True,
    )

    model.train()
    X = torch.randn(1000, 10000).cuda()
    y = torch.zeros(1000, 1).cuda()

    for epoch in range(1, 100):
        optimizer.zero_grad()
        output = model(X)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        if bagua.get_local_rank() == 0: logging.info(f"it {epoch}, loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()