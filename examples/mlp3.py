import sys
sys.path.append("../tensor_network")

from ops import matmul, place_holder
def mlp3_generator():
    tensors = list()
    x = place_holder(("batchsize", "d0"), "x")
    tensors.append(x)
    for i in range(3):
        weight = place_holder((f"d{i}", f"d{i+1}"), f"w{i}", require_grads=True)
        tensors.append(weight)
        x = matmul(x, weight, label=f"linear{i}")
        tensors.append(x)
    return tensors


if __name__ == '__main__':
    mlp3 = mlp3_generator()
    for tensor in mlp3:
        print(tensor)
        