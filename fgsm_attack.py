import torch
import torch.nn.functional as F
import torch.nn as nn
from MalConv import MalConv
import numpy as np
import sys
import struct
import os
from tqdm import tqdm

kernel_size = 512
eps = 0.7
target = 0  # benign


def reconstruction(x, y):
    """
    reconstruction restore original bytes from embedding matrix.

    Args:
        x torch.Tensor:
            x is word embedding

        y torch.Tensor:
            y is embedding matrix

    Returns:
        torch.Tensor:
    """
    x_size = x.size()[0]
    y_size = y.size()[0]
    # print(x_size, y_size)

    z = torch.zeros(x_size)

    for i in tqdm(range(x_size)):
        dist = torch.zeros(256)

        for j in range(y_size):
            dist[j] = torch.dist(x[i], y[j])  # computation of euclidean distance

        z[i] = dist.argmin()

    return z


if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        bytez = f.read()

    malconv = MalConv(channels=256, window_size=512, embd_size=8)
    weights = torch.load('./malconv.checkpoint', map_location='cpu')
    malconv.load_state_dict(weights['model_state_dict'])
    malconv.eval()

    opt = torch.optim.SGD(malconv.parameters(), lr=0.01, momentum=0.9)

    payload_size = kernel_size + (kernel_size - np.mod(len(bytez), kernel_size))

    embed = malconv.embd
    m = embed(torch.arange(0, 256))

    label = torch.tensor([target], dtype=torch.long)

    perturbation = np.random.randint(0, 256, payload_size, dtype=np.uint8)
    x = np.frombuffer(bytez, dtype=np.uint8)

    for i in range(5):
        print('[{}]'.format(str(i + 1)))
        opt.zero_grad()

        inp = torch.from_numpy(np.concatenate([x, perturbation])[np.newaxis, :]).float()
        inp_adv = inp.requires_grad_()

        embd_x = embed(inp_adv.long()).detach()
        embd_x.requires_grad = True
        # embd_x.retain_grad()

        outputs = malconv(embd_x)
        results = F.softmax(outputs, dim=1)
        r = results.detach().numpy()[0]
        print('Benign: {:.5g}'.format(r[0]), ', Malicious: {:.5g}'.format(r[1]))

        loss = nn.CrossEntropyLoss()(results, label)
        print('Loss: {:.5g}'.format(loss.item()))

        loss.backward()
        opt.step()
        grad = embd_x.grad
        grad_sign = grad.detach().sign()[0][-payload_size:]

        perturbation = embed(torch.from_numpy(perturbation).long())
        perturbation = (perturbation - eps * grad_sign).detach().numpy()

        embd_x = embd_x.detach().numpy()
        embd_x[0][-payload_size:] = perturbation

        print('Reconstruction fase:')
        perturbation = reconstruction(torch.from_numpy(perturbation), m).detach().numpy()
        print('sum of perturbation: ', perturbation.sum(), '\n')

        with open('perturb.bin', 'wb') as out:
            for s in perturbation:
                out.write(struct.pack('B', int(s)))

        if results[0][0] > 0.5:
            print('misclassification rates: {:.5g}'.format(results[0][0].item()), '\n')
            aes_name = os.path.splitext(sys.argv[1])[0] + '_AEs.exe'

            with open(aes_name, 'wb') as out:
                aes = np.concatenate([x, perturbation.astype(np.uint8)])

                for s in aes:
                    out.write(struct.pack('B', int(s)))

            print(aes_name, ' created.')

            break
