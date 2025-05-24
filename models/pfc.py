import torch
import torch.nn.functional as F

from backbone.MNISTMLP import MNISTMLP
from backbone.ResNet import resnet18, resnet34
from models.utils.incremental_model import IncrementalModel
from torch.optim.lr_scheduler import StepLR


class pfc(IncrementalModel):
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, args):
        super(pfc, self).__init__(args)

        self.epochs = args.n_epochs
        self.learning_rate = args.lr
        self.net = None
        self.loss = F.cross_entropy

        self.grad_prev = None

        self.current_task = -1

    def begin_il(self, dataset):
        if self.args.dataset == 'seq-mnist':
            self.net = MNISTMLP(28*28, dataset.nc).to(self.device)
        else:
            if self.args.backbone == 'None' or self.args.backbone == 'resnet18':
                self.net = resnet18(dataset.nc).to(self.device)
            elif self.args.backbone == 'resnet34':
                self.net = resnet34(dataset.nc).to(self.device)


        self.grad_prev = torch.zeros_like(self.net.get_params())

    def train_task(self, dataset, train_loader):
        self.current_task += 1

        self.cpt = int(dataset.nc / dataset.nt)
        self.t_c_arr = dataset.t_c_arr
        self.eye = torch.tril(torch.ones((dataset.nc, dataset.nc))).bool().to(self.device)

        self.train_(train_loader)
        self.end_(train_loader)

    def train_(self, train_loader):
        cls = self.t_c_arr[self.current_task]

        self.net.train()
        opt = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate)
        scheduler = StepLR(opt, step_size=self.args.sche_step, gamma=0.1)
        mask_ratio = self.args.ratio
        for epoch in range(self.epochs):
            for step, data in enumerate(train_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                opt.zero_grad()
                outputs = self.net(inputs)
                loss = self.loss(outputs[:, cls], labels - cls[0])
                loss.backward()

                penalty = torch.tensor(0.0)
                if self.current_task > 0:
                    grad_new = self.net.get_grads().detach()
                    NTK = self.grad_prev * torch.abs(grad_new) # n_f * n_p

                    num_elements = NTK.numel()
                    num_top = int(mask_ratio * num_elements)  

                    k_value = num_elements - num_top
                    values, indices = torch.flatten(NTK).sort(descending=True)
                    threshold = values[k_value]

                    mask = (NTK < threshold).int()

                    progress = 0
                    for pp in list(self.net.parameters()):
                        mask_params = mask[progress: progress +
                                                           torch.tensor(pp.size()).prod()].view(pp.size())
                        progress += torch.tensor(pp.size()).prod()
                        pp.grad *= mask_params

                opt.step()
            if (epoch + 1) % self.args.sche_step == 0:
                mask_ratio *= 0.5
                print('updating mask ratio:', mask_ratio)
            scheduler.step()
            if epoch % self.args.print_freq == 0:
                print('epoch:%d, loss:%.5f, loss_penalty:%.5f' % (epoch, loss.to('cpu').item(), penalty.to('cpu').item()))

    def end_(self, train_loader):
        cls = self.t_c_arr[self.current_task]

        grad_cur = torch.zeros_like(self.net.get_params())

        exp_size = 0
        for j, data in enumerate(train_loader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)

            output = self.net(inputs)
            for i in range(output.size(0)):
                self.net.zero_grad()
                loss = self.loss(output[i, cls].unsqueeze(0), labels[i].unsqueeze(0) - cls[0])
                loss.backward(retain_graph=True if i < labels.size(0) - 1 else False)
                grad_cur += torch.abs(self.net.get_grads().detach())

            exp_size += inputs.shape[0]
            if exp_size > self.args.exp_size:
                break

        grad_cur /= exp_size

        if self.grad_prev is None:
            self.grad_prev = grad_cur
        else:
            self.grad_prev *= self.args.gamma
            self.grad_prev += self.args.gamma * grad_cur


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        x = x.to(self.device)
        with torch.no_grad():
            outputs = self.net(x)
        return outputs





