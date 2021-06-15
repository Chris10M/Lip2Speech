import math
import torch.optim as optim


class Optimzer:
    def __init__(self, model, start_iter, max_iter, lr=0.01, momentum=0.9, weight_decay=0.0005, patience=2, power=0.9):
        self.net = model
        self.optim: optim.Optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        self.lr = lr

        self.start_iter = start_iter
        self.max_iter = max_iter

        self.current_iter = start_iter
        self.power = power
        self.patience = patience

        self.lr_multiplier = 1
        self.eval_losses = list()

    def update_lr(self):
        lr = self.lr * math.pow((1 - (self.current_iter / self.max_iter)), self.power) * self.lr_multiplier

        for pg in self.optim.param_groups:
            # if pg.get('lr_mul', False):
            #     pg['lr'] = lr * 10
            # else:
                pg['lr'] = lr
        # if self.optim.defaults.get('lr_mul', False):
        #     self.optim.defaults['lr'] = lr * 10
        # else:
        self.optim.defaults['lr'] = lr

    def step(self):
        self.optim.step()
        self.current_iter += 1

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def state_dict(self):
        return self.optim.state_dict()

    def grad_div(self, optim_iter):
        for param in self.net.parameters():
                if param.grad is not None:        
                    param.grad = param.grad / optim_iter

    def reduce_lr_on_plateau(self, loss):
        if len(self.eval_losses) < self.patience: 
            self.eval_losses.append(loss)
            return

        count = 0
        for i in range(1, self.patience + 1):
            if self.eval_losses[-i] <= loss:
                count += 1
        
        print(f'lr patience: {count}')
        
        if count >= self.patience:
            self.lr_multiplier = self.lr_multiplier * 0.9

            print(f'Reducing lr by: {self.lr_multiplier}')

        self.eval_losses.append(loss)
