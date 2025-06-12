import torch

class QuantizerFunc(torch.autograd.Function):

    @staticmethod
    def forward(self, input):
        #TODO: implement binarization
        input = input.detach().clone()
        input[input > 0.5] = 1.
        input[input <= 0.5] = 0.
        #mask = input <= 0.5
        return input

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


#binarization = QuantizerFunc.apply
