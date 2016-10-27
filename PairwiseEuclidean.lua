require 'nn'

local Pairwise = torch.class("nn.PairwiseEuclidean", "nn.Module")

function PairwiseEuclidean:updateOutput(input)
    batchSize = input:size(1)
    dim = input:size(2)
    input_norm = torch.cmul(input, input):sum(2):repeatTensor(1, batchSize)
    self.output = input_norm + input_norm:transpose(1, 2) - 2 * input * input:transpose(1, 2)
    return self.output
end

function PairwiseEuclidean:updateGradInput(input, gradOutput)
    if self.gradInput then
        self.gradInput:resizeAs(input)
        B = gradOutput + gradOutput:transpose(1, 2)
        self.gradInput = 2 * (B:sum(2):repeatTensor(1, input:size(2)):cmul(input) - B * input)
    end
    return self.gradInput
end
