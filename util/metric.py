import torch


def IoUGPU(output, target, K):
    # 'K' classes, output and target sizes are
    # N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    return area_intersection, area_output, area_target


def FMeasureGPU(output, target, eps=1e-20, beta=0.3):
    target = (target > 0) * 1.0
    output = (output > 0) * 1.0

    t = torch.sum(target)
    p = torch.sum(output)
    tp = torch.sum(target * output)
    recall = tp / (t + eps)
    precision = tp / (p + eps)
    f_score = (1 + beta) * precision * recall / (beta * precision + recall +
                                                 eps)

    return f_score
