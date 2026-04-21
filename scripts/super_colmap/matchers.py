import torch


# Mutual nearest neighbors matcher for L2 normalized descriptors.
def mutual_nn_matcher(descriptors1, descriptors2, device="cuda"):
    des1 = torch.from_numpy(descriptors1).to(device)
    des2 = torch.from_numpy(descriptors2).to(device)
    sim = des1 @ des2.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    matches = torch.stack([ids1[mask], nn12[mask]]).t()
    # print("sim:", sim)
    # print("nn12:", nn12)
    # print("len_nn12:", len(nn12))
    # print("nn21:", nn21)
    # print("len_nn21:", len(nn21))
    # print("ids1:", ids1)
    # print("mask:", mask)
    # print("matches:", matches)
    # exit()

    return matches.data.cpu().numpy()


def mut_nn_matcher_indices_1(descriptors1, descriptors2, indices, device="cuda"):
    des1 = torch.from_numpy(descriptors1).to(device)
    des2 = torch.from_numpy(descriptors2).to(device)
    sim = des1 @ des2.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    matches = torch.stack([ids1[mask], nn12[mask]]).t()

    # 将indices转换为张量类型
    indices_tensor = torch.tensor(indices, device=device)

    # 过滤出matches中第二个元素在indices中的匹配对
    filtered_matches = matches[torch.isin(matches[:, 1], indices_tensor)]

    return filtered_matches.data.cpu().numpy()


def mut_nn_matcher_indices_0(descriptors1, descriptors2, indices, device="cuda"):
    des1 = torch.from_numpy(descriptors1).to(device)
    des2 = torch.from_numpy(descriptors2).to(device)
    sim = des1 @ des2.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    matches = torch.stack([ids1[mask], nn12[mask]]).t()

    # 将indices转换为张量类型
    indices_tensor = torch.tensor(indices, device=device)

    # 过滤出matches中第一个元素在indices中的匹配对
    filtered_matches = matches[torch.isin(matches[:, 0], indices_tensor)]

    return filtered_matches.data.cpu().numpy()


# Symmetric Lowe's ratio test matcher for L2 normalized descriptors.
def ratio_matcher(descriptors1, descriptors2, ratio=0.8, device="cuda"):
    des1 = torch.from_numpy(descriptors1).to(device)
    des2 = torch.from_numpy(descriptors2).to(device)
    sim = des1 @ des2.t()

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn12 = nns[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn21 = nns[:, 0]

    # Symmetric ratio test.
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio)

    # Final matches.
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)

    return matches.data.cpu().numpy()


# Mutual NN + symmetric Lowe's ratio test matcher for L2 normalized descriptors.
def mutual_nn_ratio_matcher(descriptors1, descriptors2, ratio=0.8, device="cuda"):
    des1 = torch.from_numpy(descriptors1).to(device)
    des2 = torch.from_numpy(descriptors2).to(device)
    sim = des1 @ des2.t()

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN and match similarity.
    nn12 = nns[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn21 = nns[:, 0]

    # Mutual NN + symmetric ratio test.
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = torch.min(ids1 == nn21[nn12], torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio))

    # Final matches.
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)

    return matches.data.cpu().numpy()