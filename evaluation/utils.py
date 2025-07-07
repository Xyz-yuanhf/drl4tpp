import torch

def augment_inference(instance):

    loc_with_depot = torch.cat((instance['depot'][:, None, :], instance['loc']), dim=1)
    aug_loc_ls = []
    for a in range(0, 8):
        aug_loc = loc_with_depot.clone()
        if a % 2 == 0:
            aug_loc[..., 0], aug_loc[..., 1] = \
                loc_with_depot[..., 1].clone(), loc_with_depot[..., 0].clone()
        if a % 4 in [1, 3]:
            aug_loc[..., 1] = 1 - aug_loc[..., 1]
        if a % 4 in [2, 3]:
            aug_loc[..., 0] = 1 - aug_loc[..., 0]
        aug_loc_ls.append(aug_loc)
    loc_with_depot = torch.cat(aug_loc_ls, dim=0)
    instance['depot'] = loc_with_depot[:, 0, :].contiguous()
    instance['loc'] = loc_with_depot[:, 1:, :].contiguous()
    instance['supply_data'] = instance['supply_data'].expand(8, -1, -1)
    instance['price_data'] = instance['price_data'].expand(8, -1, -1)
    instance['supply_sparse'] = torch.cat([instance['supply_sparse'] for i in range(8)], dim=0)
    return instance