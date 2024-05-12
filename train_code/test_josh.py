# import os
# import sys

# import numpy as np

# import torch
# from torch.utils.data import DataLoader

# from configs.config_v1 import ConfigV1 as Config
# from network.network_utils import build_model
# from network.optimizer_utils import get_optimizer, get_scheduler
# from dataloaders import morph
# from utils.util import load_model, extract_features, find_kNN, get_absolute_score, select_pair_geometric, get_age_bounds, select_reference_global_regression, \
#     get_best_pairs_global_regression, get_results



# def main(cfg):
#     os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

#     net = build_model(cfg)

#     if torch.cuda.is_available():
#         torch.backends.cudnn.benchmark = True
#         net = net.cuda()

#     optimizer = get_optimizer(cfg, net)
#     lr_scheduler = get_scheduler(cfg, optimizer)

#     if cfg.dataset_name == 'morph':
#         test_ref_dataset = morph.MorphRef(cfg=cfg, tau=cfg.tau, dataset_dir=cfg.dataset_root)
#         test_dataset = morph.MorphTest(cfg=cfg, dataset_dir=cfg.dataset_root)

#         test_ref_loader = DataLoader(test_ref_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
#         test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)

#     else:
#         raise ValueError(f'Undefined database ({cfg.dataset_name}) has been given')

#     if cfg.load:
#         load_model(cfg, net, optimizer=optimizer, load_optim_params=False)

#     if lr_scheduler:
#        lr_scheduler.step()

#     net.eval()
#     test(cfg, net, test_ref_loader, test_loader)

# if __name__ == "__main__":
#     cfg = Config()
#     main(cfg)