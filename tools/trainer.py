import torch
import torch.nn as nn
import logging
import os
import pdb
import torch.nn.functional as F
import torch.distributed as dist
from tools.utils.build_layer import build_smplh_mesh, Pelvis_norm
from tools.utils.evaluation import evaluate
from tools.utils.loss import L_ca

def train(opt, dict, train_loader, train_sampler,val_loader, val_dataset, model, logger, device, rank):

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(opt.save_checkpoint_path, "train.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def log_string(str):
        screen_logger.info(str)
        print(str)

    batches_train, batches_val = len(train_loader), len(val_loader)
    if rank ==0 :
        log_string(f'train_batch:{batches_train} | val_batch:{batches_val}')

    loss_ca = L_ca().to(device)
    loss_ce = nn.CrossEntropyLoss().to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(pg, lr=dict['lr_rate'], betas=(0.9, 0.999), eps=1e-8, weight_decay=dict['decay_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=dict['Epoch'], eta_min=1e-6)

    best_current = {'AUC':0, 'aIOU':0, 'SIM':0, 'precision':0, 'Recall':0, 'f1':0, 'geo':100000,'MSE':100000}
    print(f"rank {rank} start training...")
    for epoch in range(dict['Epoch']):
        if rank ==0 :
            log_string(f'{epoch + 1} Start -------')
        train_sampler.set_epoch(epoch)
        model = model.train()
        loss_sum = 0
        for i, data_info in enumerate(train_loader):

            optimizer.zero_grad()
            img = data_info['img'].to(device)
            B = img.size(0)
            contact_fine_gt, contact_coarse_gt = data_info['contact']['contact_fine'].to(device), data_info['contact']['contact_mid'].to(device)
            C_o = data_info['obj_curvature']
            C_h = data_info['hm_curvature'].to(device)
            logits_labels = data_info['logits']
            H,_ = build_smplh_mesh(data_info['human'])
            H = H.to(device)
            H, pelvis = Pelvis_norm(H, device)
            mvm_mask = data_info['mvm_mask'].to(device)
            sphere_center = data_info['sphere_center'].to(device)
            distance_gt = torch.norm(sphere_center - pelvis, dim=-1)
            sphere_center = sphere_center - pelvis
            _, pelvis_norm = Pelvis_norm(H, device)

            pair_num = len(data_info['Pts'])
            temp_loss = 0
            for pair in range(pair_num):
                O = data_info['Pts'][pair].float().to(device)
                affordance_gt = data_info['aff_gt'][pair].float().unsqueeze(dim=-1).to(device)
                C_o = C_o[pair].to(device)
                logits_label = logits_labels[pair].to(device)
                pre_contact, pre_affordance, pre_spatial, logits, varphi = model(img, O, H, C_h, C_o)
                contact_coarse, contact_fine = pre_contact[0], pre_contact[1]
                distance = torch.norm(pre_spatial - pelvis_norm, dim=-1)
 
                loss_c = loss_ca(contact_coarse, contact_coarse_gt) + loss_ca(contact_fine, contact_fine_gt)
                loss_a = loss_ca(pre_affordance, affordance_gt)
                loss_s = loss_ce(logits, logits_label) + varphi
                loss_p = F.mse_loss(pre_spatial, sphere_center, reduction='none').mean(-1).sum() + F.mse_loss(distance, distance_gt, reduction='sum')
                temp_loss += (dict['w1']*loss_c + dict['w2']*loss_a + dict['w3']*loss_s + dict['w4']*loss_p)
            if rank ==0 :
                print(f'Iteration {i}/{batches_train} | loss: {temp_loss.item()}')
            temp_loss.backward()
            loss_sum += temp_loss.item()
            optimizer.step()
        
        avg_loss = loss_sum / (batches_train*pair_num)
        if rank ==0 :
            log_string(f'Epoch: {epoch + 1} : loss: {avg_loss}')
        if(epoch % 1 ==0):
            model = model.eval()
            val_loss, best_results = val(opt, dict, epoch, val_dataset, val_loader, model, best_current, loss_ca, loss_ce, logger, device, batches_val)
            if rank ==0 :
                log_string(f'Epoch: {epoch + 1} : val_loss: {val_loss}')
                if 'AUC' in best_results:
                    best_current['AUC'], best_current['aIOU'], best_current['SIM'] = best_results['AUC'], best_results['aIOU'], best_results['SIM']
                    best_model_path = opt.save_checkpoint_path + 'AUC_best.pt'
                    torch.save(model.module.state_dict(), best_model_path)
                    log_string(f'AUC best saved in {best_model_path}')
                if 'f1' in best_results:
                    best_current['f1'] = best_results['f1']
                    best_current['precision'] = best_results['precision']
                    best_current['Recall'] = best_results['Recall']
                    best_current['geo'] = best_results['geo']
                    best_model_path = opt.save_checkpoint_path + 'F1_best.pt'
                    torch.save(model.module.state_dict(), best_model_path)
                    log_string(f'F1 best saved in {best_model_path}')
                if 'MSE' in best_results:
                    best_current['MSE'] = best_results['MSE']
                    best_model_path = opt.save_checkpoint_path + 'Spatial_best.pt'
                    torch.save(model.module.state_dict(), best_model_path)
                    log_string(f'Spatial best saved in {best_model_path}')
        scheduler.step()
    if rank==0 :
        log_string(f'Best Results----AUC:{best_current["AUC"]} | aIOU:{best_current["aIOU"]} | SIM:{best_current["SIM"]} | \
                Precision:{best_current["precision"]} | Recall:{best_current["Recall"]} | F1:{best_current["f1"]} | geo:{best_current["geo"]} \
                | MSE:{best_current["MSE"]}')

def val(opt, dict, epoch, val_dataset, val_loader, model, best_current, loss_ca, loss_ce, logger, device, batches_val):
    
    best_results = {}
    loss_sum = 0

    pr_aff, gt_aff = [], []
    pr_contact, gt_contact = [], []
    pr_spatial, gt_spatial = [], []

    aff_preds = torch.zeros((len(val_dataset), 2048, 1))
    aff_targets = torch.zeros((len(val_dataset), 2048, 1))
    contact_preds = torch.zeros((len(val_dataset), 6890, 1))
    contact_targets = torch.zeros((len(val_dataset), 6890, 1))

    with torch.no_grad():
        for i, data_info in enumerate(val_loader):
            img = data_info['img'].to(device)
            B = img.size(0)
            contact_fine_gt, contact_coarse_gt = data_info['contact']['contact_fine'].to(device), data_info['contact']['contact_mid'].to(device)
            C_o = data_info['obj_curvature'].to(device)
            C_h = data_info['hm_curvature'].to(device)
            logits_labels = data_info['logits']
            H,_ = build_smplh_mesh(data_info['human'])
            H = H.to(device)
            H, pelvis = Pelvis_norm(H, device)
            mvm_mask = data_info['mvm_mask'].to(device)
            sphere_center = data_info['sphere_center'].to(device)
            distance_gt = torch.norm(sphere_center - pelvis, dim=-1)
            sphere_center = sphere_center - pelvis
            _, pelvis_norm = Pelvis_norm(H, device)

            O = data_info['Pts'].float().to(device)
            affordance_gt = data_info['aff_gt'].float().unsqueeze(dim=-1).to(device)

            logits_label = logits_labels.to(device)
            pre_contact, pre_affordance, pre_spatial, logits, varphi = model(img, O, H, C_h, C_o)
            contact_coarse, contact_fine = pre_contact[0], pre_contact[1]
            distance = torch.norm(pre_spatial - pelvis_norm, dim=-1)

            loss_c = loss_ca(contact_coarse, contact_coarse_gt) + loss_ca(contact_fine, contact_fine_gt)
            loss_a = loss_ca(pre_affordance, affordance_gt)
            loss_s = loss_ce(logits, logits_label) + varphi
            loss_p = F.mse_loss(pre_spatial, sphere_center, reduction='none').mean(-1).sum() + F.mse_loss(distance, distance_gt, reduction='sum')
            temp_loss = (dict['w1']*loss_c + dict['w2']*loss_a + dict['w3']*loss_s + dict['w4']*loss_p)
            loss_sum += temp_loss.item()

            #gather:
            cur_aff = [torch.ones_like(pre_affordance) for _ in range(dist.get_world_size())]
            cur_aff_gt = [torch.ones_like(affordance_gt) for _ in range(dist.get_world_size())]
            cur_contact = [torch.ones_like(contact_fine) for _ in range(dist.get_world_size())]
            cur_contact_gt = [torch.ones_like(contact_fine) for _ in range(dist.get_world_size())]
            cur_center = [torch.ones_like(pre_spatial) for _ in range(dist.get_world_size())]
            cur_center_gt = [torch.ones_like(sphere_center) for _ in range(dist.get_world_size())]

            dist.all_gather(cur_aff, pre_affordance)
            dist.all_gather(cur_aff_gt, affordance_gt)
            dist.all_gather(cur_contact, contact_fine)
            dist.all_gather(cur_contact_gt, contact_fine_gt)
            dist.all_gather(cur_center, pre_spatial)
            dist.all_gather(cur_center_gt, sphere_center)

            pr_aff.extend(cur_aff)
            gt_aff.extend(cur_aff_gt)
            pr_contact.extend(cur_contact)
            gt_contact.extend(cur_contact_gt)
            pr_spatial.extend(cur_center)
            gt_spatial.extend(cur_center_gt)

        aff_preds, aff_targets = torch.cat(pr_aff, 0), torch.cat(gt_aff, 0)
        contact_preds, contact_targets = torch.cat(pr_contact, 0), torch.cat(gt_contact, 0)
        spatial_preds, spatial_gt = torch.cat(pr_spatial, 0), torch.cat(gt_spatial, 0)

    AUC_, IOU_, SIM_, precision_, recall_, F1_, geo_fn, geo_fp = evaluate(contact_preds, contact_targets, aff_preds, aff_targets)
    MSE = ((spatial_preds - spatial_gt)**2).mean().cpu().detach().numpy()

    if(AUC_ > best_current['AUC']):
        best_results['AUC'] = AUC_
        best_results['aIOU'] = IOU_
        best_results['SIM'] = SIM_

    if(F1_ > best_current['f1']):
        best_results['precision'] = precision_
        best_results['Recall'] = recall_
        best_results['f1'] = F1_
        best_results['geo'] = geo_fp

    if(MSE < best_current['MSE']):
        best_results['MSE'] = MSE
    
    logger.append([int(epoch+1), AUC_, IOU_, SIM_, precision_, recall_, F1_ , geo_fn, geo_fp, MSE])

    avg_loss = loss_sum / batches_val
    return avg_loss, best_results

if __name__ == '__main__':
    pass