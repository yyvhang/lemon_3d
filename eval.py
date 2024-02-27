import torch
import numpy as np
import pdb
import random
import os
import pandas as pd
import yaml
import argparse
from tools.models.model_LEMON_d import LEMON
from dataset_utils.dataset_3DIR import _3DIR
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tools.utils.build_layer import build_smplh_mesh, Pelvis_norm
from numpy import nan

def eval_process(val_dataset, val_loader, model, device):
    
    Obejct_ = []
    aff_preds = torch.zeros((len(val_dataset), 2048, 1))
    aff_targets = torch.zeros((len(val_dataset), 2048, 1))
    spatial_mse = torch.zeros((len(val_dataset), 3))
    contact_preds = torch.zeros((len(val_dataset), 6890, 1))
    contact_targets = torch.zeros((len(val_dataset), 6890, 1))

    Points_path = []
    aff_num = 0
    with torch.no_grad():
        for i, data_info in enumerate(val_loader):
            img = data_info['img'].to(device)
            B = img.size(0)
            file_paths = data_info['img_path']
            for iter in range(B):
                obj = file_paths[iter].split('/')[2]
                Obejct_.append(obj)

            contact_fine = data_info['contact']['contact_fine'].to(device)
            vertex,_ = build_smplh_mesh(data_info['human'])
            vertex = vertex.to(device)
            vertex, pelvis = Pelvis_norm(vertex, device)
            pts = data_info['Pts'].float().to(device)
            obj_curvature = data_info['obj_curvature'].to(device)
            hm_curvature = data_info['hm_curvature'].to(device)
            
            sphere_center = data_info['sphere_center'].to(device)
            sphere_center = sphere_center - pelvis
            affordance_gt = data_info['aff_gt'].float().unsqueeze(dim=-1).to(device)
            pred_contact, pred_affordance, spatial, _, _ = model(img, pts, vertex, hm_curvature, obj_curvature)
            temp_mse = (spatial-sphere_center)**2
            pred_coarse, pred_fine = pred_contact[0], pred_contact[1]

            pts_path = data_info['Pts_path']
            for path_ in pts_path:
                Points_path.append(path_)

            pred_num = pred_fine.shape[0]
            aff_preds[aff_num : aff_num+pred_num, :, :] = pred_affordance
            aff_targets[aff_num : aff_num+pred_num, :, :] = affordance_gt
            spatial_mse[aff_num : aff_num+pred_num, :] = temp_mse

            contact_preds[aff_num : aff_num+pred_num, :, :] = pred_fine
            contact_targets[aff_num : aff_num+pred_num, :, :] = contact_fine
            aff_num += pred_num
    evaluate(contact_preds, contact_targets, aff_preds, aff_targets, spatial_mse, Points_path, Obejct_)

def evaluate(contact_pred, contact_gt, aff_pred, aff_gt, spatial_mse, pts_path, Object_):

    '''
    contact:[B, 6890, 1]
    affordance:[B, 2048, 1]
    '''
    metrics = {'Metrics':['Precision','Recall','F1','geo','AUC','aIOU','SIM','MSE']}
    data_df = pd.DataFrame(metrics)
    object_list = ['Earphone', 'Baseballbat', 'Tennisracket', 'Bag', 'Motorcycle', 'Guitar', 
                    'Backpack', 'Chair', 'Knife', 'Bicycle', 'Umbrella', 'Keyboard','Scissors', 
                    'Bottle', 'Bowl', 'Surfboard', 'Mug', 'Suitcase', 'Vase', 'Skateboard', 'Bed']

    def set_round(data):
        return np.around(data, 4)
    '''
    Object: [AUC], [aIOU], [SIM], [F1], [Precision], [Recall], [Spatial_MSE], [geo_error]
    '''

    for obj in object_list:
        exec(f'{obj} = [[], [], [], [], [], [], [], []]')
    dist_matrix = np.load('smpl_models/smpl_neutral_geodesic_dist.npy')
    dist_matrix = torch.tensor(dist_matrix).cuda()

    contact_pred = contact_pred.detach().numpy()
    contact_gt = contact_gt.detach().numpy()
    aff_pred = aff_pred.detach().numpy()
    aff_gt = aff_gt.detach().numpy()
    spatial_mse = spatial_mse.detach().numpy()

    AUC_aff = np.zeros((aff_gt.shape[0], aff_gt.shape[2]))
    IOU_aff = np.zeros((aff_gt.shape[0], aff_gt.shape[2]))

    SIM_matrix = np.zeros(aff_gt.shape[0])

    IOU_thres = np.linspace(0, 1, 20)
    num = contact_gt.shape[0]
    f1_avg = 0
    recall_avg = 0
    precision_avg = 0
    mse_avg = 0
    false_positive_dist_avg = 0
    false_negative_dist_avg = 0

    for b in range(num):
        #f1_score
        contact_tp_idx = contact_gt[b, contact_pred[b,:,0]>=0.5, 0]
        contact_tp_num = np.sum(contact_tp_idx)
        contact_precision_denominator = np.sum(contact_pred[b, :, 0]>=0.5)
        contact_recall_denominator = np.sum(contact_gt[b, :, 0])

        precision_contact = contact_tp_num / (contact_precision_denominator + 1e-10)
        recall_contact = contact_tp_num / (contact_recall_denominator + 1e-10)
        f1_contact = 2 * precision_contact * recall_contact / (precision_contact + recall_contact + 1e-10)

        gt_columns = dist_matrix[:, contact_gt[b, :, 0]==1] if any(contact_gt[b, :, 0]==1) else dist_matrix
        error_matrix = gt_columns[contact_pred[b, :, 0] >= 0.5, :] if any(contact_pred[b, :, 0] >= 0.5) else gt_columns

        false_positive_dist = error_matrix.min(dim=1)[0].mean()
        false_negative_dist = error_matrix.min(dim=0)[0].mean()

        object_cls = Object_[b]

        exec(f'{object_cls}[3].append({f1_contact})')
        exec(f'{object_cls}[4].append({precision_contact})')
        exec(f'{object_cls}[5].append({recall_contact})')
        exec(f'{object_cls}[7].append({false_positive_dist})')

        f1_avg += f1_contact
        precision_avg += precision_contact
        recall_avg += recall_contact
        false_positive_dist_avg += false_positive_dist
        false_negative_dist_avg += false_negative_dist

        #sim
        SIM_matrix[b] = SIM(aff_pred[b], aff_gt[b])
        exec(f'{object_cls}[2].append({SIM_matrix[b]})')

        #spatial mse
        temp_mse = spatial_mse[b].mean()
        mse_avg += temp_mse
        exec(f'{object_cls}[6].append({temp_mse})')

        #AUC_IOU
        aff_t_true = (aff_gt[b] >= 0.5).astype(int)
        aff_p_score = aff_pred[b]

        if np.sum(aff_t_true) == 0:
            AUC_aff[b] = np.nan
            IOU_aff[b] = np.nan
            obj_auc = AUC_aff[b]
            obj_iou = IOU_aff[b]
            exec(f'{object_cls}[0].append({obj_auc})')
            exec(f'{object_cls}[1].append({obj_iou})')
        else:
            try:
                auc_aff = roc_auc_score(aff_t_true, aff_p_score)
                AUC_aff[b] = auc_aff
            except ValueError:
                print(pts_path[b])
                AUC_aff[b] = np.nan

            temp_iou = []
            for thre in IOU_thres:
                p_mask = (aff_p_score >= thre).astype(int)
                intersect = np.sum(p_mask & aff_t_true)
                union = np.sum(p_mask | aff_t_true)
                temp_iou.append(1.*intersect/union)
            temp_iou = np.array(temp_iou)
            aiou = np.mean(temp_iou)
            IOU_aff[b] = aiou

            obj_auc = AUC_aff[b]
            obj_iou = IOU_aff[b]
            exec(f'{object_cls}[0].append({obj_auc})')
            exec(f'{object_cls}[1].append({obj_iou})')

    AUC_aff = set_round(np.nanmean(AUC_aff))
    IOU_aff = set_round(np.nanmean(IOU_aff))

    f1_avg = set_round(f1_avg / num)
    recall_avg = set_round(recall_avg / num)
    precision_avg = set_round(precision_avg / num)
    mse_avg = set_round(mse_avg / num)

    fp_error, fn_error = false_positive_dist_avg / num, false_negative_dist_avg / num
    geo_erro = fp_error
    AUC_ = set_round(AUC_aff)
    IOU_ = set_round(IOU_aff)
    SIM_ = set_round(np.mean(SIM_matrix))

    print('------Object-------')
    for i,obj in enumerate(object_list):
        aiou = set_round(np.nanmean(eval(obj)[1]))
        sim_ = set_round(np.mean(eval(obj)[2]))
        auc_ = set_round(np.nanmean(eval(obj)[0]))
        f1_ = set_round(np.mean(eval(obj)[3]))
        precision_ = set_round(np.mean(eval(obj)[4]))
        recall_ = set_round(np.mean(eval(obj)[5]))
        mse_ = set_round(np.mean(eval(obj)[6]))
        geo_ = set_round(np.mean(eval(obj)[7]))

        data_df.insert(i+1,obj,[np.round(precision_,2), np.round(recall_,2), np.round(f1_,2), \
        geo_*100, auc_*100, aiou*100, np.round(sim_,2), np.round(mse_,3)])

        print(f'{obj} | AUC:{auc_*100} | IOU:{aiou*100} | SIM:{sim_} | F1:{f1_} | Precision:{precision_} | Recall:{recall_} | geo:{geo_*100} | MSE:{mse_}')
    data_df.to_csv('eval_results.csv', mode='w', header=True,index=False)
    print('------ALL-------')
    print(f'Overall---AUC:{AUC_} | IOU:{IOU_} | SIM:{SIM_} | F1:{f1_avg} | Precision:{precision_avg} | Recall:{recall_avg} | MSE:{mse_avg} | geo:{geo_erro}')

def SIM(map1, map2, eps=1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)
    return dict

def run(opt, dict):
    val_dataset = _3DIR(dict['val_image'], dict['val_pts'], dict['human_3DIR'], dict['behave'], mode='val')
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8)
    model = LEMON(dict['emb_dim'], run_type='infer', device=opt.device)
    checkpoint = torch.load(dict['best_checkpoint'], map_location=opt.device)
    model.load_state_dict(checkpoint)
    model = model.to(opt.device)
    model = model.eval()

    eval_process(val_dataset, val_loader, model, opt.device)

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0', help='gpu device id')
    parser.add_argument('--use_gpu', type=str, default=True, help='whether or not use gpus')
    parser.add_argument('--yaml', type=str, default='config/eval.yaml', help='yaml path')
    parser.add_argument('--batch_size', type=int, default=12, help='batch_size')
    opt = parser.parse_args()
    dict = read_yaml(opt.yaml)
    seed_torch(42)
    run(opt, dict)