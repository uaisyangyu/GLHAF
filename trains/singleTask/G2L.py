import logging
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import FancyBboxPatch
from numpy.array_api import permute_dims
from torch import optim
from torch.ao.quantization.backend_config.utils import get_fusion_pattern_to_extra_inputs_getter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils import MetricsTop, dict_to_str
from trains.singleTask.HingeLoss import HingeLoss
from ..subNets.AlignNets import CMDLoss,TCA,uni_distill
from ..subNets.CLoss import HCL
from ..subNets.OT import OTLoss
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
logger = logging.getLogger('MMSA')


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    # 均方误差
    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


class G2L():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()  # 平均绝对误差损失MAE
        self.cosine = nn.CosineEmbeddingLoss()  # 余弦相似度损失Consine
        self.MSE = MSE()
        self.sim_loss = HingeLoss()  # 用于测量两个输入是否相似
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.CMDLoss = CMDLoss()
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        self.Liner=nn.Linear(dst_feature_dims*3,dst_feature_dims)
    def do_train(self, model, dataloader, return_epoch_results=False):

        # 0: DLF model
        params = model[0].parameters()

        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_DLF = model[0]
        net.append(net_DLF)
        model = net
        # 开始一个新的训练epoch，并将模型切换到训练模式
        while True:
            epochs += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()

            # 初始化当前epoch的训练损失，并使用tqdm显示训练进度条。
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with (tqdm(dataloader['train']) as td):
                for batch_data in td:

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()  # 梯度清零
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    # 将多模态数据输入模型，获取模型的输出
                    output = model[0](text, audio, vision)

                    # task loss
                    loss_task_all = self.criterion(output['output_logit'], labels)

                    loss_task_l = self.criterion(output['logits_l'], labels)
                    loss_task_v= self.criterion(output['logits_v'], labels)
                    loss_task_a = self.criterion(output['logits_a'], labels)
                    loss_task_c = self.criterion(output['logits_g'], labels)

                    # total MSA loss L_msa
                    loss_task =  loss_task_all+loss_task_a+loss_task_v+loss_task_l


                    # # reconstruction loss L_r
                    # loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                    # loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                    # loss_recon_a = self.MSE(output['recon_a'], output['origin_a'])
                    # loss_recon = loss_recon_l + loss_recon_v + loss_recon_a
                    #
                    # # specific loss L_s
                    # loss_sl_slr = self.MSE(output['s_l'].permute(1, 2, 0), output['s_l_r'])
                    # loss_sv_slv = self.MSE(output['s_v'].permute(1, 2, 0), output['s_v_r'])
                    # loss_sa_sla = self.MSE(output['s_a'].permute(1, 2, 0), output['s_a_r'])
                    # loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla

                    # ort loss L_o
                    if self.args.dataset_name == 'mosi':
                        num = 150
                    elif self.args.dataset_name == 'mosei':
                        num = 10

                    # local alignment loss(Token-level alignment)
                    g_fusion=output['g_fusion']
                    local_l,local_v,local_a=output['local_align_l'], output['local_align_v'], output['local_align_a']
                    # h_gl, h_gv, h_ga = output['h_gl'], output['h_gv'], output['h_ga']

                    #hard alignment loss
                    OT = OTLoss(eps=0.1, max_iter=50).to(self.args.device)
                    loss_ot=OT(local_l,local_a)+OT(local_l,local_v)
                    #loacl(token) alignment loss
                    tca=TCA(temperature=0.1).cuda()
                    loss_vt,loss_ta,loss_va=tca(local_l,local_v,local_a)
                    #global-local alignment loss
                    # hcl=HCL(input_dim=150).to(self.args.device)
                    # loss_cl=hcl(g_fusion,local_l)+hcl(g_fusion,local_v)+hcl(g_fusion,local_a)
                    #pairwis alignment


                    # triplet margin loss L_m
                    c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']
                    ids, feats = [], []
                    for i in range(labels.size(0)):
                        feats.append(c_l[i].view(1, -1))
                        feats.append(c_v[i].view(1, -1))
                        feats.append(c_a[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                    feats = torch.cat(feats, dim=0)
                    ids = torch.cat(ids, dim=0)
                    loss_sim = self.sim_loss(ids, feats)



                    # lva = torch.cat([output['l_proj'],output['v_proj'],output['a_proj']],dim=1)
                    # lva=self.Liner(lva)
                    # loss_ud = uni_distill(output['g_proj'], lva.detach())
                    #global alignment loss
                    #g_align_l,g_align_v,g_align_a = output['g_align_l'], output['g_align_v'], output['g_align_a']
                    #loss_g_align=self.CMDLoss(g_align_l, g_align_v)+self.CMDLoss(g_align_a, g_align_v)+self.CMDLoss(g_align_l, g_align_a)


                    combined_loss = loss_task +(loss_vt+loss_ta+loss_va)*0.1+loss_sim*0.01

                    combined_loss.backward()

                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters())

                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += combined_loss.item()

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN -({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            test_results = self.do_test(model[0], dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            # save each epoch model
            torch.save(model[0].state_dict(), './pt/' + str(self.args.dataset_name) + '_' + str(epochs) + '.pth')
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                model_save_path = './pt/G2L' + str(self.args.dataset_name) + '.pth'
                torch.save(model[0].state_dict(), model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="TEST", return_sample_results=True,visualize_tsne=False,n_clusters=7):

        model.eval()
        y_pred, y_true = [], []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
                'Feature_r':[],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    output = model(text, audio, vision)
                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if return_sample_results:
                        feature_r =output['last_hs_proj'].cpu().numpy()#(24,150)
                        # feature_r = np.transpose(feature_r, (1, 0, 2))
                        # features["Feature_a"].append(output['local_align_a'].cpu().numpy())
                        # features["Feature_v"].append(output['local_align_v'].cpu().numpy())
                        # features["Feature_t"].append(output['local_align_l'].cpu().numpy())
                        # features["Feature_f"].append(output['g_fusion'].cpu().numpy())
                        features["Feature_r"].append(feature_r)
                    if 'ids' in batch_data:
                        ids.extend(batch_data['ids'])
                    sample_results.append(output['output_logit'].cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        # all_logits=torch.cat(y_pred).numpy()
        # print(all_logits.shape())
        eval_results= self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            features_hr = np.concatenate(features["Feature_r"], axis=0)
            features["Feature_r"]=features_hr#(686,150)
            # for k in features.keys():
            #     features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels
        # print(pred.shape)
        pred_7=np.round(pred.numpy().flatten()).astype(int)
        true_7 = np.round(true.numpy().flatten()).astype(int)
        # print(len(true_7))
        # print(len(eval_results['Features']['Feature_r']))
        # print(eval_results['Features']['Feature_r'].shape)
        #
        if visualize_tsne:
            features_2d=eval_results['Features']['Feature_r']
            if features_2d.shape[1] > 50:
                print("使用PCA预降维到30维...")
                pca = PCA(n_components=50, random_state=42)
                features_2d = pca.fit_transform(features_2d)
                print(f"PCA降维后特征形状: {features_2d.shape}")

                # 应用t-SNE
            tsne = TSNE(n_components=2, perplexity=50,
                        max_iter=1000, random_state=42, verbose=1)
            features_tsne = tsne.fit_transform(features_2d)


            print(f"t-SNE降维后特征形状: {features_tsne.shape}")
            plt.figure(figsize=(7, 5))

            #获取唯一标签和颜色映射
            unique_labels = np.unique(true_7)
            n_classes = len(unique_labels)

            # 使用seaborn的调色板
            colors = [  # 蓝色（清晰、信任）
                      '#629edb',  #
                      '#AFD2F3',  # 绿色（冷静、中性）
                      '#417BBC',
                      '#E9F4FA',  # 红色（表达负面情绪如愤怒）
                      '#F9E078',  # 青色（理性、科技感、也适合中性或“惊讶”等类）
                      '#FDF5D0',  # 紫色（代表快乐、复杂性）
                      '#FAE179',  # 棕紫（忧郁、沮丧感）
                      ]
            palette = sns.color_palette("hsv", n_classes)

            # 绘制散点图
            for i, label in enumerate(unique_labels):
                mask = (true_7 == label)
                plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                            c=[colors[i]], label=f'Class {label}',
                            alpha=0.7, s=30)

            plt.title(f'Complete', fontsize=12)
            plt.xlabel('')
            plt.ylabel('')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)

            # save_path = f'tsne_visualization_{mode.lower()}_{self.args.model_name}.png'
            plt.tight_layout()
            # plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # print(f"t-SNE可视化图像已保存到: {save_path}")

            # 可选：显示图像（在Jupyter notebook中）
            plt.show()

            # 关闭图形以释放内存
            plt.close()

        return eval_results