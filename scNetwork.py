import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Layers import ZINBLoss, MeanAct, DispAct
import numpy as np
from sklearn.cluster import KMeans
import math
import os
from sklearn import metrics
from Generate import cluster_acc,pairwise_select

def buildNetwork(layers, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)

class scNetwork(nn.Module):
    def __init__(self, input_dim, z_dim,
                 encodeLayer=[], decodeLayer=[],
                 activation="relu",
                 sigma=1., alpha=1., gamma=1.,
                 device="cuda"):
        super(scNetwork, self).__init__()
        self.z_dim = z_dim
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.encoder = buildNetwork([input_dim]+encodeLayer, activation=activation)
        self.decoder = buildNetwork([z_dim]+decodeLayer, activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())

        self.zinb_loss = ZINBLoss().to(self.device)
        self.to(device)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q
    
    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def forwardAE(self, x):
        h = self.encoder(x+torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        return z0, _mean, _disp, _pi

    def forward(self, x):
        h = self.encoder(x+torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        q = self.soft_assign(z0)
        return z0, q, _mean, _disp, _pi

    def encodeBatch(self, X, batch_size=256):
        self.eval()
        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs = Variable(xbatch).to(self.device)
            z, _, _, _= self.forwardAE(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded.to(self.device)

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
        kldloss = kld(p, q)
        return kldloss

    def triplet_loss(self, anchor, positive, negative, margin_constant):
        # loss = max(d(anchor, negative) - d(anchor, positve) + margin, 0), margin > 0
        # d(x, y) = q(x) * q(y)
        negative_dis = torch.sum(anchor * negative, dim=1)
        positive_dis = torch.sum(anchor * positive, dim=1)
        margin = margin_constant * torch.ones(negative_dis.shape).cuda()
        diff_dis = negative_dis - positive_dis
        penalty = diff_dis + margin
        triplet_loss = 1*torch.max(penalty, torch.zeros(negative_dis.shape).cuda())
        return torch.mean(triplet_loss)

    def weighted_cross_entropy_loss(self,y_true, y_prob,yratio):
        n_classes =y_prob.shape[1]#the number of type
        class_counts = torch.bincount(y_true, minlength=n_classes)# celltype proportions and normalize
        ratio = class_counts.float() / torch.sum(class_counts).float()
        class_ratio=torch.where( ratio!= 0, 1/ratio, torch.zeros_like(ratio))
        w = class_ratio / torch.sum(class_ratio)*yratio#weight
        y_onehot = torch.nn.functional.one_hot(y_true, n_classes) # Transform the real label vector into a one-hot matrix
        wce_loss = -torch.mean(torch.sum(w * y_onehot * torch.log(y_prob), dim=1))
        return wce_loss
    def  pairwise_contrastive_loss(self,y_true, y_prob):
        norms_p = torch.norm(y_prob, dim=1, keepdim=True, p=1.5)
        unit_p=y_prob/norms_p
        unit_p_T=torch.transpose(unit_p,0,1)
        S=torch.matmul(unit_p,unit_p_T)#similarity matrix
        n_samples = y_true.shape[0]# the number of cell
        R = (y_true.unsqueeze(0) == y_true.unsqueeze(1)).int()
        epsilon = 5e-6
        S = torch.clamp(S, epsilon, 1 - epsilon)
        pwc_loss= -1/n_samples/n_samples*(torch.sum(R * torch.log(S) + (1 - R) * torch.log(1 - S)))
        return pwc_loss

    def pretrain_autoencoder(self, X, X_raw, size_factor,
                             batch_size=256, lr=0.001, epochs=400,
                             ae_save=True, ae_weights='AE_weights.pth.tar'):
        self.train()
        dataset = TensorDataset(torch.Tensor(X), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("---Pretraining stage---")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        for epoch in range(epochs):
            loss_val = 0
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch).to(self.device)
                x_raw_tensor = Variable(x_raw_batch).to(self.device)
                sf_tensor = Variable(sf_batch).to(self.device)
                _, mean_tensor, disp_tensor, pi_tensor = self.forwardAE(x_tensor)
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=sf_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val += loss.item() * len(x_batch)
            print('Pretrain epoch %3d, ZINB loss: %.8f' % (epoch+1, loss_val/X.shape[0]))

        if ae_save:
            torch.save({ 'ae_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                ae_weights)

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit(self, X, X_raw, size_factor, n_clusters,anchor, positive, negative,
            init_centroid=None, y=None, y_pred_init=None,lr=1., batch_size=256,
            num_epochs=10, update_interval=1, tol=1e-3, save_dir=""):
        self.train()
        print("---Clustering stage---")
        X = torch.tensor(X, dtype=torch.float)
        X_raw = torch.tensor(X_raw, dtype=torch.float)
        size_factor = torch.tensor(size_factor, dtype=torch.float)
        self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim).to(self.device))
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.85)

        print("Initializing cluster centers with Soft-Kmeans.")
        if init_centroid is None:
            kmeans = KMeans(n_clusters, init='k-means++')
            data = self.encodeBatch(X)# Encode the input sample
            self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())# Returns the prediction label
            self.y_pred_last = self.y_pred
            self.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float))
        else:
            self.mu.data.copy_(torch.tensor(init_centroid, dtype=torch.float))
            self.y_pred = y_pred_init# Returns the predict label
            self.y_pred_last = self.y_pred
        print('Initializing soft-Kmeans: ' )

        num = X.shape[0]#the number of clusters
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))#sub cluster batches
        tri_num = anchor.shape[0]  # the number of triple
        tri_num_batch = int(math.ceil(1.0 * anchor.shape[0] / batch_size))  # sub triple batches

        final_acc, final_nmi, final_ari, final_epoch , final_ami= 0, 0, 0, 0,0
        loss_history = []
        for epoch in range(num_epochs):
            if epoch > 0:
                loss_history.append(total_loss.item() / num)
            if epoch % update_interval == 0:#Update the cluster center
                latent = self.encodeBatch(X.to(self.device))
                q = self.soft_assign(latent)#soft distribution matrix q
                p = self.target_distribution(q).data#target distribution matrix p
                self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()#upgrade label

                final_acc = acc = np.round(cluster_acc(y, self.y_pred), 5)
                final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                final_epoch = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                final_ami = ami = np.round(metrics.adjusted_mutual_info_score(y, self.y_pred), 5)
                print('\t\tClustering %d: NMI= %.4f, ARI= %.4f,ACC= %.4f,AMI= %.4f' % (epoch, nmi, ari, acc,ami))

                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float) / num
                if (epoch > 0 and delta_label < tol) or epoch%10 == 0:# save current model
                    self.save_checkpoint({'epoch': epoch+1,
                            'state_dict': self.state_dict(),
                            'mu': self.mu,
                            'y_pred': self.y_pred,
                            'y_pred_last': self.y_pred_last,
                            'y': y
                            }, epoch+1, filename=save_dir)

                self.y_pred_last = self.y_pred
                if epoch>0:
                    # print("loss_history",loss_history)
                    if delta_label < tol: #check stop criterion
                        print('delta_label ', delta_label, '< tol ', tol)
                        print("Reach tolerance threshold. Stopping training.")
                        break
                    elif epoch > 5:
                        # print(np.mean(abs(np.diff(loss_history[-6:]))))
                        if np.mean(abs(np.diff(loss_history[-6:]))) < tol:
                            print("Reach tolerance threshold. Stopping running.")
                            break

            # train 1 epoch for clustering loss
            zinb_loss_val = 0.0
            cluster_loss_val = 0.0
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                xrawbatch = X_raw[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sfbatch = size_factor[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                optimizer.zero_grad()
                inputs = Variable(xbatch).to(self.device)
                rawinputs = Variable(xrawbatch).to(self.device)
                sfinputs = Variable(sfbatch).to(self.device)
                target = Variable(pbatch).to(self.device)
                zbatch, qbatch, meanbatch, dispbatch, pibatch = self.forward(inputs)
                cluster_loss = self.cluster_loss(target, qbatch)
                zinb_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)
                loss = cluster_loss*self.gamma + zinb_loss
                loss.backward()
                optimizer.step()
                cluster_loss_val += cluster_loss.item() * len(inputs)
                zinb_loss_val += zinb_loss.item() * len(inputs)

            # train 1 epoch for triple loss
            triplet_loss = 0.0
            triplet_loss_val = 0.0
            if epoch % update_interval == 0:
                for tri_batch_idx in range(tri_num_batch):
                    px1 = X[anchor[tri_batch_idx * batch_size: min(tri_num, (tri_batch_idx + 1) * batch_size)]]
                    px2 = X[positive[tri_batch_idx * batch_size: min(tri_num, (tri_batch_idx + 1) * batch_size)]]
                    px3 = X[negative[tri_batch_idx * batch_size: min(tri_num, (tri_batch_idx + 1) * batch_size)]]
                    optimizer.zero_grad()
                    inputs1 = Variable(px1).to(self.device)
                    inputs2 = Variable(px2).to(self.device)
                    inputs3 = Variable(px3).to(self.device)
                    z1, q_t1, _, _, _ = self.forward(inputs1)
                    z2, q_t2, _, _, _ = self.forward(inputs2)
                    z3, q_t3, _, _, _ = self.forward(inputs3)
                    loss = self.triplet_loss(q_t1, q_t2, q_t3, 0.1)
                    triplet_loss += loss.data
                    loss.backward()
                    optimizer.step()
                    triplet_loss_val += triplet_loss * len(inputs1)

            # train 1 epoch for  weighted cross entropy loss
            wce_loss=0.0
            wce_loss_val = 0.0
            labels = y
            # torch.cuda.empty_cache()
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                ybatch=labels[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                yratio=len(ybatch)/len(y)
                ytensor = torch.as_tensor(ybatch, dtype=torch.long)  # torch
                optimizer.zero_grad()
                inputs = Variable(xbatch).to(self.device)
                y_true = Variable(ytensor).to(self.device)
                _, q_w, _, _, _ = self.forward(inputs)
                loss = self.weighted_cross_entropy_loss(y_true, q_w,yratio)
                wce_loss+=loss.data
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                wce_loss_val += wce_loss* len(ybatch)

            #train 1 epoch for pairwise contrastive loss
            X_sub, y_sub = pairwise_select(X, y, 0.8)
            yy = torch.as_tensor(y_sub, dtype=torch.float)
            optimizer.zero_grad()
            inputss = Variable(X_sub).to(self.device)
            _, q_c, _, _, _ = self.forward(inputss)
            yy_prob = Variable(q_c).to('cpu')
            yy_true = Variable(yy).to('cpu')
            loss = self.pairwise_contrastive_loss(yy_true, yy_prob)
            pwc_loss =loss.data
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            pwc_loss_val = pwc_loss.item() * len(y_sub)

            print("Epoch%3d:\n\t\t Soft_Assign Loss: %.8f ZINB Loss: %.8f" % (
                epoch + 1, cluster_loss_val / num, zinb_loss_val / num), end=' ')
            if tri_num_batch > 0:
                print("Triplet Loss: %.8f" % (triplet_loss_val / num))
            print("\t\tWeightedCrossEntropy Loss: %.8f PairwiseContrastive Loss: %.8f"
                  % (wce_loss_val / num,pwc_loss_val/ num))
                  # % (wce_loss_val / num,0))
                  # % (0,pwc_loss_val/ num))
            total_loss = cluster_loss_val * self.gamma + zinb_loss_val + triplet_loss_val*0.001\
                         +pwc_loss_val*0.001+ wce_loss_val*0.001
            print("\t\t***Total Loss: %.8f" % (total_loss / num))
        return self.y_pred, final_acc, final_nmi, final_ari, final_epoch,final_ami