import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Layers import ZINBLoss, MeanAct, DispAct
from sklearn.cluster import KMeans
import os
from sklearn import metrics
from Generate import *

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
    def __init__(self, input_dim, z_dim,encodeLayer=[], decodeLayer=[],activation="relu",
                 sigma=1., alpha=1., gamma1=1.,gamma2=1.,gamma3=1.,gamma4=1.,device="cuda"):
        super(scNetwork, self).__init__()
        self.z_dim = z_dim
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.gamma4 = gamma4
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
        norms_p = torch.norm(y_prob, dim=1, keepdim=True, p=2)
        unit_p=y_prob/norms_p
        unit_p_T=torch.transpose(unit_p,0,1)
        S=torch.matmul(unit_p,unit_p_T)#similarity matrix
        n_samples = y_true.shape[0]# the number of cell
        R = (y_true.unsqueeze(0) == y_true.unsqueeze(1)).int()
        epsilon = 1e-6
        S = torch.clamp(S, epsilon, 1 - epsilon)
        pwc_loss= -1/n_samples/n_samples*(torch.sum(R * torch.log(S) + (1 - R) * torch.log(1 - S)))
        return pwc_loss

    def pretrain_autoencoder(self, X, X_raw, size_factor,batch_size=256, lr=0.001, epochs=100,
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

    def fit(self, X, X_raw, size_factor, n_clusters, generate, margin, ratio,
            init_centroid=None, y=None, y_pred_init=None,lr=1., batch_size=256,
            num_epochs=10, update_interval=1, tol=1e-3, save_dir=""):
        self.train()
        print("---Clustering stage---")
        X = torch.tensor(X, dtype=torch.float)
        X_raw = torch.tensor(X_raw, dtype=torch.float)
        size_factor = torch.tensor(size_factor, dtype=torch.float)
        self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim).to(self.device))
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.9)

        print("Initializing cluster centers with Soft-Kmeans:")
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

        num = X.shape[0]#the number of clusters
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))#sub cluster batches

        final_acc, final_nmi, final_ari,  final_ami= 0, 0, 0, 0
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
                final_ari = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                final_ami = ami = np.round(metrics.adjusted_mutual_info_score(y, self.y_pred), 5)
                print('\t\tClustering %d: NMI= %.4f, ARI= %.4f,ACC= %.4f,AMI= %.4f' % (epoch, nmi, ari, acc,ami))

                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float) / num
                if (epoch > 0 and delta_label < tol) or epoch%25 == 0:# save current model
                    self.save_checkpoint({
                        'epoch': epoch+1,
                        'state_dict': self.state_dict(),
                        'mu': self.mu,
                        'y_pred': self.y_pred,
                        'y_pred_last': self.y_pred_last,
                        'y': y}, epoch+1, filename=save_dir)

                self.y_pred_last = self.y_pred
                if epoch>0:
                    if delta_label < tol: #check stop criterion
                        print('delta_label ', delta_label, '< tol ', tol)
                        print("Reach tolerance threshold. Stopping training.")
                        break
                    elif epoch > 5:
                        if np.mean(abs(np.diff(loss_history[-6:]))) < tol*5:
                            print("Reach tolerance threshold. Stopping running.")
                            break

            # train 1 epoch for clustering loss
            zinb_loss_val = cluster_loss_val = 0.0
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
                _, qbatch, meanbatch, dispbatch, pibatch = self.forward(inputs)
                cluster_loss = self.cluster_loss(target, qbatch)
                zinb_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)
                loss1 = cluster_loss*self.gamma1 + zinb_loss
                loss1.backward()
                optimizer.step()
                cluster_loss_val += cluster_loss * len(inputs)
                zinb_loss_val += zinb_loss * len(inputs)

            # Semi
            X_sub, y_sub , p_sub = cell_select(X, y, p,ratio)
            anchor, positive, negative = generate_triplets(y_sub, generate=generate)
            tri_num = anchor.shape[0]  # the number of triple
            tri_num_batch = int(math.ceil(1.0 * anchor.shape[0] / batch_size))  # sub triple batches

            # train 1 epoch for triple loss
            triplet_loss_val =0.0
            for tri_batch_idx in range(tri_num_batch):
                xt1 = X_sub[anchor[tri_batch_idx * batch_size: min(tri_num, (tri_batch_idx + 1) * batch_size)]]
                xt2 = X_sub[positive[tri_batch_idx * batch_size: min(tri_num, (tri_batch_idx + 1) * batch_size)]]
                xt3 = X_sub[negative[tri_batch_idx * batch_size: min(tri_num, (tri_batch_idx + 1) * batch_size)]]
                optimizer.zero_grad()
                inputs1 = Variable(xt1).to(self.device)
                inputs2 = Variable(xt2).to(self.device)
                inputs3 = Variable(xt3).to(self.device)
                _, q_t1, _, _, _ = self.forward(inputs1)
                _, q_t2, _, _, _ = self.forward(inputs2)
                _, q_t3, _, _, _ = self.forward(inputs3)
                triplet_loss = self.triplet_loss(q_t1, q_t2, q_t3, margin)
                loss2 = triplet_loss*self.gamma2
                loss2.requires_grad_(True)
                loss2.backward()
                optimizer.step()
                triplet_loss_val += triplet_loss * len(y_sub)

            #train 1 epoch for pairwise contrastive loss , weighted cross entropy loss
            yy = torch.as_tensor(y_sub, dtype=torch.long)
            optimizer.zero_grad()
            yy_prob = Variable(p_sub).to(self.device)
            yy_true = Variable(yy).to(self.device)
            pwc_loss = self.pairwise_contrastive_loss(yy_true, yy_prob)
            loss31 = pwc_loss * self.gamma3
            loss31.requires_grad_(True)

            wce_loss = self.weighted_cross_entropy_loss(yy_true, yy_prob ,1)
            loss32 = wce_loss * self.gamma4
            loss32.requires_grad_(True)
            loss3 = loss31 + loss32
            loss3.backward()
            optimizer.step()
            pwc_loss_val = pwc_loss * len(y_sub)
            wce_loss_val = wce_loss * len(y_sub)

            print("Epoch%3d:\n\t\tZINB Loss: %.5f DeepClustering Loss: %.5f " % (
                epoch + 1 , zinb_loss_val / num , cluster_loss_val / num), end=' ')
            if tri_num_batch > 0:
                print("Triplet Loss: %.5f" % (triplet_loss_val / num))
            print("\t\tWeightedCrossEntropy Loss: %.5f PairwiseContrastive Loss: %.5f"
                  % (wce_loss_val / num,pwc_loss_val/ num))
                  # % (wce_loss_val / num,0))
                  # % (0,pwc_loss_val/ num))
            total_loss = zinb_loss_val + cluster_loss_val * self.gamma1 + triplet_loss_val * self.gamma2\
                         + pwc_loss_val * self.gamma3 + wce_loss_val * self.gamma4
            print("\t\t***Total Loss: %.5f" % (total_loss / num))
        return self.y_pred, final_nmi, final_ari, final_acc, final_ami