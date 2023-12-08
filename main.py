from time import time
import argparse
from scNetwork import scNetwork
from Generate import*
from Preprocess import *

# for repeatability
setup_seed(1111)
# filename="Mouse_bladder_cell"#16
# filename="Mouse_ES_cell"#4
# filename="Worm_neuron_cell"#10
# filename="10X_PBMC"#8
# filename="Young"#11
# filename="Plasschaert"#8
# filename="Wang_Lung"#2
# filename="Qx_Spleen"#5
# filename="Qx_Trachea"#5
filename="Chen"#46

if __name__ == "__main__":
    file1 = ["Mouse_bladder_cell", "Mouse_ES_cell", "Worm_neuron_cell", "10X_PBMC"]
    parser = argparse.ArgumentParser(description='train',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='D:/Code/Dataset/'+filename+'/data.h5')#Path of data
    parser.add_argument('--maxiter', default=150, type=int)# Maximum number of iterations
    parser.add_argument('--pretrain_epochs', default=300, type=int)# Number of iterations during pre-training
    parser.add_argument('--gamma', default=[0.5,1,0.1,0.1], type=float)#Parameters of  Loss
    parser.add_argument('--sigma', default=1.5, type=float)#Coefficient of random noise
    parser.add_argument('--update_interval', default=1, type=int)#Update frequency of cluster tags
    parser.add_argument('--tol', default=0.001, type=float)#Threshold, used to determine whether to stop the training
    # parser.add_argument('--ae_weights', default=None),#filename+'AE_weights.pth.tar', Pre-training weight path
    parser.add_argument('--ae_weights', default= '512/'+filename+'_AEweights.pth.tar',)  # None, Utilize this line after pre-training to save time.
    parser.add_argument('--save_dir', default='results/'+filename)#directory to save model weights during the training stage
    parser.add_argument('--ae_weight_file', default='512/'+filename+'_AEweights.pth.tar')#file name to save model weights after the pretraining stage
    parser.add_argument('--final_latent_file', default='final/'+filename+'_latent_file.txt')# Save the file name of the final encoding representation
    parser.add_argument('--predict_label_file', default='final/'+filename+'_pred_labels.txt')# File name to save the clustering result
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    if filename in file1:
        data_mat = h5py.File(args.data_file, 'r')
        x = np.array(data_mat['X']).astype('float')
        y = np.array(data_mat['Y'])
        data_mat.close()
    else:
        x, y = prepro(args.data_file)
        x = np.ceil(x)#.astype('int')

    adata=sc.AnnData(x, dtype=x.dtype)
    adata.obs['Group'] = adjust_labels(y)
    adata = read_dataset(adata,transpose=False,test_split=False,copy=True)
    adata = normalize(adata,size_factors=True,normalize_input=True,logtrans_input=True)
    input_size = adata.n_vars

    print("Args:",args)
    print("Adata.X.shape:",adata.X.shape)
    print("Y.shape:",y.shape)
    adjust_y = adjust_labels(y)

    model = scNetwork(input_dim=adata.n_vars,
                      z_dim=32,encodeLayer=[256, 128], decodeLayer=[128,256],
                      sigma=args.sigma, gamma1=args.gamma[0],gamma2=args.gamma[1],
                      gamma3=args.gamma[2],gamma4=args.gamma[3],device=args.device)
    print(str(model))

    t0 = time()
    if args.ae_weights is None:#pretrain
        model.pretrain_autoencoder(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors,
                                   batch_size=args.batch_size, epochs=args.pretrain_epochs,
                                   ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> Loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> No checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError
    print('Pretraining time: %d seconds.' % int(time() - t0))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    #Semi, need to be optimized
    ratio = 0.1
    generate = 10000
    print('Ratio of labeled cells: {:.1%}, Generate Triplets : {}.' .format(ratio, generate))

    #Clustering
    n_clusters=int(max(adjust_y)-min(adjust_y)+1)
    y_pred, nmi, ari, acc, ami= model.fit(
        X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors,
        n_clusters=n_clusters, generate = generate,margin=0.1, ratio=ratio,
        init_centroid=None,y_pred_init=None, y=adjust_y, batch_size=args.batch_size,
        num_epochs=args.maxiter, update_interval=args.update_interval,
        tol=args.tol, save_dir=args.save_dir)
    print('Total time: %d seconds.' % int(time() - t0))
    print('Evaluating cells: NMI= %.4f, ARI= %.4f, ACC= %.4f, AMI= %.4f.' % (nmi, ari, acc, ami))

    final_latent = model.encodeBatch(torch.tensor(adata.X, dtype=torch.float32)).cpu().numpy()
    np.savetxt(args.final_latent_file, final_latent, delimiter=",")
    np.savetxt(args.predict_label_file, y_pred, delimiter=",", fmt="%i")
