from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from utils import load_data, load_graph, normalize_adj, numpy_to_torch
from GNN import GNNLayer
from evaluation import eva


class MLP_L(nn.Module):

    def __init__(self, n_mlp):
        super(MLP_L, self).__init__()
        self.wl = Linear(n_mlp, 4)

    def forward(self, mlp_in):

        weight_output = F.softmax(F.leaky_relu(self.wl(mlp_in)), dim=1)
        return weight_output


class MLP_1(nn.Module):

    def __init__(self, n_mlp):
        super(MLP_1, self).__init__()
        self.w1 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w1(mlp_in)), dim=1)
        return weight_output


class MLP_2(nn.Module):

    def __init__(self, n_mlp):
        super(MLP_2, self).__init__()
        self.w2 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w2(mlp_in)), dim=1)
        return weight_output


class MLP_3(nn.Module):

    def __init__(self, n_mlp):
        super(MLP_3, self).__init__()
        self.w3 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w3(mlp_in)), dim=1)
        return weight_output


class MLP_ZQ(nn.Module):

    def __init__(self, n_mlp):
        super(MLP_ZQ, self).__init__()
        self.w_ZQ = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w_ZQ(mlp_in)), dim=1)
        return weight_output

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        r_e1 = F.relu(self.enc_1(x))
        r_e2 = F.relu(self.enc_2(r_e1))
        r_e3 = F.relu(self.enc_3(r_e2))
        r = self.z_layer(r_e3)

        r_d1 = F.relu(self.dec_1(r))
        r_d2 = F.relu(self.dec_2(r_d1))
        r_d3 = F.relu(self.dec_3(r_d2))
        x_bar = self.x_bar_layer(r_d3)

        return x_bar, r_e1, r_e2, r_e3, r


class Adiin(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1):
        super(Adiin, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_1, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_z)

        self.gnn_6 = GNNLayer(n_z, n_dec_1)
        self.gnn_7 = GNNLayer(n_dec_1, n_dec_2)
        self.gnn_8 = GNNLayer(n_dec_2, n_dec_3)
        self.gnn_9 = GNNLayer(n_dec_3, n_input)

        self.agnn_z = GNNLayer(3010, n_clusters)

        self.mlp = MLP_L(3010)

        # attention on [z_i, h_i]
        self.mlp1 = MLP_1(2 * n_enc_1)
        self.mlp2 = MLP_2(2 * n_enc_2)
        self.mlp3 = MLP_3(2 * n_enc_3)

        self.mlp_ZQ = MLP_ZQ(2*n_clusters)
        self.mlp_ZQ1 = MLP_ZQ(2*n_z)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.s = nn.Sigmoid()
        self.n_clusters=n_clusters
        self.n_z=n_z

        # degree
        self.v = v

    def forward(self, x, adj):
        # content information learning module
        x_bar, r_e1, r_e2, r_e3, r = self.ae(x)
        x_array = list(np.shape(x))
        n_x = x_array[0]

        z1 = self.gnn_1(x, adj)
        h1 = r_e1

        p1 = self.mlp1(torch.cat((h1, z1), 1))
        p1 = F.normalize(p1, p=2)
        p11 = torch.reshape(p1[:, 0], [n_x, 1])
        p12 = torch.reshape(p1[:, 1], [n_x, 1])
        p11_broadcast = p11.repeat(1, 500)
        p12_broadcast = p12.repeat(1, 500)

        z2 = self.gnn_2( p11_broadcast.mul(z1)+p12_broadcast.mul(h1), adj)
        h2 = r_e2
        p2 = self.mlp2(torch.cat((h2, z2), 1))
        p2 = F.normalize(p2, p=2)
        p21 = torch.reshape(p2[:, 0], [n_x, 1])
        p22 = torch.reshape(p2[:, 1], [n_x, 1])
        p21_broadcast = p21.repeat(1, 500)
        p22_broadcast = p22.repeat(1, 500)

        z3 = self.gnn_3(z2, adj)
        h3 = F.relu(self.ae.enc_3(p21_broadcast.mul(z2)+p22_broadcast.mul(h2)))
        p3 = self.mlp3(torch.cat((h3, z3), 1))  # self.mlp3(h2)
        p3 = F.normalize(p3, p=2)
        p31 = torch.reshape(p3[:, 0], [n_x, 1])
        p32 = torch.reshape(p3[:, 1], [n_x, 1])
        p31_broadcast = p31.repeat(1, 2000)
        p32_broadcast = p32.repeat(1, 2000)
        z = self.gnn_4( p31_broadcast.mul(z3)+p32_broadcast.mul(h3), adj)
        h = self.ae.z_layer(h3)

        z_i = z + r
        z_l = torch.spmm(adj, z_i)
        dec_z1 = self.gnn_6(z, adj, active=True)
        dec_z2 = self.gnn_7(dec_z1, adj, active=True)
        dec_z3 = self.gnn_8(dec_z2, adj, active=True)
        z_hat = self.gnn_9(dec_z3, adj, active=True)

        # adj_hat = self.s(torch.mm(z, z.t()))
        adj_hat = self.s(torch.mm(z_hat, z_hat.t()))

        w = self.mlp(torch.cat((z1, z2, z3, z), 1))
        w = F.normalize(w, p=2)

        w0 = torch.reshape(w[:, 0], [n_x, 1])
        w1 = torch.reshape(w[:, 1], [n_x, 1])
        w2 = torch.reshape(w[:, 2], [n_x, 1])
        w3 = torch.reshape(w[:, 3], [n_x, 1])

        # 2️⃣ [Z+H]
        tile_w0 = w0.repeat(1, 500)
        tile_w1 = w1.repeat(1, 500)
        tile_w2 = w2.repeat(1, 2000)
        tile_w3 = w3.repeat(1, 10)


        # 2️⃣ concat
        net_output = torch.cat((tile_w0.mul(z1), tile_w1.mul(z2), tile_w2.mul(z3), tile_w3.mul(z)), 1)
        net_output = self.agnn_z(net_output, adj, active=False)

        pred = F.softmax(net_output, dim=1)

        # Joint-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z_l.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        q1 = 1.0 / (1.0 + torch.sum(torch.pow(r.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        return x_bar, z_hat, adj_hat, q, q1, z, r, z_l, pred


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def num_net_parameter(net):
    all_num = sum(i.numel() for i in net.parameters())
    print ('[The network parameters]', all_num)

def train_Adiin(dataset):
    model = Adiin(500, 500, 2000, 2000, 500, 500,
                  n_input=args.n_input,
                  n_z=args.n_z,
                  n_clusters=args.n_clusters,
                  v=1.0).to(device)


    print(model)
    print("alpha_1: ", args.alpha1, " alpha_2: ", args.alpha2, " alpha_3: ", args.alpha3,
          " alpha_4: ", args.alpha4, " alpha_5: ", args.alpha5, " alpha_6: ", args.alpha6)

    print(num_net_parameter(model))

    acc_reuslt_q1 = []
    nmi_result_q1 = []
    ari_result_q1 = []
    f1_result_q1 = []

    acc_reuslt_q = []
    nmi_result_q = []
    ari_result_q = []
    f1_result_q = []

    acc_reuslt_p = []
    nmi_result_p = []
    ari_result_p = []
    f1_result_p = []

    optimizer = Adam(model.parameters(), lr=args.lr)


    if args.name == 'amap' or args.name == 'pubmed' or args.name == 'cora' or args.name == 'phy' or args.name == 'wiki':
        load_path = "data/" + args.name + "/" + args.name
        adj = np.load(load_path+"_adj.npy", allow_pickle=True)
        adj = normalize_adj(adj, self_loop=True, symmetry=False)
        adj = numpy_to_torch(adj, sparse=True).to(torch.device("cuda")) # opt.args.device = torch.device("cuda" if opt.args.cuda else "cpu")
    else:
        adj = load_graph(args.name, args.k)
        adj = adj.cuda()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')

    for epoch in range(args.epo):

        if epoch % 1 == 0:
            # update_interval
            _, _, _, q, q1, _, _, _, pred = model(data, adj)
            tmp_q = q.data
            p = target_distribution(tmp_q)
            tmp_q1 = q1.data
            p1 = target_distribution(tmp_q1)

            res = tmp_q1.cpu().numpy().argmax(1)
            acc, nmi, ari, f1 = eva(y, res, str(epoch) + 'Q1')
            acc_reuslt_q1.append(acc)
            nmi_result_q1.append(nmi)
            ari_result_q1.append(ari)
            f1_result_q1.append(f1)

            res1 = tmp_q.cpu().numpy().argmax(1)
            acc, nmi, ari, f1 = eva(y, res1, str(epoch) + 'Q')
            acc_reuslt_q.append(acc)
            nmi_result_q.append(nmi)
            ari_result_q.append(ari)
            f1_result_q.append(f1)

            res2 = pred.data.cpu().numpy().argmax(1)
            acc, nmi, ari, f1 = eva(y, res2, str(epoch) + 'Pred')
            acc_reuslt_p.append(acc)
            nmi_result_p.append(nmi)
            ari_result_p.append(ari)
            f1_result_p.append(f1)


        x_bar, z_hat, adj_hat, q, q1, z, r, z_l, pred = model(data, adj)


        # if (epoch+1) % 10 == 0:
        #     h_cpu = r.cpu().detach().numpy()
        #     h_tsne = tsne.fit_transform(h_cpu)
        #     labels = q.detach().cpu().numpy().argmax(1)
        #
        #     # 可视化降维后的数据
        #     plot_embedding(h_tsne, labels, str(epoch),'AIJSS')
        #     print("graph "+str(epoch)+" is ok.")

        ae_loss = F.mse_loss(x_bar, data)
        w_loss = F.mse_loss(z_hat, torch.spmm(adj, data))
        a_loss = F.mse_loss(adj_hat, adj.to_dense())
        qp_loss = F.kl_div(q.log(), p, reduction='batchmean')
        q1q_loss = F.kl_div(q1.log(), q, reduction='batchmean')
        q1p1_loss = F.kl_div(q1.log(), p1, reduction='batchmean')
        qz_loss = F.kl_div(q.log(), pred, reduction='batchmean')
        q1z_loss = F.kl_div(q1.log(),pred,reduction='batchmean')

        # if args.name=='dblp':
        #     loss = ae_loss + 1 *( w_loss + a_loss) \
        #     + 10 * qp_loss + args.alpha4 * q1p1_loss + args.alpha5 * q1q_loss \
        #     + 0.1 * q1z_loss
        # elif args.name=='acm':
        #     loss = ae_loss + 1 *( w_loss + a_loss) \
        #     + 10 * qp_loss + args.alpha4 * q1p1_loss + args.alpha5 * q1q_loss \
        #     + 0.1 * q1z_loss
        # elif args.name=='cite':
        #     loss = ae_loss + 1 *( w_loss + a_loss) \
        #     + 10 * qp_loss + args.alpha4 * q1p1_loss + args.alpha5 * q1q_loss \
        #     + 0.01 * q1z_loss
        # elif args.name=='hhar':
        #     loss = ae_loss + 1 *( w_loss + a_loss) \
        #     + 10 * qp_loss + args.alpha4 * q1p1_loss + args.alpha5 * q1q_loss \
        #     + 0.01 * q1z_loss
        # elif args.name=='usps':
        #     loss = ae_loss + 1 *( w_loss + a_loss) \
        #     + 10 * qp_loss + args.alpha4 * q1p1_loss + args.alpha5 * q1q_loss \
        #     + 0.01 * q1z_loss
        # elif args.name=='reut':
        #     loss = ae_loss + 1 *( w_loss + a_loss) \
        #     + 10 * qp_loss + args.alpha4 * q1p1_loss + args.alpha5 * q1q_loss \
        #     + 0.01 * q1z_loss

        loss = ae_loss + args.alpha3 * qp_loss + args.alpha4 * q1p1_loss+args.alpha6*q1q_loss+0.1*qz_loss
        # loss = ae_loss + (w_loss + a_loss) + \
        #        args.alpha7* (qp_loss + q1q_loss) +\
        #        args.alpha8*(q1p1_loss + q1z_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # Sample noise as discriminator ground truthR
        # z = Variable(Tensor(np.random.normal(0, 1, (adj.shape[0], 10))))
        #
        # # Measure discriminator's ability to classify real from generated samples
        # real_loss = adversarial_loss(discriminator(z), valid)
        # fake_loss = adversarial_loss(discriminator(a_r.detach()), fake)
        # d_loss = 0.5 * (real_loss + fake_loss)
        #
        # d_loss.backward()
        # optimizer_D.step()

    print("#################" + args.name + "##" + str(args.alpha3) + "##" + str(args.alpha4) + "##" + str(args.alpha6) + "####################")
    acc_max = np.max(acc_reuslt_q1)
    nmi_max = nmi_result_q1[acc_reuslt_q1.index(np.max(acc_reuslt_q1))]
    ari_max = ari_result_q1[acc_reuslt_q1.index(np.max(acc_reuslt_q1))]
    f1_max = f1_result_q1[acc_reuslt_q1.index(np.max(acc_reuslt_q1))]
    epoch_max =acc_reuslt_q1.index(np.max(acc_reuslt_q1))

    acc_max_q = np.max(acc_reuslt_q)
    nmi_max_q = nmi_result_q[acc_reuslt_q.index(np.max(acc_reuslt_q))]
    ari_max_q = ari_result_q[acc_reuslt_q.index(np.max(acc_reuslt_q))]
    f1_max_q = f1_result_q[acc_reuslt_q.index(np.max(acc_reuslt_q))]
    epoch_max_q = acc_reuslt_q.index(np.max(acc_reuslt_q))

    acc_max_p = np.max(acc_reuslt_p)
    nmi_max_p = nmi_result_p[acc_reuslt_p.index(np.max(acc_reuslt_p))]
    ari_max_p = ari_result_p[acc_reuslt_p.index(np.max(acc_reuslt_p))]
    f1_max_p = f1_result_p[acc_reuslt_p.index(np.max(acc_reuslt_p))]
    epoch_max_p = acc_reuslt_p.index(np.max(acc_reuslt_p))
    print('the pred result of this iter:\nacc:{:.4f},\nnmi:{:.4f},\nari:{:.4f},\nf1:{:.4f},\nmax_epoch:{}\n'.format(
        round(acc_max_p, 5),
        round(nmi_max_p, 5),
        round(ari_max_p, 5),
        round(f1_max_p, 5),
        epoch_max_p))
    return acc_max, nmi_max, ari_max, f1_max, acc_max_q, nmi_max_q, ari_max_q, f1_max_q, acc_max_p, nmi_max_p, ari_max_p, f1_max_p

if __name__ == "__main__":

    iters = 10  #

    acc = []
    nmi = []
    ari = []
    f1 = []
    acc_q = []
    nmi_q = []
    ari_q = []
    f1_q = []
    acc_p = []
    nmi_p = []
    ari_p = []
    f1_p = []

    for iter_num in range(iters):
        print('the {}-th iter'.format(iter_num + 1))
        parser = argparse.ArgumentParser(
            description='train',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--name', type=str, default='pubmed')
        parser.add_argument('--k', type=int, default=3)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--n_clusters', default=3, type=int)
        parser.add_argument('--n_z', default=10, type=int)
        parser.add_argument('--pretrain_path', type=str, default='pkl')
        parser.add_argument('--adlr', type=float, default=1e-4)
        parser.add_argument('--hidden1', type=int, default=64)
        parser.add_argument('--hidden2', type=int, default=16)
        parser.add_argument('--alpha3', type=float, default=0.1)  # The parameters of qp_loss
        parser.add_argument('--alpha4', type=float, default=0.1)  # The parameters of q1p1_loss
        parser.add_argument('--alpha6', type=float, default=1)
        parser.add_argument('--epo', type=int, default=100)
        parser.add_argument('--save_dir', type=str, default='tsne')

        args = parser.parse_args()
        args.cuda = torch.cuda.is_available()
        print("use cuda: {}".format(args.cuda))
        device = torch.device("cuda" if args.cuda else "cpu")

        args.pretrain_path = 'data/{}.pkl'.format(args.name)
        dataset = load_data(args.name)

        if args.name == 'usps':
            args.n_clusters = 10
            args.n_input = 256
            args.alpha3 = 10
            args.alpha4 = 1
            args.epo = 300

        if args.name == 'hhar':
            args.k = 5
            args.n_clusters = 6
            args.n_input = 561
            args.alpha3 = 0.01
            args.alpha4 = 1

        if args.name == 'reut':
            args.lr = 1e-4
            args.n_clusters = 4
            args.n_input = 2000
            args.k = 10
            args.alpha3 = 0.01
            args.alpha4 = 1
            args.epo = 100

        if args.name == 'acm':
            args.k = None
            args.lr = 5e-5
            args.n_clusters = 3
            args.n_input = 1870
            args.alpha3 = 1
            args.alpha4 = 1
            args.epo = 200

        if args.name == 'dblp':
            args.k = None
            args.lr = 2e-3
            args.n_clusters = 4
            args.n_input = 334
            args.epo = 200

        if args.name == 'cite':
            args.lr = 1e-4
            args.k = None
            args.n_clusters = 6
            args.n_input = 3703
            args.alpha3 = 10
            args.alpha4 = 1
            args.alpha6 = 10
            args.epo = 150

        if args.name == 'cora':
            args.lr = 6e-4
            args.k = None
            args.n_clusters = 3
            args.alpha3 = 0.1
            args.alpha4 = 0.1
            args.n_input = 1433

        if args.name == 'amap':
            args.lr = 1e-4
            args.k = None
            args.n_clusters = 8
            args.alpha3 = 10
            args.alpha4 = 0.01
            args.n_input = 745

        if args.name == 'pubmed':
            args.lr = 1e-3
            args.k = None
            args.n_clusters = 3
            args.alpha3 = 0.1
            args.alpha4 = 0.01
            args.n_input = 500

        print(args)
        acc_max, nmi_max, ari_max, f1_max, accq, nmiq, ariq, f1q, accp, nmip, arip, f1p = train_Adiin(dataset)
        acc.append(acc_max)
        nmi.append(nmi_max)
        ari.append(ari_max)
        f1.append(f1_max)
        acc_q.append(accq)
        nmi_q.append(nmiq)
        ari_q.append(ariq)
        f1_q.append(f1q)
        acc_p.append(accp)
        nmi_p.append(nmip)
        ari_p.append(arip)
        f1_p.append(f1p)

    print('PRED MEAN:acc, nmi, ari, f1: \n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}\n'.format(np.mean(acc_p), np.mean(nmi_p),
                                                                                   np.mean(ari_p),
                                                                                   np.mean(f1_p)))
    print('PRED MAX:acc, nmi, ari, f1: \n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}\n'.format(np.max(acc_p), np.max(nmi_p),
                                                                                  np.max(ari_p),
                                                                                  np.max(f1_p)))
