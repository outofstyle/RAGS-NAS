import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from torch.utils.data import DataLoader, Dataset
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as pylab
import torch
from sklearn.decomposition import KernelPCA
import os
import numpy as np
import torch.nn as nn

def plot_surface(x1, x2, y, query, save = None, xmax = 7.0, zlim = True,
                 ):
    fig = plt.figure(figsize = (6,6), dpi = 90)
    ax = fig.add_subplot(projection='3d')
    ax.view_init(azim = 131, elev = 35)
    p = ax.plot_surface(x1, x2, y, cmap='Spectral_r', alpha = 0.9)
   
    ax.grid(True)
    x_major_locator = MultipleLocator(7.0)
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator = MultipleLocator(7.0)
    ax.yaxis.set_major_locator(y_major_locator)
    
    z_major_locator = MultipleLocator(1.0)
    ax.zaxis.set_major_locator(z_major_locator)
      
    # fig.tight_layout()
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_zlabel('Accuracy')
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-xmax, xmax)
    ax.set_zlim(0,)
    # cbar.set_label('Values')
    '''ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('k')
    ax.yaxis.pane.set_edgecolor('k')
    ax.zaxis.pane.set_edgecolor('k')'''
    #plt.title(title)
    #if save is not None:
    plt.savefig(os.path.join('/home/liugroup/ranker/AG-Net-main/search/', 'pca-{}_40000_x7_3D.png'.format(query)),
            bbox_inches='tight', dpi = 400)
    plt.close()
        
def plot_scatter(x1, x2, y, query, save = None, xmax = 7.0, zlim = True,
                 ):
    fig = plt.figure(figsize = (6,6), dpi = 90)
    # ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot()
    # ax.view_init(azim = -30, elev = 35)
    
    values, indices = y.sort()
    
    p = ax.scatter(x1[indices], x2[indices],c = list(range(len(y))), cmap='Spectral', alpha = 0.8, s = 1.0)
   
    ax.grid(True)
    x_major_locator = MultipleLocator(7.0)
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator = MultipleLocator(7.0)
    ax.yaxis.set_major_locator(y_major_locator)
    
    # z_major_locator = MultipleLocator(0.3)
    # ax.zaxis.set_major_locator(z_major_locator)
      
    # fig.tight_layout()
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    # ax.set_zlabel('Accuracy')
    ax.set_xlim(-7.0, 7.0)
    ax.set_ylim(-7.0, 7.0)
    # ax.set_xlim(-0.5, 0.5)
    # ax.set_ylim(-0.5, 0.5)

    # cbar.set_label('Values')
    '''ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('k')
    ax.yaxis.pane.set_edgecolor('k')'''
    # ax.zaxis.pane.set_edgecolor('k')
    #plt.title(title)
    #if save is not None:
    plt.savefig(os.path.join('/home/liugroup/ranker/AG-Net-main/search/', 'pca-{}_40000_x7.png'.format(query)),
                bbox_inches='tight', dpi = 400)
    plt.close()
    

# In[PCA]
def pca_vis(pop, pop_acc, query):
    out_path = '/home/liugroup/ranker/AG-Net-main/search/'
    kpca = KernelPCA(2, kernel = 'linear')
    feature = [x.detach().unsqueeze(0) for x in pop]
    #test_acc = [acc.detach().cpu() for acc in pop_acc]
    # feature = np.stack(feature, axis=0)
    # test_acc = np.stack(test_acc, axis=0).flatten()
    x_list = torch.cat(feature, dim = 0)
    y_list = torch.cat(pop_acc, dim = 0)
    all_x = kpca.fit_transform(x_list.numpy())


    plot_scatter(all_x[:,0], all_x[:,1], y_list, query=query, xmax = 7.0)
    plt.show()
    # plt.savefig(os.path.join('/home/liugroup/ranker/AG-Net-main/search/', 'pca-{}.png'.format(query)),
    #             bbox_inches='tight')
    plt.close()

    x_list = torch.tensor(all_x).cuda()
    y_list = torch.tensor(y_list).cuda()
    class Train_Dataset(Dataset):
        def __init__(self, x_list, y_list):
            super().__init__()
            self.x_list = x_list
            self.y_list = y_list
                
        def __getitem__(self, idx):
            return self.x_list[idx], self.y_list[idx]
                
        def __len__(self):
            return len(self.x_list)

    train_dataset = Train_Dataset(x_list, y_list)
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)

    class MLP(nn.Module):
        def __init__(self, input_dim = 2, hidden_dim = 256, output_dim = 1, hidden_layer = 4):
            super(MLP, self).__init__()
            self.layers = nn.ModuleList()
            self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim),                                        
                                            nn.LeakyReLU()))    
            for l in range(hidden_layer):
                self.layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim, bias = False),
                    nn.ReLU()))
            self.layers.append(nn.Linear(hidden_dim, output_dim))
            
        def forward(self, x):
            h = x
            for l in range(len(self.layers)):
                h = self.layers[l](h)
            return h

    net = MLP(input_dim = 2).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4, betas = (0.0,0.5), weight_decay = 0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = float(50), eta_min = 0.0)
    mse = nn.MSELoss(reduction = 'mean')
    import tqdm 
    for epoch in tqdm.tqdm(range(50)):
        for step, (x, y) in enumerate(train_loader):
            p = net(x[:]).squeeze()
            loss = mse(0.01*y,p)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(loss)
        scheduler.step()

    net = net.cuda()

    scope = torch.arange(-7.0, 7.0, 1.0).cuda()
    x1, x2 = torch.meshgrid([scope, scope])
    inputs = torch.cat([x1.reshape(-1).unsqueeze(1), x2.reshape(-1).unsqueeze(1)], dim = 1)
    with torch.no_grad():
        pred = net(inputs)
    pred = pred.reshape(x1.shape)

    plot_surface(x1.cpu().numpy(), x2.cpu().numpy(), pred.cpu().numpy(), query=query, xmax = 7.0)
    plt.show()
    # plt.savefig(os.path.join('/home/liugroup/ranker/AG-Net-main/search/', 'pca-{}.png'.format(query)),
    #             bbox_inches='tight')
    plt.close()