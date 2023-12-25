import torch
import numpy as np
import sys, os, random, argparse, pickle, tqdm
from datetime import datetime
import torch.nn.functional as F

from torch.utils.data import WeightedRandomSampler, Dataset
from torch_geometric.data import DataLoader
sys.path.insert(1, os.path.join(os.getcwd()))
from Generator import Generator_Decoder
from Measurements import Measurements
from Optimizer import Optimizer
from Checkpoint import Checkpoint
import Settings
import datasets.NASBench301 as NASBench301

'''ranker and EA'''
from search_EA.ranker import Listwise_Ranker
import torch.nn.functional as F
from loss_utils.loss import LambdaNDCGLoss2
from cmaes import CMA, get_warm_start_mgd, CMAwM
from acquisition import acquisition_fct
import copy
import sklearn.cluster as skc
from sklearn.cluster import cluster_optics_dbscan
from Setup import setup_logger, setup_seed

##############################################################################
#
#                              Arguments
#
##############################################################################
DEBUGGING = False
if DEBUGGING:
    print("!"*28)
    print("!!!! WARNING: DEBUGGING !!!!")
    print("!"*28)
    print()

parser = argparse.ArgumentParser(description='Args for NAS latent space search experiments')
parser.add_argument("--device",                 type=str, default="cpu")
parser.add_argument('--trials',                 type=int, default=10, help='Number of trials')
parser.add_argument('--dataset',                type=str, default='NB301')
parser.add_argument('--image_data',             type=str, default='cifar10', help='Only for NB201 relevant, choices between [cifar10_valid_converged, cifar100, ImageNet16-120]')
parser.add_argument("--name",                   type=str, default="Surrogate_Search_CMA_ES")
parser.add_argument("--weight_factor",          type=float, default=10e-3)
parser.add_argument("--num_init",               type=int, default=64)
parser.add_argument("--k",                      type=int, default=16)
parser.add_argument("--num_test",               type=int, default=5_00)
parser.add_argument("--ticks",                  type=int, default=1)
parser.add_argument("--tick_size",              type=int, default=16) 
parser.add_argument("--batch_size",             type=int, default=16)
parser.add_argument("--search_data",            type=int, default=608)
parser.add_argument("--saved_path",             type=str, help="Load pretrained Generator", default="state_dicts/NASBench301")
parser.add_argument("--saved_iteration",        type=str, default="best", help="Which iteration to load of pretrained Generator")
parser.add_argument("--seed",                   type=int, default=1)
parser.add_argument("--alpha",                  type=float, default=0.9)
parser.add_argument("--verbose",                type=str, default=True)
parser.add_argument("--epochs",                 type=int, default=50)
parser.add_argument("--lr",                     type=float, default=0.01)
parser.add_argument('--save_dir',
                        default='/output/nasbench_301',
                        type=str,
                        help='Path to save output')
parser.add_argument('--save_file_name',
                        default='result_NB301.log',
                        type=str,
                        help='save file name')
args = parser.parse_args()

##############################################################################
#
#                              Runfolder
#
##############################################################################
now = datetime.now()
runfolder = now.strftime("%Y_%m_%d_%H_%M_%S")
runfolder = f"NAS_Search_{args.dataset}/surrogate_search/{args.search_data}/reduce/{runfolder}_reduce_{args.name}_{args.dataset}_{args.seed}"
runfolder = os.path.join(Settings.FOLDER_EXPERIMENTS, runfolder)
if not os.path.exists(runfolder):
    os.makedirs(runfolder)

# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(runfolder, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')




##############################################################################
#
#                            Definitions
#
##############################################################################
def sample_data(G,
    random_cell,
    visited,
    measurements,
    device,
    search_space,
    num = 128,
    latent_dim = 32,
    normal=True,
    ):
    i = 0
    validity = 0
    possible_candidates = []
    with torch.no_grad():
        G.eval()
        # generate up to  4500 possible graphs
        for j in range(1_000,5_000,1_000):
            noise = (-6) * torch.rand(j, latent_dim) + 3
            try:
                graphs = G(noise.to(device))
            except:
                continue
            cells = measurements._compute_validity_score(graphs, search_space,  return_valid_spec=True)   

            validity += len(cells)
            sampled_data = []
            if normal:
                normal_cells = [random_cell[0] for _ in range(len(cells))]
                reduction_cells = cells
            else:
                normal_cells = cells
                reduction_cells = [random_cell[0] for _ in range(len(cells))]
            for g_n,g_r in zip(normal_cells, reduction_cells):
                if normal:
                    d =g_r
                    d.edge_index_normal = g_n.edge_index_normal
                    d.x_normal = g_n.x_normal
                    d.x_binary_normal = g_n.x_binary_normal
                    d.y_normal = g_n.y_normal
                    d.scores_normal = g_n.scores_normal
                    d.edge_index_reduce = g_r.edge_index
                    d.x_reduce = g_r.x
                    d.x_binary_reduce = g_r.x_binary
                    d.y_reduce = g_r.y
                    d.scores_reduce = g_r.scores
                else:
                    d =g_n
                    d.edge_index_normal = g_n.edge_index
                    d.x_normal = g_n.x
                    d.x_binary_normal = g_n.x_binary
                    d.y_normal = g_n.y
                    d.scores_normal = g_n.scores
                    d.edge_index_reduce = g_r.edge_index_reduce
                    d.x_reduce = g_r.x_reduce
                    d.x_binary_reduce = g_r.x_binary_reduce
                    d.y_reduce = g_r.y_reduce
                    d.scores_reduce = g_r.scores_reduce
                if hasattr(d, "x"):
                    del d.x
                    del d.edge_index
                    del d.y
                    del d.x_binary
                    del d.g
                    del d.scores
                sampled_data.append(d)

            for sample in sampled_data:
                if str(sample.y_normal.detach().tolist()+sample.y_reduce.detach().tolist()) not in visited:
                    sample.acc = sample.val_acc
                    possible_candidates.append(sample)
                    visited[str(d.y_normal.detach().tolist()+d.y_reduce.detach().tolist())] = 1

            i += j

            if len(possible_candidates) == num:
                break
            elif len(possible_candidates) > num :
                random_shuffle = np.random.permutation(range(len(possible_candidates)))
                possible_candidates = [possible_candidates[i] for i in random_shuffle[:num]]
                break


    validity = validity/i
    return possible_candidates


def search_EA(G,
    random_cell,
    visited,
    measurements,
    device,
    search_space,
    optimizer,
    mean,
    bound, 
    conditional_train_data, 
    num = 100,
    latent_dim = 32,
    normal=True,
    ):
    v = 0
    validity = 0
    possible_candidates = []
    with torch.no_grad():
        G.eval()
        # generate up to  4500 possible graphs
        j = 0
        num_data = 0
        while j < num:
            noise = torch.rand(num, latent_dim)
            for k in range(num):
                noise[k] = torch.FloatTensor(optimizer.ask())
            
            if v > 250000:
                solution = []
                sort = sorted(conditional_train_data, key=lambda i: i.val_acc)
                for i, arch in enumerate(sort):
                    if i < len(sort)/2:
                        z = arch.z.cpu().detach().tolist()
                        true_acc = arch.val_acc.item()
                        solution.append((z, true_acc))
                temp_optimizer = CMA(population_size=args.num_test, mean=mean, bounds=bound, sigma=0.1)
                #ws_mean, ws_sigma, ws_cov = get_warm_start_mgd(source_solutions=solution, gamma=0.1, alpha=0.1)
                #temp_optimizer = CMA(population_size=args.num_test, mean=ws_mean, bounds=bound, sigma=ws_sigma, cov=ws_cov)
                pop = []
                for k in range(num):
                    pop.append(temp_optimizer.ask())
                noise = torch.FloatTensor(pop)
            
            elif v > 500000:
                solution = []
                sort = sorted(conditional_train_data, key=lambda i: i.val_acc)
                for i, arch in enumerate(sort):
                    if i < len(sort)/4:
                        z = arch.z.cpu().detach().tolist()
                        true_acc = arch.val_acc.item()
                        solution.append((z, true_acc))
                #ws_mean, ws_sigma, ws_cov = get_warm_start_mgd(source_solutions=solution, gamma=0.1, alpha=0.1)
                #temp_optimizer = CMA(population_size=args.num_test, mean=ws_mean, bounds=bound, sigma=ws_sigma, cov=ws_cov)
                temp_optimizer = CMA(population_size=args.num_test, mean=mean, bounds=bound, sigma=0.1)
                pop = []
                for k in range(num):
                    pop.append(temp_optimizer.ask())
                noise = torch.FloatTensor(pop)
            else:
                temp_optimizer = CMA(population_size=args.num_test, mean=mean, bounds=bound, sigma=0.15)
                pop = []
                for k in range(num):
                    pop.append(temp_optimizer.ask())
                noise = torch.FloatTensor(pop)

            #noise = (-6) * torch.rand(100, latent_dim) + 3
            num_data += num
            #noise = [no. for no in noise]
            #ind_noise = torch.tensor(noise)
            #noise = torch.stack(noise)
            #print(noise.shape)
            try:
                graphs = G(noise.to(device))
            except:
                continue
            cells = measurements._compute_validity_score(graphs, search_space,  return_valid_spec=True)   

            validity += len(cells)
            sampled_data = []
            if normal:
                normal_cells = [random_cell[0] for _ in range(len(cells))]
                reduction_cells = cells
            else:
                normal_cells = cells
                reduction_cells = [random_cell[0] for _ in range(len(cells))]
            for g_n,g_r in zip(normal_cells, reduction_cells):
                if normal:
                    d =g_r
                    d.edge_index_normal = g_n.edge_index_normal
                    d.x_normal = g_n.x_normal
                    d.x_binary_normal = g_n.x_binary_normal
                    d.y_normal = g_n.y_normal
                    d.scores_normal = g_n.scores_normal
                    d.edge_index_reduce = g_r.edge_index
                    d.x_reduce = g_r.x
                    d.x_binary_reduce = g_r.x_binary
                    d.y_reduce = g_r.y
                    d.scores_reduce = g_r.scores
                else:
                    d =g_n
                    d.edge_index_normal = g_n.edge_index
                    d.x_normal = g_n.x
                    d.x_binary_normal = g_n.x_binary
                    d.y_normal = g_n.y
                    d.scores_normal = g_n.scores
                    d.edge_index_reduce = g_r.edge_index_reduce
                    d.x_reduce = g_r.x_reduce
                    d.x_binary_reduce = g_r.x_binary_reduce
                    d.y_reduce = g_r.y_reduce
                    d.scores_reduce = g_r.scores_reduce
                if hasattr(d, "x"):
                    del d.x
                    del d.edge_index
                    del d.y
                    del d.x_binary
                    del d.g
                    del d.scores
                sampled_data.append(d)

            for sample in sampled_data:
                if str(sample.y_normal.detach().tolist()+sample.y_reduce.detach().tolist()) not in visited:
                    sample.acc = sample.val_acc
                    possible_candidates.append(sample)
                    visited[str(d.y_normal.detach().tolist()+d.y_reduce.detach().tolist())] = 1

            v += num_data
            j = len(possible_candidates)

        if len(possible_candidates) > num :
            random_shuffle = np.random.permutation(range(len(possible_candidates)))
            possible_candidates = [possible_candidates[i] for i in random_shuffle[:num]]


    validity = validity/v
    return possible_candidates, visited, validity

def sample_one(G,
    measurements,
    device,
    search_space,
    num = 1,
    latent_dim = 32,
    normal=True
    ):
    i = 0
    validity = 0
    visited = {}
    possible_candidates = []
    with torch.no_grad():
        G.eval()
        # generate up to  4500 possible graphs
        for j in range(1_000,5_000,1_000):
            noise = (-6) * torch.rand(j, latent_dim) + 3
            graphs = G(noise.to(device))
            cells = measurements._compute_validity_score(graphs, search_space,  return_valid_spec=True)   
            validity += len(cells)
            for sample in cells:
                if str(sample.y.detach().tolist()) not in visited:
                    possible_candidates.append(sample)
                    visited[str(sample.y.detach().tolist())] = 1

            i += j
            if len(possible_candidates) >= num:
                random_shuffle = np.random.permutation(range(len(possible_candidates)))
                possible_candidates = [possible_candidates[i] for i in random_shuffle[:num]]
                break

    validity = validity/i
    for arch in possible_candidates:
        if normal:
            arch.edge_index_normal = arch.edge_index
            arch.x_normal = arch.x
            arch.x_binary_normal = arch.x_binary
            arch.y_normal = arch.y
            arch.scores_normal = arch.scores

        else:
            arch.edge_index_reduce = arch.edge_index
            arch.x_reduce = arch.x
            arch.x_binary_reduce = arch.x_binary
            arch.y_reduce = arch.y
            arch.scores_reduce = arch.scores

        if hasattr(arch, "x"):
            del arch.x
            del arch.edge_index
            del arch.y
            del arch.x_binary
            del arch.g
            del arch.scores

    return possible_candidates, validity

def get_rank_weights(outputs, weight_factor):
    outputs_argsort = np.argsort(-np.asarray(outputs))
    ranks = np.argsort(outputs_argsort)
    return 1 / (weight_factor * len(outputs) + ranks)

def w_dataloader(train_data, weight_factor, batch_size, weighted_retraining=True):
    b_size = batch_size
    if weighted_retraining:
        weight_factor = weight_factor
        outputs = np.array([graph.acc.item() for graph in train_data])
        weights = torch.tensor(get_rank_weights(outputs, weight_factor))

    else:
        weights = torch.ones(b_size)

    sampler = WeightedRandomSampler(
            weights, len(train_data))
    weighted_train_data = [(train_data[i],weights[i]) for i,w in enumerate(weights)]
    weighted_dataloader = DataLoader(weighted_train_data, sampler = sampler, batch_size = b_size, num_workers = 0, pin_memory = True)

    return weighted_dataloader


##############################################################################
#
#                              Training Loop
#
##############################################################################

##############################################################################
def train_ranker(train_data, val_data, ranker, optimizer, scheduler, batch_n):
    lambda_loss = LambdaNDCGLoss2().cpu()
    n = torch.LongTensor([batch_n]).cpu()
    ranker.train()
    best_loss = 1000000.0
    for i in range(args.epochs):
        
        
        var, acc = train_data[0], train_data[1]
        #acc = F.softmax(acc, 0)
        acc = acc.unsqueeze(0).cpu()
        optimizer.zero_grad()
        mu, std = ranker(var)
        #mu = F.log_softmax(mu, 0)
        mu = mu.unsqueeze(0).cpu()
        l_loss = lambda_loss(mu, acc, n)
        l_loss.backward()
        optimizer.step()
        scheduler.step()


        ranker.eval()
        
            
        var, acc = val_data[0], val_data[1]
        #acc = F.softmax(acc, 0)
        acc = acc.unsqueeze(0).cpu()
        optimizer.zero_grad()
        mu, std = ranker(var)
        #mu = F.log_softmax(mu, 0)
        mu = mu.unsqueeze(0).cpu()
        val_loss = lambda_loss(mu, acc, n)
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(ranker.state_dict())
        print('train_loss', val_loss.item())
        try:
            ranker.load_state_dict(best_model_wts)
        except:
            ranker = ranker
    return ranker

def train(
    real,
    b_size,
    G,
    weights,
    optimizer,
    alpha,
    normal_cell,
    ranker
):
    optimizer.zero_grad()


   
    lambda_loss = LambdaNDCGLoss2().cpu()
    n = torch.LongTensor([10]).cpu()
    nodes, edges = G.Decoder(real.z.reshape(b_size, -1))
    #nodes, edges = G.Decoder(noise)
    if normal_cell:
        ln = G.node_criterion(nodes.view(-1,G.num_node_atts),  torch.argmax(real.x_binary_normal, dim=1).view(b_size,-1).flatten())
        le = G.edge_criterion(edges, real.scores_normal.view(b_size, -1))
        #encodings = real.y_normal.reshape(b_size,-1).numpy()

    else:
        ln = G.node_criterion(nodes.view(-1,G.num_node_atts),  torch.argmax(real.x_binary_reduce, dim=1).view(b_size,-1).flatten())
        le = G.edge_criterion(edges, real.scores_reduce.view(b_size, -1))
        #encodings = real.y_reduce.reshape(b_size,-1).numpy()


    ln = torch.mean(ln.view(b_size, -1),1)
    le = torch.mean(le,1)
    recon_loss= 2*(ln+ 0.5*le)


    y = real.acc
    #y = F.softmax(y, 0)
    
    train_size = int(0.8 * len(real))
    train_acc = y[0:train_size]
    val_acc = y[train_size:]
    
    real.z =real.z.reshape(b_size, -1)
    noise_train = real.z[0:train_size]
    noise_val = real.z[train_size:]
    
    encodings_train = noise_train.cpu()#real.z.reshape(b_size, -1).cpu()
    encodings_val = noise_val.cpu()
    
    train_acc = train_acc.unsqueeze(0).cpu()
    val_acc = val_acc.unsqueeze(0).cpu()
    y = y.unsqueeze(0).cpu()
    best_loss = 1000000.0
    
    mean, std = ranker(real.z.cpu())
    
    mean = mean.unsqueeze(0).cpu()
    #ytrain = ytrain.unsqueeze(0).cpu()
    l_loss = lambda_loss(mean, y, n)
    recon_loss.requires_grad_(True)
    l_loss.requires_grad_(True)
    err = (1-alpha)*recon_loss + alpha*l_loss
    err = torch.mean(err*weights.to(recon_loss.device))

    err.backward()
    # optimize
    optimizer.step()
    # return stats

    return (err.item(),
            recon_loss.mean().item(),
            l_loss.mean().item(), 
            ranker
            )

def save_data(Dataset, train_data, path_measures, verbose=False):
    train_data = [Dataset.get_info_generated_graph(d, args.image_data) for d in train_data]
    torch.save(
        train_data,
        path_measures.format("all")
        )

    if verbose:
        top_5_acc = sorted([np.round(d.acc.item(),4) for d in train_data])[-5:]
        print('Top 5 acc after gradient method {}'.format(top_5_acc))

    return train_data

def eval_ranker(ranker, test_data, best, normal_cell):

    if normal_cell:    
        encodings = torch.stack([arch.z.cpu().detach() for arch in test_data])
    else:    
        encodings = torch.stack([arch.z.cpu().detach() for arch in test_data])

    mu, std = ranker(encodings)
    acq_candidates = acquisition_fct(mu.cpu().detach(), std.cpu().detach(), best, 'its')

    for i, arch in enumerate(test_data):
        arch.acc = torch.FloatTensor([mu[i]])

    return test_data, mu

##############################################################################
def training_loop(train_data, ranker):
    conditional_train_data = []
    search_data = args.search_data
    tmp_visited = {}
    reduce_cell,_ = sample_one(G, measurements, args.device, args.dataset, num=1, normal=False)
    sampled_data = sample_data(G, reduce_cell, tmp_visited, measurements, args.device, args.dataset, 128, normal=False)
    for i in range(len(sampled_data)):
        try:
            arch = sampled_data[i]
            arch = Dataset.get_info_generated_graph(arch, args.image_data)
            arch.to('cpu')
            conditional_train_data.append(arch)
        except:
            continue
        if len(conditional_train_data) == args.num_init:
            break
       
    #conditional_train_data = test_data[0:len(train_data)]
    print('Amount of to be searched data: {}'.format(search_data))

    print(f"G on device: {next(G.parameters()).device}")

    print("Creating Dataset.")

    path_measures_normal = os.path.join(
                trial_runfolder, "{}_normal.data"
                )

    instances = 0
    tick_size = args.tick_size
    instances_total = args.ticks * tick_size
    batch_n = 8
    reduce_cell,_ = sample_one(G, measurements, args.device, args.dataset, num=1, normal=False)

    with tqdm.tqdm(total=search_data//2, desc="Instances", unit="") as pbar:
        while True:
            weighted_dataloader = w_dataloader(conditional_train_data, args.weight_factor, args.batch_size)
            ####################################### train ranker
            ranker_optimizer = torch.optim.AdamW(
                                                ranker.parameters(),
                                                betas=(0.9, 0.999), eps=1.0e-9, weight_decay=5e-4
            )
            ranker_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                ranker_optimizer, T_max=float(100), eta_min=1e-5
            )
            latent_var = torch.stack([arch.z.cpu().detach() for arch in conditional_train_data])
            latent_var_acc = torch.stack([arch.acc.cpu().detach() for arch in conditional_train_data])
            
            train_size = int(0.8*len(conditional_train_data))
            train_data = [[], []]
            train_data[0] = latent_var[0:train_size]
            train_data[1] = latent_var_acc[0:train_size]
            val_data = [[], []]
            val_data[0] = latent_var[train_size:]
            val_data[1] = latent_var_acc[train_size:]
            ranker = train_ranker(train_data, val_data, ranker, ranker_optimizer, ranker_scheduler, batch_n)
            #######################################
            upd = len(conditional_train_data)-pbar.n
            if upd > 0:
                pbar.update(upd)
            for batch, w in weighted_dataloader:

                G.train()

                batch = batch.to(args.device)

                b_size = batch.batch.max().item() + 1

                err, recon_loss, pred_loss, ranker = train(
                    real=batch,
                    b_size=b_size,
                    G=G,
                    weights=w,
                    optimizer=optimizerG,
                    alpha=args.alpha,
                    normal_cell=True,
                    ranker=ranker
                )
                # measurements for saving
                measurements.add_measure("train_loss",      err,      instances)
                measurements.add_measure("recon_loss",      recon_loss,      instances)
                measurements.add_measure("pred_loss",      pred_loss,      instances)

                instances += b_size


            if instances >= instances_total:
                pbar.write("Starting evaluation of conditional generator for data size {}...".format(len(conditional_train_data)))

                visited = {}
                for d in conditional_train_data:
                    h = str(d.y_normal.detach().tolist()+d.y_reduce.detach().tolist())
                    visited[h] = 1
                sort = sorted(conditional_train_data, key=lambda i: i.acc)
                source_solutions = []
                for i, arch in enumerate(sort):
                    if i < int(len(sort)/2):
                        z = arch.z.cpu().detach().tolist()
                        true_acc = arch.acc.item()
                        source_solutions.append((z, true_acc))
                ws_mean, ws_sigma, ws_cov = get_warm_start_mgd(source_solutions=source_solutions, gamma=0.1, alpha=0.1)
                y = np.array([graph.acc.item() for graph in conditional_train_data])
                best = y.max()
                sort = sorted(conditional_train_data, key=lambda i: i.acc)
                mean = sort[0].z.cpu().numpy()
                bound = np.ones((len(mean), 2))
                bound[:, 0] = -3.0 * bound[:, 0]
                bound[:, 1] = 3.0 * bound[:, 1]
    
                optimizer = CMA(population_size=args.num_test, mean=ws_mean, bounds=bound, sigma=ws_sigma, cov=ws_cov)
                #optimizer = CMA(population_size=args.num_test, mean=mean, bounds=bound, sigma=0.1)
                test_data,_,_ = search_EA(G, reduce_cell, visited, measurements, args.device, args.dataset, optimizer, mean, bound, conditional_train_data, num=args.num_test, normal=False)
                #test_data = sample_data(G, reduce_cell, tmp_visited, measurements, args.device, args.dataset, num=args.num_test, normal=False)
                torch.save(test_data,
                    path_measures_normal.format("sampled_all_test_"+str(len(conditional_train_data)))
                    )

                test_data, acq_candidates =  eval_ranker(ranker, test_data, best, normal_cell=True)
                acc_pop = []
                for i, arch in enumerate(test_data):
                    #arch.val_acc = torch.FloatTensor([mu[i]])
                    acc_pop.append((arch.z.numpy(), acq_candidates[i]))
                optimizer.tell(acc_pop)
                # Sort given the surrogate model
                sort = sorted(test_data, key=lambda i:i.acc, reverse=True)

                
                n = len(conditional_train_data)
                k = 0
                for arch in sort:
                    try:
                        arch = Dataset.get_info_generated_graph(arch, args.image_data)
                        arch.to('cpu')
                        conditional_train_data.append(arch)
                        k +=1
                    except:
                        continue
                    if k == args.k:
                        break


                instances = 0
                tick_size += len(conditional_train_data) - n
                instances_total = args.ticks * tick_size

                save_data(Dataset, conditional_train_data, path_measures_normal, verbose=False)


                if args.verbose:
                    top_5_acc = sorted([np.round(d.acc.item(),4) for d in conditional_train_data])[-5:]
                    print('grid search, top 5 val acc {}'.format(top_5_acc))


            if len(conditional_train_data) > args.search_data/2:
                output_name = 'round'
                trial = i
                print("Finished Grid Search with conditional prediction...", flush=True)

                print('\n* Trial summary: results')

                results = []
                for query in range(args.num_init, len(conditional_train_data), args.k):
                    best_arch = sorted(conditional_train_data[:query], key=lambda i:i.acc)[-1]
                    test_acc = best_arch.acc.item()
                    results.append((query, test_acc))
                    #logger.info('current query: {:2d} best acc: {:.4f} '.format(query, best.item()))
            

                path = os.path.join(runfolder, '{}_{}_normal_cell.pkl'.format(output_name, trial))
                print('\n* Trial summary: results')
                print(results)
                print('\n* Saving to file {}'.format(path))
                with open(path, 'wb') as f:
                    pickle.dump([results, conditional_train_data], f )
                    f.close()

                break

    normal_cell = best_arch

    path_measures_reduce= os.path.join(
                trial_runfolder, "{}_reduction.data"
                )
    instances = 0
    tick_size = len(conditional_train_data)
    instances_total = args.ticks * tick_size
    with tqdm.tqdm(total=search_data, desc="Instances", unit="") as pbar:
        while True:
            weighted_dataloader = w_dataloader(conditional_train_data, args.weight_factor, args.batch_size)
            ####################################### train ranker
            ranker_optimizer = torch.optim.AdamW(
                                                ranker.parameters(),
                                                betas=(0.9, 0.999), eps=1.0e-9, weight_decay=5e-4
            )
            ranker_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                ranker_optimizer, T_max=float(100), eta_min=1e-5
            )
            latent_var = torch.stack([arch.z.cpu().detach() for arch in conditional_train_data])
            latent_var_acc = torch.stack([arch.acc.cpu().detach() for arch in conditional_train_data])
            train_size = int(0.8*len(conditional_train_data))
            train_data = [[], []]
            train_data[0] = latent_var[0:train_size]
            train_data[1] = latent_var_acc[0:train_size]
            val_data = [[], []]
            val_data[0] = latent_var[train_size:]
            val_data[1] = latent_var_acc[train_size:]
            ranker = train_ranker(train_data, val_data, ranker, ranker_optimizer, ranker_scheduler, batch_n)
            #######################################
            upd = len(conditional_train_data)-pbar.n
            if upd > 0:
                pbar.update(upd)
            for batch, w in weighted_dataloader:

                G.train()

                batch = batch.to(args.device)

                b_size = batch.batch.max().item() + 1

                err, recon_loss, pred_loss, ranker = train(
                    real=batch,
                    b_size=b_size,
                    G=G,
                    weights=w,
                    optimizer=optimizerG,
                    alpha=args.alpha,
                    normal_cell=False,
                    ranker=ranker
                )
                # measurements for saving
                measurements.add_measure("train_loss",      err,      instances)
                measurements.add_measure("recon_loss",      recon_loss,      instances)
                measurements.add_measure("pred_loss",      pred_loss,      instances)

                instances += b_size


            if instances >= instances_total:
                pbar.write("Starting evaluation of conditional generator for data size {}...".format(len(conditional_train_data)))

                visited = {}
                for d in conditional_train_data:
                    h = str(d.y_normal.detach().tolist()+d.y_reduce.detach().tolist())
                    visited[h] = 1

                sort = sorted(conditional_train_data, key=lambda i: i.acc)
                source_solutions = []
                for i, arch in enumerate(sort):
                    if i < int(len(sort)/2):
                        z = arch.z.cpu().detach().tolist()
                        true_acc = arch.acc.item()
                        source_solutions.append((z, true_acc))
                ws_mean, ws_sigma, ws_cov = get_warm_start_mgd(source_solutions=source_solutions, gamma=0.1, alpha=0.1)
                y = np.array([graph.acc.item() for graph in conditional_train_data])
                best = y.max()
                sort = sorted(conditional_train_data, key=lambda i: i.acc)
                mean = sort[0].z.cpu().numpy()
                bound = np.ones((len(mean), 2))
                bound[:, 0] = -3.0 * bound[:, 0]
                bound[:, 1] = 3.0 * bound[:, 1]
        
                optimizer = CMA(population_size=args.num_test, mean=ws_mean, bounds=bound, sigma=ws_sigma, cov=ws_cov)
              
                #optimizer = CMA(population_size=args.num_test, mean=mean, bounds=bound, sigma=0.1)
                test_data,_,_ = search_EA(G, [normal_cell], visited, measurements, args.device, args.dataset, optimizer, mean, bound, conditional_train_data, num=args.num_test, normal=True)
                #test_data= sample_data(G, [normal_cell], visited, measurements, args.device, args.dataset, num=args.num_test, normal=True)
                torch.save(test_data,
                    path_measures_reduce.format("sampled_all_test_"+str(len(conditional_train_data)))
                    )

                test_data, acq_candidates =  eval_ranker(ranker, test_data, best, normal_cell=False)
                acc_pop = []
                for i, arch in enumerate(test_data):
                    #arch.val_acc = torch.FloatTensor([mu[i]])
                    acc_pop.append((arch.z.numpy(), acq_candidates[i]))
                optimizer.tell(acc_pop)
        

                # Sort given the surrogate model
                sort = sorted(test_data, key=lambda i:i.acc, reverse=True)

                
                n = len(conditional_train_data)
                k = 0
                for arch in sort:
                    try:
                        arch = Dataset.get_info_generated_graph(arch, args.image_data)
                        arch.to('cpu')
                        
                        conditional_train_data.append(arch)
                        k +=1
                    except:
                        continue
                    if k == args.k:
                        break


                instances = 0
                tick_size += len(conditional_train_data) - n
                instances_total = args.ticks * tick_size

                save_data(Dataset, conditional_train_data, path_measures_reduce, verbose=False)


                if args.verbose:
                    top_5_acc = sorted([np.round(d.acc.item(),4) for d in conditional_train_data])[-5:]
                    print('grid search, top 5 val acc {}'.format(top_5_acc))


            if len(conditional_train_data) > args.search_data:
                output_name = 'round'
                trial = i
                print("Finished Grid Search with conditional prediction...", flush=True)

                print('\n* Trial summary: results')

                results = []
                #path = os.path.join(runfolder, '{}_{}_current_best.pkl'.format(output_name, trial))
                for query in range(args.num_init, len(conditional_train_data), args.k):
                    best_arch = sorted(conditional_train_data[:query], key=lambda i:i.acc)[-1]
                    test_acc = best_arch.acc.item()
                    results.append((query, test_acc))
                    logger.info('current query: {:2d} best acc: {:.4f} '.format(query, test_acc))
                    #best_arch_gen = Dataset.get_genotype(best_arch)
                    #logger.info('current genotype normal: {}\n reduce: {}'.format(best_arch_gen[0], best_arch_gen[1]))
                    

                path = os.path.join(runfolder, '{}_{}_reduction_cell.pkl'.format(output_name, trial))
                print('\n* Trial summary: results')
                print(results)
                print('\n* Saving to file {}'.format(path))
                with open(path, 'wb') as f:
                    pickle.dump([results, conditional_train_data], f )
                    f.close()

                path = os.path.join(runfolder, '{}_{}.pkl'.format(output_name, trial))
                print('\n* Trial summary: results')
                print(results)
                print('\n* Saving to file {}'.format(path))
                with open(path, 'wb') as f:
                    pickle.dump([results, conditional_train_data], f )
                    f.close()
                best_arch = sorted(conditional_train_data, key=lambda i:i.acc)[-1]
                #logger.info('current query: {:2d} best acc: {:.4f} '.format(results[0], best_arch.acc.item()))
                best_arch_gen = Dataset.get_genotype(best_arch)
                logger.info('current genotype normal: {}\n reduce: {}'.format(best_arch_gen[0], best_arch_gen[1]))

                break


##############################################################################

# torch.autograd.set_detect_anomaly(True)
if __name__ == "__main__":
    logger = setup_logger(save_path=os.path.join(
        args.save_dir, args.save_file_name))
    for i in range(args.trials):
        if args.trials > 1:
            args.seed = i+7
        # Set random seed for reproducibility
        print("Search deterministically.")
        seed = args.seed
        print(f"Random Seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        trial_runfolder = os.path.join(runfolder, 'round_{}'.format(i))
        print(f"Experiment folder: {trial_runfolder}")
        
        if not os.path.exists(trial_runfolder):
            os.makedirs(trial_runfolder)

        ##############################################################################
        #
        #                              Generator
        #
        ##############################################################################

        # load Checkpoint for pretrained Generator + MLP Predictor
        m = torch.load(os.path.join(args.saved_path, f"{args.saved_iteration}.model"), map_location=args.device) #pretrained_dict
        m["nets"]["G"]["pars"]["data_config"]["regression_input"] = 176
        m["nets"]["G"]["pars"]["data_config"]["regression_hidden"] = 176
        m["nets"]["G"]["pars"]["data_config"]["regression_output"] = 1
        m["nets"]["G"]["pars"]["acc_prediction"] = True
        m["nets"]["G"]["pars"]["list_all_lost"] = True
        
        G = Generator_Decoder(**m["nets"]["G"]["pars"]).to(args.device)
        state_dict = m["nets"]["G"]["state"]
        G_dict = G.state_dict()
        new_state_dict = {}
        for k,v in zip(G_dict.keys(), state_dict.values()):
            if k in G_dict :
                if v.size() == G_dict[k].size():
                    new_state_dict[k] = v
                else:
                    new_state_dict[k] = G_dict[k]
                    if "bias" in k:
                        new_state_dict[k][: v.size(0)] = v
                    else:
                        new_state_dict[k][: v.size(0),:v.size(1)] = v

        G_dict.update(new_state_dict)
        G.load_state_dict(G_dict)

        ##############################################################################
        #
        #                              Losses
        #
        ##############################################################################

        print("Initialize optimizers.")
        optimizerG = Optimizer(G, 'prediction').optimizer
        ranker = Listwise_Ranker(input_dim=32, hidden_dim=128, hidden_layer=2)
        ##############################################################################
        #
        #                              Dataset
        #
        ##############################################################################
        print("Creating Dataset.")
        NASBench = NASBench301
        Dataset = NASBench301.Dataset
        dataset =  NASBench301.Dataset(batch_size=args.batch_size, sample_size=args.num_init, only_prediction=True)

        conditional_train_data = dataset.train_data
        for arch in conditional_train_data:
            arch = Dataset.get_info_generated_graph(arch)

        ##############################################################################
        #
        #                              Checkpoint
        #
        ##############################################################################
        # Load Measurements
        measurements = Measurements(
                        G = G,
                        batch_size = args.batch_size,
                        NASBench= NASBench
                    )

        chkpt_nets = {
            "G": G,
        }
        chkpt_optimizers = {
            "G": optimizerG,
        }
        checkpoint = Checkpoint(
            folder = trial_runfolder,
            nets = chkpt_nets,
            optimizers = chkpt_optimizers,
            measurements = measurements
        )




        training_loop(conditional_train_data, ranker)