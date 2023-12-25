import torch
import numpy as np
import sys, os, random, argparse, pickle, tqdm
from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader

sys.path.insert(1, os.path.join(os.getcwd()))
from Generator import Generator_Decoder
from Measurements import Measurements
from Optimizer import Optimizer
from Checkpoint import Checkpoint
import Settings
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
    print("!" * 28)
    print("!!!! WARNING: DEBUGGING !!!!")
    print("!" * 28)
    print()

parser = argparse.ArgumentParser(description='Args for NAS latent space search experiments')
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument('--trials', type=int, default=10, help='Number of trials')
parser.add_argument("--dataset", type=str, default='NB101', help='Choice between NB101 and NB201')
parser.add_argument('--image_data', type=str, default='cifar10', help='Only for NB201 relevant, choices between [cifar10_valid_converged, cifar100, ImageNet16-120]')
parser.add_argument("--name", type=str, default="Search_S_CMA_ES")
parser.add_argument("--weight_factor", type=float, default=10e-3)
parser.add_argument("--num_init", type=int, default=64)
parser.add_argument("--k", type=int, default=16)
parser.add_argument("--num_test", type=int, default=500)
parser.add_argument("--ticks", type=int, default=1)
parser.add_argument("--tick_size", type=int, default=16)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--search_data", type=int, default=608)
parser.add_argument("--saved_path", type=str, help="Load pretrained Generator", default="/state_dicts/NASBench101")
parser.add_argument("--saved_iteration", type=str, default='best', help="Which iteration to load of pretrained Generator")
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--alpha", type=float, default=0.9)
parser.add_argument("--verbose", type=str, default=True)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument('--save_dir',
                        default='/output/nasbench_201',
                        type=str,
                        help='Path to save output')
parser.add_argument('--save_file_name',
                        default='result_CMA_nas101.log',  
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
runfolder = f"NAS_Search_S_CMA_ES_{args.dataset}/{args.image_data}/{runfolder}_{args.name}_{args.dataset}_{args.seed}"
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
    visited,
    measurements, 
    device,
    search_space,
    num = 100,
    latent_dim = 32, ):
    v = 0
    validity = 0
    possible_candidates = []
    with torch.no_grad():
        G.eval()
        # generate up to  4500 possible graphs
        for j in range(1_00,1_000,1_00):
            noise = (-6) * torch.rand(j, latent_dim) + 3
            graphs = G(noise.to(device))
            valid_sampled_data = measurements._compute_validity_score(graphs, search_space,  return_valid_spec=True)
            if valid_sampled_data == []:
                continue
            sampled_y = torch.stack([g.y for g in valid_sampled_data])
            sampled_hash_idx = dataset.query(sampled_y)  
            validity += len(valid_sampled_data)
            for i, idx in enumerate(sampled_hash_idx):
                if idx not in visited:
                    visited.append(idx)
                    possible_candidates.append(valid_sampled_data[i])
            v += j
            if len(possible_candidates) > num:
                random_shuffle = np.random.permutation(range(len(possible_candidates)))
                possible_candidates = [possible_candidates[i] for i in random_shuffle[:num]]
                break

    print('In total generated graphs: {}'.format(v))
    if v !=0:
        validity = validity/v
        print('Validity: {}'.format(validity))
    else:
        print('Validity: {}'.format(0))
    return possible_candidates, visited, validity




def sample_data_with_EA(G,
                visited,
                measurements,
                device,
                search_space,
                optimizer,
                mean,
                bound, 
                conditional_train_data,
                num=100,
                latent_dim=32):

    v = 0
    validity = 0
    possible_candidates = []
    with torch.no_grad():
        G.eval()
        # generate up to  4500 possible graphs

        j = 0
        num_data = 0
        while j < num:
            pop = []
            for k in range(num):
                pop.append(optimizer.ask())
            noise = torch.FloatTensor(pop)
            
           
            if v > 200000:
                break
            num_data += num
            v += num_data
            #ind_noise = torch.tensor(noise).float()
            #print(noise.shape)
            try:
                graphs = G(noise.to(device))
            except:
                continue
            valid_sampled_data = measurements._compute_validity_score(graphs, search_space, return_valid_spec=True)
            if valid_sampled_data == []:
                continue
            sampled_y = torch.stack([g.y for g in valid_sampled_data])
            sampled_hash_idx = dataset.query(sampled_y)
            validity += len(valid_sampled_data)
            for i, idx in enumerate(sampled_hash_idx):
                if idx not in visited:
                    visited.append(idx)
                    possible_candidates.append(valid_sampled_data[i])
            
            j = len(possible_candidates)
        if len(possible_candidates) > num:
            random_shuffle = np.random.permutation(range(len(possible_candidates)))
            possible_candidates = [possible_candidates[i] for i in random_shuffle[:num]]


    print('In total generated graphs: {}'.format(v))
    if v != 0:
        validity = validity / v
        print('Validity: {}'.format(validity))
    else:
        print('Validity: {}'.format(0))
    return possible_candidates, visited, validity


def get_rank_weights(outputs, weight_factor):
    outputs_argsort = np.argsort(-np.asarray(outputs))
    ranks = np.argsort(outputs_argsort)
    return 1 / (weight_factor * len(outputs) + ranks)


def w_dataloader(train_data, weight_factor, batch_size, weighted_retraining=True):
    b_size = batch_size
    if weighted_retraining:
        weight_factor = weight_factor
        outputs = np.array([graph.val_acc.item() for graph in train_data])
        weights = torch.tensor(get_rank_weights(outputs, weight_factor))

    else:
        weights = torch.ones(b_size)

    sampler = WeightedRandomSampler(
        weights, len(train_data))
    weighted_train_data = [(train_data[i], weights[i]) for i, w in enumerate(weights)]
    weighted_dataloader = DataLoader(weighted_train_data, sampler=sampler, batch_size=b_size, num_workers=0, pin_memory=True)

    return weighted_dataloader


##############################################################################
#
#                              Training Loop
#
##############################################################################
# https://github.com/martius-lab/blackbox-backprop
class Ranker(torch.autograd.Function):
    """Black-box differentiable rank calculator."""

    _lambda = Variable(torch.tensor(2.0))  # treat as hyperparm

    @staticmethod
    def forward(ctx, scores):
        """
        scores:  batch_size x num_elements tensor of real valued scores
        """
        arg_ranks = torch.argsort(
            torch.argsort(scores, dim=1, descending=True)
        ).float() + 1
        arg_ranks.requires_grad = True
        ctx.save_for_backward(scores, arg_ranks, Ranker._lambda)
        return arg_ranks

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_outputs: upstream gradient batch_size x num_elements
        """
        scores, arg_ranks, _lambda = ctx.saved_tensors
        perturbed_scores = scores + _lambda * grad_output
        perturbed_arg_ranks = torch.argsort(
            torch.argsort(perturbed_scores, dim=1, descending=True)
        ) + 1
        return - 1 / _lambda * (arg_ranks - perturbed_arg_ranks), None  # gradient according to backprob paper


##############################################################################
def train_ranker(ranker, optimizer, scheduler, batch_n, conditional_train_data):
    
    encodings = torch.stack([arch.y.cpu().detach() for arch in conditional_train_data])
    
    latent_var_acc = torch.stack([arch.val_acc.cpu().detach() for arch in conditional_train_data])
    
    train_size = int(0.8*len(conditional_train_data))
    train_data = [[], []]
    train_data[0] = encodings[0:train_size]
    train_data[1] = latent_var_acc[0:train_size]
    val_data = [[], []]
    val_data[0] = encodings[train_size:]
    val_data[1] = latent_var_acc[train_size:]
    
    lambda_loss = LambdaNDCGLoss2().cpu()
    n = torch.LongTensor([batch_n]).cpu()
    ranker.train()
    best_loss = 1000000.0
    for i in range(args.epochs):
        
        
        var, acc = train_data[0], train_data[1]
        acc = F.softmax(acc, 0)
        acc = acc.unsqueeze(0).cpu()
        optimizer.zero_grad()
        mu, std = ranker(var)
        mu = F.log_softmax(mu, 0)
        mu = mu.unsqueeze(0).cpu()
        l_loss = lambda_loss(mu, acc, n)
        l_loss.backward()
        optimizer.step()
        scheduler.step()


        ranker.eval()
        
            
        var, acc = val_data[0], val_data[1]
        acc = F.softmax(acc, 0)
        acc = acc.unsqueeze(0).cpu()
        optimizer.zero_grad()
        mu, std = ranker(var)
        mu = F.log_softmax(mu, 0)
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
        ranking_function,
        weights,
        optimizer,
        alpha,
        ranker,
        batch_n
):
    optimizer.zero_grad()
    generated, recon_loss, _ = G.loss(real, b_size)
    encodings = real.y.reshape(b_size,-1).cpu()
    lambda_loss = LambdaNDCGLoss2().cpu()
    n = torch.LongTensor([batch_n]).cpu()
    ytrain = real.val_acc.cpu()
   
    ytrain = F.softmax(ytrain, 0)
    ytrain = ytrain.unsqueeze(0).cpu()
    
    mean, std = ranker(encodings)
    mean = F.log_softmax(mean, 0)
    #ytrain = F.softmax(ytrain, 0)
    mean = mean.unsqueeze(0).cpu()
    #ytrain = ytrain.unsqueeze(0).cpu()
    l_loss = lambda_loss(mean, ytrain, n)
    recon_loss.requires_grad_(True)
    l_loss.requires_grad_(True)

    err = (1 - alpha) * recon_loss + alpha * l_loss
    err = torch.mean(err * weights.to(real.val_acc.device))

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
        top_5_acc = sorted([np.round(d.acc.item(), 4) for d in train_data])[-5:]
        print('Top 5 acc after gradient method {}'.format(top_5_acc))

    return train_data


def eval_ranker(ranker, best, test_data):
    encodings = torch.stack([arch.y.cpu().detach() for arch in test_data])
    mu, std = ranker(encodings)
                
               
    '''from search ablation LSO'''
    acq_candidates = acquisition_fct(mu.cpu().detach(), std.cpu().detach(), best, 'ei')
    for i, arch in enumerate(test_data):
        arch.val_acc = torch.FloatTensor([acq_candidates[i]])

    return test_data, acq_candidates


##############################################################################
def training_loop(train_data, ranker):
    data = []
    while len(data) < len(train_data):
        init_data = G.generate(instances=1, device=args.device)
        true_data = Dataset.get_info_generated_graph(init_data, args.image_data)
        if true_data != []:
            data.append(true_data[0])
    conditional_train_data = data
    ranking_function = Ranker.apply

    search_data = args.search_data
    print('Amount of to be searched data: {}'.format(search_data))

    print(f"G on device: {next(G.parameters()).device}")

    print("Creating Dataset.")

    path_measures = os.path.join(
        trial_runfolder, "{}.data"
    )
    batch_n = 10
    instances = 0
    tick_size = args.tick_size
    instances_total = args.ticks * tick_size
    with tqdm.tqdm(total=search_data, desc="Instances", unit="") as pbar:
        while True:
            if args.verbose:
                pbar.write("Starting Training Loop...")
            weighted_dataloader = w_dataloader(conditional_train_data, args.weight_factor, args.batch_size)
            ####################################### train ranker
            ranker_optimizer = torch.optim.Adam(
                                                ranker.parameters(),
                                                betas=(0.9, 0.999), eps=1.0e-9, weight_decay=5e-4
            )
            ranker_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                ranker_optimizer, T_max=float(100), eta_min=1e-5
            )
            
            ranker = train_ranker(ranker, ranker_optimizer, ranker_scheduler, batch_n, conditional_train_data)
            #######################################
            upd = len(conditional_train_data) - pbar.n
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
                    ranking_function=ranking_function,
                    weights=w,
                    optimizer=optimizerG,
                    alpha=args.alpha,
                    ranker=ranker,
                    batch_n=batch_n
                )
                # measurements for saving
                measurements.add_measure("train_loss", err, instances)
                measurements.add_measure("recon_loss", recon_loss, instances)
                measurements.add_measure("pred_loss", pred_loss, instances)

                instances += b_size

            if instances >= instances_total:
                if args.verbose:
                    pbar.write("Starting evaluation of conditional generator for data size {}...".format(len(conditional_train_data)))

                visited = dataset.query(torch.stack([g.y for g in conditional_train_data]))
                # cluster_x = []
                # for i, arch in enumerate(conditional_train_data):
                #     z = arch.z.cpu().detach().tolist()
                #     #z.extend([arch.val_acc.item()])
                #     cluster_x.append(z)
                # cluster_x = np.array(cluster_x)
                # clust = skc.OPTICS(min_samples=3, xi=0.1, min_cluster_size=0.1)
                # clust.fit(cluster_x)
                # labels = cluster_optics_dbscan(reachability=clust.reachability_, core_distances=clust.core_distances_, ordering=clust.ordering_, eps=2)
                # labels = [clust.labels_] + labels
                # centers = []
                # outliers = []
                # for i, ind in enumerate(labels[0]):
                #     if ind == -1:
                #         outliers.extend(cluster_x[labels[0] == ind])
                #         continue
                #     centers.append(np.mean(cluster_x[labels[0] == ind], axis=0))

                # dis = []
                # for center in centers:
                #     dist = np.sqrt(np.sum(np.square(cluster_x - center), axis=1))
                #     dis.append(np.min(dist))
                # index = np.argmin(dis)
                # c_data = centers[index]
                # c_data = c_data.reshape(mean.shape[0],-1).flatten()
                
                all_solutions = []
                sort = sorted(conditional_train_data, key=lambda i: i.val_acc)
                for i, arch in enumerate(sort):
                    if i < int(len(sort)/2):
                        z = arch.z.cpu().detach().tolist()
                        true_acc = arch.acc.item()
                        all_solutions.append((z, true_acc))
               
                ws_mean, ws_sigma, ws_cov = get_warm_start_mgd(source_solutions=all_solutions, gamma=0.3, alpha=0.3) ##0.1,0.1for cifar100
                y = np.array([graph.val_acc.item() for graph in conditional_train_data])
                best = y.max()
                
                mean = sort[0].z.cpu().numpy()
                bound = np.ones((len(mean), 2))
                bound[:, 0] = -3.0 * bound[:, 0]
                bound[:, 1] = 3.0 * bound[:, 1]
                #steps = np.zeros(len(mean)) for CMAwM
                optimizer = CMA(population_size=args.num_test, mean=ws_mean, bounds=bound, sigma=ws_sigma, cov=ws_cov)
                #test_data,_,_ = sample_data(G, visited, measurements, args.device, args.dataset, num=args.num_test)
                # Finished Training, now evaluate trained surrogate model for next samples
                test_data, _, _ = sample_data_with_EA(G, visited, measurements, args.device, args.dataset, optimizer, mean, bound, conditional_train_data, num=args.num_test)
                n = len(conditional_train_data)

                torch.save(test_data,
                           path_measures.format("sampled_all_test_" + str(len(conditional_train_data)))
                           )

                if test_data == [] or len(test_data) < args.num_test:
                    continue
               



                test_data, acq_candidates =  eval_ranker(ranker, best, test_data)

                acc_pop = []
                for i, arch in enumerate(test_data):
                    #arch.val_acc = torch.FloatTensor([mu[i]])
                    acc_pop.append((arch.z.numpy(), acq_candidates[i]))

                # 3) add the k architectures with highest acquisition function value
                # k = topk
                sort = sorted(test_data, key=lambda i:i.val_acc)

                for arch in sort[-args.k:]:

                    arch = Dataset.get_info_generated_graph(arch, args.image_data)

                    arch.to('cpu')
                    conditional_train_data.append(arch)

                ''''''

                instances = 0
                tick_size += len(conditional_train_data) - n
                instances_total = args.ticks * tick_size

                save_data(Dataset, conditional_train_data, path_measures, verbose=False)
                optimizer.tell(acc_pop)
                if args.verbose:
                    top_5_acc = sorted([np.round(d.val_acc.item(), 4) for d in conditional_train_data])[-5:]
                    print('grid search, top 5 val acc {}'.format(top_5_acc))
                    

            if len(conditional_train_data) > search_data:
                output_name = 'round'
                trial = i
                print("Finished Grid Search with conditional prediction...", flush=True)

                print('\n* Trial summary: results')

                results = []
                for query in range(args.num_init, len(conditional_train_data), args.k):
                    best_arch = sorted(conditional_train_data[:query], key=lambda i: i.val_acc)[-1]
                    test_acc = best_arch.acc.item()
                    val_acc = best_arch.val_acc.item()
                    results.append((query, val_acc, test_acc))
                    logger.info('current query: {:2d} best val acc: {:.4f} best test acc: {:.4f} '.format(query, val_acc, test_acc))
                path = os.path.join(runfolder, '{}_{}.pkl'.format(output_name, trial))
                print('\n* Trial summary: results')
                print(results)
                print('\n* Saving to file {}'.format(path))
                with open(path, 'wb') as f:
                    pickle.dump([results, conditional_train_data], f)
                    f.close()

                break


##############################################################################

# torch.autograd.set_detect_anomaly(True)
if __name__ == "__main__":
    logger = setup_logger(save_path=os.path.join(
        args.save_dir, args.save_file_name))
    for i in range(args.trials):
        if args.trials > 1:
            args.seed = i
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

        # load Checkpoint for pretrained Generator + pretrained MLP Predictor
        m = torch.load(os.path.join(args.saved_path, f"{args.saved_iteration}.model"), map_location=args.device)  # pretrained_dict
        m["nets"]["G"]["pars"]["list_all_lost"] = True
        m["nets"]["G"]["pars"]["acc_prediction"] = False
        # m["nets"]["G"]["pars"]["mse"] = False
        G = Generator_Decoder(**m["nets"]["G"]["pars"]).to(args.device)

        state_dict_g = m["nets"]["G"]["state"]
        G_dict = G.state_dict()
        state_dict = {k: v for k, v in state_dict_g.items() if k in G_dict}

        G_dict.update(state_dict)
        G.load_state_dict(G_dict)

        
        
        ##############################################################################
        #
        #                              Dataset
        #
        ##############################################################################
        print("Creating Dataset.")
        if args.dataset == 'NB101':
            import datasets.NASBench101 as NASBench101
            NASBench = NASBench101
            Dataset = NASBench101.Dataset
            dataset = NASBench101.Dataset(batch_size=args.batch_size, sample_size=args.num_init, only_prediction=True)
            ranker = Listwise_Ranker(input_dim=56, hidden_dim=512, hidden_layer=2)
        elif args.dataset == 'NB201':
            import datasets.NASBench201 as NASBench201
            NASBench = NASBench201
            Dataset = NASBench.Dataset
            dataset = NASBench.Dataset(batch_size=args.batch_size, sample_size=args.num_init, only_prediction=True, dataset=args.image_data)
            ranker = Listwise_Ranker(input_dim=84, hidden_dim=512, hidden_layer=2)
        else:
            raise TypeError("Unknow Seach Space : {:}".format(args.dataset))

        conditional_train_data = dataset.train_data
        dataset.load_hashes()

        ##############################################################################
        #
        #                              Losses
        #
        ##############################################################################

        print("Initialize optimizers.")
        optimizerG = Optimizer(G, 'prediction').optimizer

        ##############################################################################
        #
        #                              Checkpoint
        #
        ##############################################################################
        # Load Measurements
        measurements = Measurements(
            G=G,
            batch_size=args.batch_size,
            NASBench=NASBench
        )

        chkpt_nets = {
            "G": G,
        }
        chkpt_optimizers = {
            "G": optimizerG,
        }
        checkpoint = Checkpoint(
            folder=trial_runfolder,
            nets=chkpt_nets,
            optimizers=chkpt_optimizers,
            measurements=measurements
        )

        training_loop(conditional_train_data, ranker)