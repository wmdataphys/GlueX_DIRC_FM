import os
import re
import json
import time
import copy
import pkbar
import random
import pickle
import warnings
import argparse
import itertools
import numpy as np

import torch

from models.GPT import Cherenkov_GPT
from dataloader.tokenizer import TimeTokenizer

from make_plots_GlueX import make_plots_fastsim,combine_images_to_pdf,photon_yield_plots,make_ratios

warnings.filterwarnings("ignore", message=".*weights_only.*")


def main(config,args):

    # Remove seeding, make it random.
    seed = int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

    inference_batch = config['Inference']['batch_size']
    print('------------------------ Setup ------------------------')
    print("Generating",args.method,"in batches of:", inference_batch)
    print(f"Config File: {args.config}")
    print(f"Generated Particle Type: {args.method}")
    print(f"Trained with Multiple GPUs (DP): {'Yes' if args.distributed else 'No'}")
    print(f"Generation Temperature: {args.temperature}")
    print(f"Use Dynamic Temperature: {'Yes' if args.dynamic_temperature else 'No'}")
    print(f"Sampling Method: {args.sampling}")
    print(f"TopK Value: {args.topK}")
    print(f"Nucleus P Value: {args.nucleus_p}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise RuntimeError("No GPU was found! Exiting.")
   
    # Model params.
    vocab_size = config['model']['vocab_size']
    time_vocab = config['model']['time_vocab']
    embed_dim = config['model']['embed_dim']
    attn_heads = config['model']['attn_heads']
    num_blocks = config['model']['num_blocks']
    kin_size = config['model']['kin_size']
    hidden_units = config['model']['hidden_units']
    mlp_scale = config['model']['mlp_scale']
    msl = config['model']['max_seq_length']
    drop_rates = config['model']['drop_rates']
    use_MoE = bool(config['model']['use_MoE'])
    num_experts = config['model']['num_experts']
    num_classes = config['model']['num_classes']
    # data params
    stats = config['stats']
    conditional_maxes = np.array([stats['P_max'],stats['theta_max'],stats['phi_max']])#.reshape(-1,1)
    conditional_mins = np.array([stats['P_min'],stats['theta_min'],stats['phi_min']])#.reshape(-1,1)

    # Time tokenization
    digitize_time = bool(config['digitize_time'])
    if digitize_time:
        print("Digitizing time - classification over adjacent vocabulary.")
        time_res = config['stats']['time_res']
        t_max = config['stats']['time_max']
        t_min = config['stats']['time_min']
        print("Time Res: ",time_res)
        print("Time vocab: ",time_vocab)
        time_digitizer = TimeTokenizer(t_max=t_max,t_min=t_min,resolution=time_res)
        de_tokenize_func = time_digitizer.de_tokenize

    else:
        print("Using regression over time domain.")
        time_digitizer = None
        de_tokenize_func = None

    net = Cherenkov_GPT(vocab_size, msl, embed_dim,attn_heads=attn_heads,kin_size=kin_size,
        num_blocks=num_blocks,hidden_units=hidden_units,digitize_time=digitize_time,mlp_scale=mlp_scale,
        time_vocab=time_vocab,detokenize_func=de_tokenize_func,drop_rates=drop_rates,use_MoE=use_MoE,num_experts=num_experts,num_classes=num_classes)

    if args.distributed:
        net = DataParallel(net)
    
    if args.method == 'Kaon':
        dicte = torch.load(config['Inference']['kaon_model_path'])
        datapoints = (
            list(np.load(config['dataset']['testing']["kaon_data_path"], allow_pickle=True)) +
            list(np.load(config['dataset']['validation']["kaon_data_path"], allow_pickle=True)) +
            list(np.load(config['dataset']['training']["kaon_data_path"], allow_pickle=True))
        )
    elif args.method == 'Pion':
        dicte = torch.load(config['Inference']['pion_model_path'])
        datapoints = (
            list(np.load(config['dataset']['testing']["pion_data_path"], allow_pickle=True)) +
            list(np.load(config['dataset']['validation']["pion_data_path"], allow_pickle=True)) +
            list(np.load(config['dataset']['training']["pion_data_path"], allow_pickle=True))
        )
    else:
        raise ValueError('Method not found. Choose from Pion or Kaon.')


    net.to('cuda')
    net.load_state_dict(dicte['net_state_dict'])
    net.eval()
    net = torch.compile(model=net,mode="max-autotune")

    filtered_datapoints = []
    
    for dp in datapoints:
        theta__ = dp['Theta']
        p__ = dp['P']
        phi__ = dp['Phi']
        nh = dp['NHits']

        if ((theta__ > stats['theta_min']) and (theta__ < stats['theta_max']) 
            and (p__ > stats['P_min']) and (p__ < stats['P_max']) 
            and (phi__ > stats['phi_min']) and (phi__ < stats['phi_max'])
            and (nh > 5)):  
            #and (np.min(lt) > stats['time_min']) and (np.max(lt) < stats['time_max'])
            time_ = np.array(dp['leadTime'])
            pmtID_ = np.array(dp['pmtID'])
            pixelID_ = np.array(dp['pixelID'])
            pos_time = np.where((time_ > stats['time_min']) & (time_ < stats['time_max']))[0]
            time_ = time_[pos_time]
            pmtID_ = pmtID_[pos_time]
            pixelID_ = pixelID_[pos_time]
            dp['leadTime'] = time_
            dp['pixelID'] = pixelID_
            dp['pmtID'] = pmtID_
            filtered_datapoints.append(dp)

    barIDs = np.array([dp['BarID'] for dp in filtered_datapoints])
    barX = np.array([dp['X'] for dp in filtered_datapoints])

    assert len(barIDs) == len(filtered_datapoints)
    assert len(barX) == len(filtered_datapoints)

    # Control what you want to generate pair wise here:
    xs = [(-30,-20),(-20,-10),(-10,0),(0,10),(10,20),(20,30)]
    # bars = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    #         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    #         34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]  

    bars = [0, 10, 31, 43]  

    # Primary regions of interest
    # bars = [10,31]
    # xs = [(0,10)]

    stats=config['stats']
    combinations = list(itertools.product(xs,bars))
    print('Generating plots for {0} combinations of BarID and x ranges.'.format(len(combinations)))

    out_folder = os.path.join("Generations",config['Inference']['generation_dir'])
    os.makedirs(out_folder,exist_ok=True)
    os.makedirs(os.path.join(out_folder,"Plots"),exist_ok=True)
    print("Outputs can be found in " + str(out_folder))
    
    for j,combination in enumerate(combinations):
        (x_low, x_high), barID = combination
        idx = np.where((barIDs == barID) & (barX > x_low) & (barX < x_high))[0]

        if len(idx) == 0:
            print(f"Skipping bar {barID}, x ({x_low},{x_high}) - no data.")
            continue


        ground_truth = [filtered_datapoints[l] for l in idx]
        track_params = [[filtered_datapoints[l]['P'],filtered_datapoints[l]['Theta'],filtered_datapoints[l]['Phi']] for l in idx]
        track_params = np.vstack(track_params)

        numTracks = len(idx)
        num_itter = numTracks // inference_batch
        last_batch = numTracks % inference_batch

        out_path_ = os.path.join(out_folder, args.method + f"_barID_{barID}_barX_{x_low}_{x_high}_ntracks_{numTracks}.pkl")

        if os.path.exists(out_path_):
            print(f"File found for bar {barID}, x in ({x_low},{x_high}). Skipping generation and running plotting.")
            filename = args.method + f"_barID_{barID}_barX_{x_low}_{x_high}_ntracks_{numTracks}.pdf"
            outpath = os.path.join(out_folder,"Plots")
            make_plots_fastsim(out_path_,args.method,barID,x_low,x_high,outpath,filename,log_norm=True)
        else:
            kbar = pkbar.Kbar(target=num_itter + 1, width=20, always_stateful=False)
            generations = []
            start = time.time()
            for i in range(num_itter):

                with torch.set_grad_enabled(False):
                    start_idx = i * inference_batch
                    end_idx = (i + 1) * inference_batch
                    k = track_params[start_idx:end_idx, :]  

                    k_unscaled = k.copy()
                    k = 2*(k - conditional_mins) / (conditional_maxes - conditional_mins) - 1.0
                    k = torch.tensor(k).to('cuda').float()

                    if use_MoE:
                        if args.method == "Kaon":
                            class_label = torch.ones((k.shape[0],),dtype=torch.float32,device=k.device)
                        else:
                            class_label = torch.zeros((k.shape[0],),dtype=torch.float32,device=k.device)
                    else:
                        class_label = None

                    gen = net.generate(k,unscaled_k=k_unscaled,class_label=class_label,method=args.sampling,temperature=args.temperature,topK=args.topK,nucleus_p=args.nucleus_p,
                                            dynamic_temp=args.dynamic_temperature,add_dark_noise=args.dark_noise,PID=321)

                generations += gen
                kbar.add(1)

            end = time.time()

            if last_batch > 0:
                start_idx = num_itter * inference_batch
                k = track_params[start_idx:, :]  # Correct slice
                k_unscaled = k.copy()
                k = 2*(k - conditional_mins) / (conditional_maxes - conditional_mins) - 1.0
                k = torch.tensor(k).to('cuda').float()

                if use_MoE:
                    if args.method == "Kaon":
                        class_label = torch.ones((k.shape[0],),dtype=torch.float32,device=k.device)
                    else:
                        class_label = torch.zeros((k.shape[0],),dtype=torch.float32,device=k.device)
                else:
                    class_label = None

                gen = net.generate(k,unscaled_k=k_unscaled,class_label=class_label,method=args.sampling,temperature=args.temperature,topK=args.topK,nucleus_p=args.nucleus_p,
                                        dynamic_temp=args.dynamic_temperature,add_dark_noise=args.dark_noise,PID=321)

                generations += gen
                kbar.add(1)

            torch.cuda.empty_cache()

            n_photons = 0
            n_gamma = 0

            for i in range(len(ground_truth)):
                n_photons += ground_truth[i]['NHits']

            for i in range(len(generations)):
                n_gamma += generations[i]['NHits']

            print(" ")
            print("Number of tracks generated: ",numTracks)
            print("True number of tracks: ",len(ground_truth))
            print("Elapsed Time: ", end - start)
            print("Average time / track: ",(end - start) / (numTracks))
            print("True photon yield: ",n_photons," Generated photon yield: ",n_gamma)
            print(" ")

            gen_dict = {}
            gen_dict['FastSimPhotons'] = n_gamma
            gen_dict['TruePhotons'] = n_photons
            gen_dict['fast_sim'] = generations
            gen_dict['truth'] = ground_truth


            out_path_ = os.path.join(out_folder, args.method + f"_barID_{barID}_barX_{x_low}_{x_high}_ntracks_{numTracks}.pkl")
            with open(out_path_,"wb") as file:
                pickle.dump(gen_dict,file)

            filename = args.method + f"_barID_{barID}_barX_{x_low}_{x_high}_ntracks_{numTracks}.pdf"
            outpath = os.path.join(out_folder,"Plots")

            make_plots_fastsim(out_path_,args.method,barID,x_low,x_high,outpath,filename,log_norm=True)

    
    print("Making photon yield plots.")
    outpath = os.path.join(out_folder,"Plots",f"{args.method}_Photon_Yield.pdf")
    photon_yield_plots(out_folder,args.method,outpath)

    print("Making ratio plots.")
    outpath = os.path.join(out_folder,"Plots",f"{args.method}_Ratios.pdf")
    make_ratios(out_folder,args.method,outpath)
    # image_folder = os.path.join(out_folder,"Plots")
    # output_pdf = os.path.join(image_folder,"Combined_Plots.pdf")
    # combine_images_to_pdf(image_folder, output_pdf, images_per_page=(2, 2), figure_size=(8, 6))
    

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='GlueX Plotting.')
    parser.add_argument('-c', '--config', default='CA_config.json',type=str,
                        help='Path to the config file (default: CA_config.json)')
    parser.add_argument('-nt', '--n_tracks', default=1e5,type=int,help='Number of particles to generate. Take the first n_tracks.')
    parser.add_argument('-nd', '--n_dump', default=None, type=int, help='Number of particles to dump per .pkl file.')
    parser.add_argument('-m', '--method',default="Pion",type=str,help='Generated particle type, Kaon, Pion.')
    parser.add_argument('-d','--distributed',action='store_true',help='Trained with multiple GPUs - DDP.')
    parser.add_argument('-p','--momentum',default="6",type=str,help='Momentum value, or range.')
    parser.add_argument('-th','--theta',default="30",type=str,help='Theta value, or range.')
    parser.add_argument('-dn','--dark_noise',action='store_true',help='Included hits from dark noise with predefined rate. See source code for more details.')
    parser.add_argument('-tmp','--temperature',default=1.05,type=float,help='Generation temperature.')
    parser.add_argument('-dt', '--dynamic_temperature',action='store_true',help='Use dynamic temperature with predefined values. See source code for more details.')
    parser.add_argument('-s','--sampling',default="Nucleus",type=str,help='Default,TopK,Nucleus')
    parser.add_argument('-tk','--topK',default=300,type=int,help="TopK")
    parser.add_argument('-np','--nucleus_p',default=0.995,type=float,help="Nucleus P value - only used if sampling = Nucleus")
    args = parser.parse_args()

    config = json.load(open(args.config))

    os.makedirs("Generations",exist_ok=True)

    main(config,args)