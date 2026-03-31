import os
import sys 
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn.parallel import DataParallel

from utils.utils import time_loss_fn

import json
import argparse
import random
import numpy as np
import pkbar
import math
import warnings
from datetime import datetime

from dataloader.dataset import DIRC_Dataset_Classification
from dataloader.tokenizer import TimeTokenizer
from dataloader.dataloader import CreateLoaderMoE

from models.GPT import Cherenkov_GPT
import torch.multiprocessing as mp
import torch.distributed as dist

warnings.filterwarnings("ignore", message=".*weights_only.*")


def create_model(config,state_dict=None):
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
    num_experts = config['model']['num_experts']
    num_classes = config['model']['num_classes']
    use_MoE = bool(config['model']['use_MoE'])
    digitize_time = bool(config['digitize_time'])
    
    net = Cherenkov_GPT(vocab_size, msl, embed_dim,attn_heads=attn_heads,kin_size=kin_size,
            num_blocks=num_blocks,hidden_units=hidden_units,digitize_time=digitize_time,mlp_scale=mlp_scale,
            time_vocab=time_vocab,drop_rates=drop_rates,use_MoE=use_MoE,num_experts=num_experts,num_classes=num_classes)

    if state_dict:
        net.load_state_dict(state_dict)

    return net 

class Trainer:
    def __init__(self, config, rank, world_size, model):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.output_folder = config['output']['dir']
        self.exp_name = config['name']
        self.device = torch.device(f"cuda:{rank}")
        self.model = model.to(self.device)
        t_params = sum(p.numel() for p in self.model.parameters())
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[rank])


        self.use_MoE = bool(config['model']['use_MoE'])
        self.digitize_time = bool(config['digitize_time'])
        self.pad_token = config['special_tokens']['pad_token']
        self.time_pad_token = config['special_tokens']['time_pad_token']
        self.time_EOS_token = config['special_tokens']['time_EOS_token']
        self.SOS_token = config['special_tokens']['SOS_token']
        self.EOS_token = config['special_tokens']['EOS_token']
        self.stats = config['stats']
        self.max_seq_length = config['model']['max_seq_length']

        if self.rank == 0:
            print("Network Parameters: ",t_params)
            print("========= Special Tokens ============")
            print(f"Pixels - Pad: {self.pad_token}, SOS: {self.SOS_token}, EOS: {self.EOS_token}")
            print(f"Time   - Pad: {self.time_pad_token}, SOS: {self.SOS_token}, EOS: {self.time_EOS_token}")
            print("=====================================")
            print("Stats: ")
            print("Momentum range: ",self.stats['P_min']," to ",self.stats['P_max'])
            print("Theta range: ",self.stats['theta_min']," to ",self.stats['theta_max'])
            print('Phi range: ',self.stats['phi_min'],' to ',self.stats['phi_max'])

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token)
        self.time_ce = nn.CrossEntropyLoss(ignore_index=self.time_pad_token) if self.digitize_time else None
        self.time_loss_fn = time_loss_fn  # you’ll need to define this or import it

        #self.optimizer = torch.optim.RAdam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=float(config['optimizer']['lr']))
        self.optimizer = torch.optim.AdamW([{'params': self.model.parameters(), 'lr': float(config['optimizer']['lr'])}])
        milestones = [10,20] 
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,gamma=0.1)
        self.history = {'train_loss': [], 'val_loss': []}
        self.global_step = 0
        self.epoch = 0
        self.global_batch_in_epoch = 0

        # Time tokenization
        if self.digitize_time:
            time_res = self.stats['time_res']
            t_max = self.stats['time_max']
            t_min = self.stats['time_min']
            self.time_digitizer = TimeTokenizer(t_max=t_max,t_min=t_min,resolution=time_res)

            if self.rank == 0:
                print("Digitizing time - classification over adjacent vocabulary.")
                print("Time vocab: ",config['model']['time_vocab'])
                print("T_Max: ",t_max," T_Min: ",t_min, "T_Res: ",time_res)
        else:
            self.time_digitizer = None
            if self.rank == 0:
                print("Using regression over time domain.")

    def init_kbar(self,total_samples,num_epochs=1):
        total_batches = total_samples // self.config['dataloader']['train']['batch_size_MoE'] # // self.world_size
        self.kbar = pkbar.Kbar(target=total_batches, epoch=self.epoch, num_epochs=num_epochs, width=20)
            
    def load_chunked_dataset(self, pion_files,kaon_files,verbose=False):
        dataset = ChunkedDataset(pion_files,kaon_files,self.max_seq_length,self.time_digitizer,self.stats,verbose=verbose)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        loader = CreateLoaderMoE(dataset, sampler=sampler, batch_size=self.config['dataloader']['train']['batch_size_MoE'] // self.world_size,
                                num_workers=self.config['dataloader']['train']['num_workers'],
                                pin_memory=False,persistent_workers=False,
                                prefetch_factor=self.config['dataloader']['train']['prefetch_factor'])
        return loader, sampler

    def train_epoch(self, train_loader, sampler):
        self.model.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            tokens = data[0].to(self.device).long()
            times = data[1].to(self.device).long() if self.digitize_time else data[1].to(self.device).float()
            k = data[2].to(self.device).float()
            class_label = data[-1].to(self.device).float() if self.use_MoE else None

            next_tokens = tokens[:, 1:].clone()
            tokens = tokens[:, :-1]
            next_times = times[:, 1:].clone()
            times = times[:, :-1]
            padding_mask = (tokens == self.pad_token).to(self.device)

            self.optimizer.zero_grad()

            logits, t, load_balance = self.model(tokens, times, k, class_label=class_label, padding_mask=padding_mask)
            logits = logits[:, k.shape[1]:, :]
            t = t[:, k.shape[1]:, :]

            pixel_loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), next_tokens.reshape(-1))

            if self.digitize_time:
                time_loss = self.time_ce(t.reshape(-1, t.size(-1)), next_times.reshape(-1))
            else:
                regression_mask = ~torch.isin(next_tokens, torch.tensor([self.pad_token, self.SOS_token, self.EOS_token], device=self.device))
                time_loss = self.time_loss_fn(next_times, t, regression_mask)

            loss = pixel_loss + time_loss + load_balance
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item() * tokens.size(0)

            with torch.no_grad():
                losses = torch.tensor([loss.item(), pixel_loss.item(), time_loss.item(), load_balance.item()],
                                    device=self.device)

                dist.all_reduce(losses, op=dist.ReduceOp.SUM)
                losses /= self.world_size

            if self.rank == 0:
                self.kbar.update(self.global_batch_in_epoch, values=[("loss", losses[0].item()), ("pix", losses[1].item()), ("time", losses[2].item()), ("load", losses[3].item())])

            self.global_batch_in_epoch += 1
            self.global_step += 1

        epoch_loss = running_loss / len(train_loader.dataset)
        self.history['train_loss'].append(epoch_loss)

    def on_epoch_end(self, val_loader=None,write_path=None):
        if val_loader:
            self.model.eval()
            val_pixel_loss = 0.0
            val_time_loss = 0.0
            with torch.no_grad():
                for i,data in enumerate(val_loader):
                    tokens = data[0].to(self.device).long()
                    times = data[1].to(self.device).long() if self.digitize_time else data[1].to(self.device).float()
                    k = data[2].to(self.device).float()
                    class_label = data[-1].to(self.device).float() if self.use_MoE else None

                    next_tokens = tokens[:, 1:].clone()
                    tokens = tokens[:, :-1]
                    next_times = times[:, 1:].clone()
                    times = times[:, :-1]
                    padding_mask = (tokens == self.pad_token).to(self.device)

                    logits, t = self.model(tokens, times, k, class_label=class_label, padding_mask=padding_mask)
                    logits = logits[:, k.shape[1]:, :]
                    t = t[:, k.shape[1]:, :]

                    if self.digitize_time:
                        val_time_loss += self.time_ce(t.reshape(-1, t.size(-1)), next_times.reshape(-1))
                    else:
                        regression_mask = ~torch.isin(next_tokens, torch.tensor([self.pad_token, self.SOS_token, self.EOS_token], device=self.device))
                        val_time_loss += self.time_loss_fn(next_times, t, regression_mask)

                    val_pixel_loss += self.loss_fn(logits.reshape(-1, logits.size(-1)), next_tokens.reshape(-1))

            val_time_loss /= len(val_loader)
            val_pixel_loss /= len(val_loader)
            val_loss = val_pixel_loss + val_time_loss
            self.history['val_loss'].append(val_loss.item())
        else:
            val_loss = torch.tensor(0.0)
            val_pixel_loss = torch.tensor(0.0)
            val_time_loss = torch.tensor(0.0)

        with torch.no_grad():
            losses = torch.tensor([val_loss.item(), val_pixel_loss.item(), val_time_loss.item()],
                                device=self.device)

            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
            losses /= self.world_size

        if self.rank == 0:
            self.kbar.add(1, values=[("Val_loss", losses[0].item()),("val_pix",losses[1].item()),("val_time",losses[2].item())])
            #print(f"\n[Epoch {self.epoch}] Val Loss: {losses[0].item():.6f} | Val Pix: {losses[1].item():.6f} | Val Time: {losses[2].item():.6f}")        
            sys.stdout.flush()  
            filename = os.path.join(write_path, f'Epoch{self.epoch:02d}_loss_{val_loss:.6f}.pth')
            torch.save({
                'net_state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': self.epoch,
                'history': self.history,
                'global_step': self.global_step,
            }, filename)

def run_worker(rank, world_size, config, all_train_files, all_val_files, state_dict=None, run_val=True, write_path=None, checkpoint=None):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    if rank == 0:
        print("Running validation: ", run_val)

    train_pions,train_kaons = all_train_files
    val_pions,val_kaons = all_val_files
    val_pions = np.array(val_pions)
    val_kaons = np.array(val_kaons)

    num_epochs = config['num_epochs_MoE'] if config['model']['use_MoE'] else config['num_epochs']


    model = create_model(config, state_dict)
    trainer = Trainer(config, rank, world_size, model)

    if checkpoint is not None:
        if 'net_state_dict' in checkpoint:
            trainer.model.module.load_state_dict(checkpoint['net_state_dict'])
            print(f"Rank {rank} - Loaded model state from checkpoint.")
        if 'optimizer' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer'])
            trainer.epoch = checkpoint.get('epoch', 0) + 1
            trainer.history = checkpoint.get('history', trainer.history)
            trainer.global_step = checkpoint.get('global_step', 0)
            print(f"Rank {rank} - Loaded optimizer state from checkpoint, starting at epoch {trainer.epoch}.")
        else:
            trainer.epoch = 0
            trainer.global_step = 0
            trainer.history = {'train_loss': [], 'val_loss': []}
    
    trainer.global_batch_in_epoch = 0

    pion_path,kaon_path = all_train_files
    val_pion_path,val_kaon_path = all_val_files

    msl = config['model']['max_seq_length']

    if config['digitize_time']:
        time_res = config['stats']['time_res']
        t_max = config['stats']['time_max']
        t_min = config['stats']['time_min']
        time_digitizer = TimeTokenizer(t_max=t_max, t_min=t_min, resolution=time_res)
    else:
        time_digitizer = None


    train_dataset = DIRC_Dataset_Classification(pion_path=pion_path,kaon_path=kaon_path,max_seq_length=msl,time_digitizer=time_digitizer,stats=config['stats'])
    val_dataset = DIRC_Dataset_Classification(pion_path=val_pion_path,kaon_path=val_kaon_path,max_seq_length=msl,time_digitizer=time_digitizer,stats=config['stats'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = CreateLoaderMoE(train_dataset, sampler=train_sampler, batch_size=config['dataloader']['train']['batch_size_MoE'] // world_size, num_workers=2, pin_memory=False,prefetch_factor=1)
    val_loader = CreateLoaderMoE(val_dataset, sampler=val_sampler, batch_size=config['dataloader']['train']['batch_size_MoE'] // world_size, num_workers=2, pin_memory=False,prefetch_factor=1)

    print(f"Rank {rank} - Starting training with {len(train_dataset)} events, num epochs: {num_epochs}, batch size: {config['dataloader']['train']['batch_size_MoE'] // world_size}")

    for epoch in range(trainer.epoch,num_epochs):
        if rank == 0:
            print("Learning rate: ", trainer.scheduler.get_last_lr()[0])

        trainer.epoch = epoch
        trainer.global_batch_in_epoch = 0  
        trainer.init_kbar(len(train_dataset),num_epochs)
        torch.cuda.empty_cache()  
        gc.collect()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        trainer.train_epoch(train_loader, train_sampler)

        trainer.scheduler.step()

        if run_val:
            trainer.on_epoch_end(val_loader,write_path)
        else:
            trainer.on_epoch_end(None,write_path)
    
    dist.destroy_process_group()

def main(config,resume):
    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # Create experiment name
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    exp_name = exp_name[:-11]
    print(exp_name)

    # Create directory structure
    output_folder = config['output']['dir']
    os.makedirs(os.path.join(output_folder,exp_name),exist_ok=True)
    write_path = os.path.join(output_folder,exp_name)
    with open(os.path.join(output_folder,exp_name,'config.json'),'w') as outfile:
        json.dump(config, outfile)

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print("Conditional generation with MoE - singular generative model training.")

    train_pions = config['dataset']['training']['pion_data_path']
    train_kaons = config['dataset']['training']['kaon_data_path']
    val_pions = config['dataset']['validation']['pion_data_path']
    val_kaons = config['dataset']['validation']['kaon_data_path']

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    checkpoint = None if resume is None else torch.load(resume, map_location='cpu')
    
    run_worker(rank,world_size, config, (train_pions,train_kaons), (val_pions,val_kaons),
               write_path=write_path,
               checkpoint=checkpoint)

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Generative Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args.resume)

