from comet_ml import Experiment, ExistingExperiment

from data.datasets import MonostyleDataset, ParallelRefDataset
from cyclegan_tst.models.CycleGANModel import CycleGANModel
from cyclegan_tst.models.GeneratorModel import GeneratorModel
from cyclegan_tst.models.DiscriminatorModel import DiscriminatorModel
from cyclegan_tst.models.ClassifierModel import ClassifierModel
from eval import *
from utils.utils import *

import argparse
import logging
from tqdm import tqdm
import os, sys, time
import pickle
import numpy as np, pandas as pd
import random

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler

logging.basicConfig(level=logging.INFO)

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

''' 
    ----- ----- ----- ----- ----- ----- ----- -----
                    PARSING PARAMs       
    ----- ----- ----- ----- ----- ----- ----- -----
'''

parser = argparse.ArgumentParser()

# basic parameters
parser.add_argument('--style_a', type=str, dest="style_a", help='style A for the style transfer task (source style for G_ab).')
#parser.add_argument('--style_b', type=str, dest="style_b", help='style B for the style transfer task (target style for G_ab).')
parser.add_argument('--style_list', type=str, dest="style_list", help='Style list of all possible styles')

parser.add_argument('--lang', type=str, dest="lang", default='en', help='Dataset language.')
parser.add_argument('--max_samples_train', type=int, dest="max_samples_train", default=None, help='Max number of examples to retain from the training set. None for all available examples.')
parser.add_argument('--max_samples_eval',  type=int, dest="max_samples_eval",  default=None, help='Max number of examples to retain from the evaluation set. None for all available examples.')
parser.add_argument('--nonparal_same_size', action='store_true', dest="nonparal_same_size",  default=True, help='Whether to reduce non-parallel data to same size.')

parser.add_argument('--path_train', type=str, dest="path_train", help='Path to non-parallel dataset folder for train. (All Styles, both sources and targets)')
#parser.add_argument('--path_mono_A', type=str, dest="path_mono_A", help='Path to monostyle dataset (style A) for training.')
#parser.add_argument('--path_mono_B', type=str, dest="path_mono_B", help='Path to monostyle dataset (style B) for training.')


parser.add_argument('--path_eval', type=str, dest="path_eval", help='Path to non-parallel dataset folder for eval. (All Styles, both sources and targets)')

#parser.add_argument('--path_mono_A_eval', type=str, dest="path_mono_A_eval", help='Path to non-parallel dataset (style A) for evaluation.')
#parser.add_argument('--path_mono_B_eval', type=str, dest="path_mono_B_eval", help='Path to non-parallel dataset (style B) for evaluation.')
#parser.add_argument('--path_paral_A_eval', type=str, dest="path_paral_A_eval", help='Path to parallel dataset (style A) for evaluation.')
#parser.add_argument('--path_paral_B_eval', type=str, dest="path_paral_B_eval", help='Path to parallel dataset (style B) for evaluation.')


#parser.add_argument('--path_paral_eval_ref', type=str, dest="path_paral_eval_ref", help='Path to human references for evaluation.')
#parser.add_argument('--n_references',  type=int, dest="n_references",  default=None, help='Number of human references for evaluation.')
#parser.add_argument('--lowercase_ref', action='store_true', dest="lowercase_ref", default=False, help='Whether to lowercase references.')
#parser.add_argument('--bertscore', action='store_true', dest="bertscore", default=True, help='Whether to compute BERTScore metric.')

parser.add_argument('--max_sequence_length', type=int,  dest="max_sequence_length", default=64, help='Max sequence length')

# Training arguments
parser.add_argument('--batch_size', type=int,  dest="batch_size",  default=64,     help='Batch size used during training.')
parser.add_argument('--shuffle',    action='store_true', dest="shuffle",     default=False, help='Whether to shuffle the training/eval set or not.')
parser.add_argument('--num_workers',type=int,  dest="num_workers", default=2,     help='Number of workers used for dataloaders.')
parser.add_argument('--pin_memory', action='store_true', dest="pin_memory",  default=False, help='Whether to pin memory for data on GPU during data loading.')

parser.add_argument('--use_cuda_if_available', action='store_true', dest="use_cuda_if_available", default=False, help='Whether to use GPU if available.')

parser.add_argument('--learning_rate',     type=float, dest="learning_rate",     default=5e-5,     help='Initial learning rate (e.g., 5e-5).')
parser.add_argument('--epochs',            type=int,   dest="epochs",            default=10,       help='The number of training epochs.')
parser.add_argument('--lr_scheduler_type', type=str,   dest="lr_scheduler_type", default="linear", help='The scheduler used for the learning rate management.')
parser.add_argument('--warmup', action='store_true', dest="warmup", default=False, help='Whether to apply warmup.')
parser.add_argument('--lambdas', type=str,   dest="lambdas", default="1|1|1|1|1|1", help='Lambdas for loss-weighting.')

parser.add_argument('--generator_model_tag', type=str, dest="generator_model_tag", help='The tag of the model for the generator (e.g., "facebook/bart-base").')
parser.add_argument('--discriminator_model_tag', type=str, dest="discriminator_model_tag", help='The tag of the model discriminator (e.g., "distilbert-base-cased").')
parser.add_argument('--pretrained_classifier_model', type=str, dest="pretrained_classifier_model", help='The folder to use as base path to load the pretrained classifier for classifier-guided loss.')
parser.add_argument('--pretrained_classifier_eval', type=str, dest="pretrained_classifier_eval", help='The folder to use as base path to load the pretrained classifier for metrics evaluation.')

# arguments for saving the model and running evaluation
parser.add_argument('--save_base_folder', type=str, dest="save_base_folder", help='The folder to use as base path to store model checkpoints')
parser.add_argument('--from_pretrained', type=str, dest="from_pretrained", default=None, help='The folder to use as base path to load model checkpoints')
parser.add_argument('--save_steps',       type=int, dest="save_steps",       help='How many training epochs between two checkpoints.')
parser.add_argument('--eval_strategy',    type=str, dest="eval_strategy",    help='Evaluation strategy for the model (either epochs or steps)') #epochs lo fa una volta alla fine 
parser.add_argument('--eval_steps',       type=int, dest="eval_steps",       help='How many training steps between two evaluations.')
parser.add_argument('--additional_eval',       type=int, dest="additional_eval", default=0, help='Whether to perform evaluation at the half of the first N epochs.')

# temporary arguments to control execution
parser.add_argument('--control_file', type=str, dest="control_file", default=None, help='The path of the file to control execution (e.g., whether to stop)')
parser.add_argument('--lambda_file', type=str, dest="lambda_file", default=None, help='The path of the file to define lambdas')

# arguments for comet
parser.add_argument('--comet_logging', action='store_true', dest="comet_logging",   default=False, help='Set flag to enable comet logging')
parser.add_argument('--comet_key',       type=str,  dest="comet_key",       default=None,  help='Comet API key to log some metrics')
parser.add_argument('--comet_workspace', type=str,  dest="comet_workspace", default=None,  help='Comet workspace name (usually username in Comet, used only if comet_key is not None)')
parser.add_argument('--comet_project_name',  type=str,  dest="comet_project_name",  default=None,  help='Comet experiment name (used only if comet_key is not None)')
parser.add_argument('--comet_exp',  type=str,  dest="comet_exp",  default=None,  help='Comet experiment key to continue logging (used only if comet_key is not None)')

args = parser.parse_args()

style_a = args.style_a
style_list = args.style_list.split(',') #Python list of possible styles 
#style_b = args.style_b


max_samples_train = args.max_samples_train
max_samples_eval = args.max_samples_eval

if args.lambda_file is not None:
    while not os.path.exists(args.lambda_file):
        time.sleep(60)
    with open(args.lambda_file, 'r') as f:
        args.lambdas = f.read()
    os.remove(args.lambda_file)

hyper_params = {}
print ("Arguments summary: \n ")
for key, value in vars(args).items():
    hyper_params[key] = value
    print (f"\t{key}:\t\t{value}")

# lambdas: cycle-consistency, generator-fooling, disc-fake, disc-real, classifier-guided
lambdas = [float(l) for l in args.lambdas.split('|')]
args.lambdas = lambdas

# Creare un dizionario per memorizzare dataset e dataloader
datasets_train = {}
dataloaders_train = {}

#   Iterare attraverso ogni stile nella lista
for style in style_list:
    dataset_path_train = os.path.join(args.path_train, f"{style}_train.csv")
    
    # Creazione del dataset per lo stile corrente
    datasets_train[style] = MonostyleDataset(
        dataset_format="line_file",
        style=style,
        dataset_path=dataset_path_train,
        separator='\n',
        max_dataset_samples=args.max_samples_test
    )



if args.nonparal_same_size:
    # Ottieni i nomi degli stili disponibili
    style_names = list(datasets_train.keys())
    
    # Assicurati che ci siano almeno due stili da confrontare
    if len(style_names) < 2:
        raise ValueError("Per bilanciare i dataset, devono esserci almeno due stili.")

    # Ottieni la lunghezza di ciascun dataset
    dataset_lengths = {style: len(datasets_train[style]) for style in style_names}
    print(f"Train dataset lengths by style: {dataset_lengths.items()}")
    
    # Trova la lunghezza minima tra i dataset
    min_length = min(dataset_lengths.values())
    
    # Riduci ciascun dataset alla lunghezza minima
    for style in style_names:
        datasets_train[style].reduce_data(min_length)
        print(f"Train dataset {style} ridotto a {min_length} campioni.")

for style in style_list:
    dataset_path_train = os.path.join(args.path_train, f"{style}_train.csv")

    # Creazione del dataloader per lo stile corrente
    dataloaders_train[style] = DataLoader(
        datasets_train[style],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

# Itera attraverso i dataloader per stampare il numero di batch
print ("Training Dataloader Lenghts(batches) by style: ")
for style, dataloader in dataloaders_train.items():
    print(f"{style.capitalize()}: {len(dataloader)}")

print()

# Creare un dizionario per memorizzare dataset e dataloader
datasets_eval = {}
dataloaders_eval = {}

#   Iterare attraverso ogni stile nella lista
for style in style_list:
    dataset_path_eval = os.path.join(args.path_train, f"{style}_train.csv")
    
    # Creazione del dataset per lo stile corrente
    datasets_eval[style] = MonostyleDataset(
        dataset_format="line_file",
        style=style,
        dataset_path=dataset_path_eval,
        separator='\n',
        max_dataset_samples=args.max_samples_test
    )



if args.nonparal_same_size:
    # Ottieni i nomi degli stili disponibili
    style_names = list(datasets_eval.keys())

    # Assicurati che ci siano almeno due stili da confrontare
    if len(style_names) < 2:
        raise ValueError("Per bilanciare i dataset, devono esserci almeno due stili.")

    # Ottieni la lunghezza di ciascun dataset
    dataset_lengths = {style: len(datasets_eval[style]) for style in style_names}
    print(f"Eval dataset lengths by style: {dataset_lengths.items()}")

    # Trova la lunghezza minima tra i dataset
    min_length = min(dataset_lengths.values())

    # Riduci ciascun dataset alla lunghezza minima
    for style in style_names:
        datasets_eval[style].reduce_data(min_length)
        print(f"Eval dataset {style} ridotto a {min_length} campioni.")

for style in style_list:
    dataset_path_eval = os.path.join(args.path_train, f"{style}_train.csv")

    # Creazione del dataloader per lo stile corrente
    dataloaders_eval[style] = DataLoader(
        datasets_eval[style],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

# Itera attraverso i dataset per stampare la lunghezza

print ("Eval Dataloader Lenghts(batches) by style: ")
# Itera attraverso i dataloader per stampare il numero di batch
for style, dataloader in dataloaders_eval.items():
    print(f"{style.capitalize()}: {len(dataloader)}")

print()


# Pulire i dataset per risparmiare memoria, se non più necessari
for style in style_list:
    del datasets_train[style]
    del datasets_eval[style]




''' 
    ----- ----- ----- ----- ----- ----- ----- -----
              Instantiate Generators       
    ----- ----- ----- ----- ----- ----- ----- -----
'''

if args.from_pretrained is not None:
    G_ab = GeneratorModel(args.generator_model_tag, f'{args.from_pretrained}G_ab/', max_seq_length=args.max_sequence_length)
    G_ba = GeneratorModel(args.generator_model_tag, f'{args.from_pretrained}G_ba/', max_seq_length=args.max_sequence_length)
    print('Generator pretrained models loaded correctly')
else:
    G_ab = GeneratorModel(args.generator_model_tag, max_seq_length=args.max_sequence_length)
    G_ba = GeneratorModel(args.generator_model_tag, max_seq_length=args.max_sequence_length)
    print('Generator pretrained models not loaded - Initial weights will be used')


''' 
    ----- ----- ----- ----- ----- ----- ----- -----
             Instantiate Discriminators       
    ----- ----- ----- ----- ----- ----- ----- -----
'''

if args.from_pretrained is not None:
    D_ab = DiscriminatorModel(args.discriminator_model_tag, f'{args.from_pretrained}D_ab/', max_seq_length=args.max_sequence_length)
    D_ba = DiscriminatorModel(args.discriminator_model_tag, f'{args.from_pretrained}D_ba/', max_seq_length=args.max_sequence_length)
    print('Discriminator pretrained models loaded correctly')
else:
    D_ab = DiscriminatorModel(args.discriminator_model_tag, max_seq_length=args.max_sequence_length)
    D_ba = DiscriminatorModel(args.discriminator_model_tag, max_seq_length=args.max_sequence_length)
    print('Discriminator pretrained models not loaded - Initial weights will be used')


''' 
    ----- ----- ----- ----- ----- ----- ----- -----
             Instantiate Classifier       
    ----- ----- ----- ----- ----- ----- ----- -----
'''

if lambdas[4] != 0:
    Cls = ClassifierModel(args.pretrained_classifier_model, max_seq_length=args.max_sequence_length)
    print('Classifier pretrained model loaded correctly')
else:
    Cls = None


''' 
    ----- ----- ----- ----- ----- ----- ----- -----
                    SETTINGS       
    ----- ----- ----- ----- ----- ----- ----- -----
'''

if args.use_cuda_if_available:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    device = torch.device("cpu")

cycleGAN = CycleGANModel(G_ab, G_ba, D_ab, D_ba, Cls, device=device)

#n_batch_epoch = min(len(mono_dl_a), len(mono_dl_b))

# Calcolare il numero minimo di batch tra tutti i dataloader
n_batch_epoch = min(len(dataloader) for dataloader in dataloaders_train.values())

print(f"Numero minimo di batch per epoca considerando tutti gli stili: {n_batch_epoch}")


num_training_steps = args.epochs * n_batch_epoch

print(f"Total number of training steps: {num_training_steps}")

warmup_steps = int(0.1*num_training_steps) if args.warmup else 0

optimizer = AdamW(cycleGAN.get_optimizer_parameters(), lr=args.learning_rate)
# scheduler types: ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
lr_scheduler = get_scheduler(args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
#ATTENZIONE: con training multistyle il LR_SCHEDULER dovrebbe andare più piano - nel caso di ML il migliore era risultato lo scheduler cosine_with_restarts
    
start_epoch = 0
current_training_step = 0

if args.from_pretrained is not None:
    checkpoint = torch.load(f"{args.from_pretrained}checkpoint.pth", map_location=torch.device("cpu"))
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    start_epoch = checkpoint['epoch']
    current_training_step = checkpoint['training_step']
    del checkpoint

''' 
    ----- ----- ----- ----- ----- ----- ----- -----
                 COMET LOGGING SETUP
    ----- ----- ----- ----- ----- ----- ----- -----
'''

if args.comet_logging:
    if args.from_pretrained is not None:
        experiment = ExistingExperiment(api_key=args.comet_key, previous_experiment=args.comet_exp)
    else:
        experiment = Experiment(
            api_key=args.comet_key,
            project_name=args.comet_project_name,
            workspace=args.comet_workspace,
        )
    experiment.log_parameters(hyper_params)
else:
    experiment = None

loss_logging = {'Cycle Loss A-B-A':[], 'Loss generator  A-B':[], 'Classifier-guided A-B':[], 'Loss D(A->B)':[],
                'Cycle Loss B-A-B':[], 'Loss generator  B-A':[], 'Classifier-guided B-A':[], 'Loss D(B->A)':[]}
loss_logging['hyper_params'] = hyper_params

''' 
    ----- ----- ----- ----- ----- ----- ----- -----
                    TRAINING LOOP       
    ----- ----- ----- ----- ----- ----- ----- -----
'''

progress_bar = tqdm(range(num_training_steps))
progress_bar.update(current_training_step)

evaluator = Evaluator(cycleGAN, args, experiment)


print('Start training...')
for epoch in range(start_epoch, args.epochs):
    print (f"\nTraining epoch: {epoch}")
    cycleGAN.train() # set training mode

    # Recupera il dataloader principale (dl_a) e gli altri dataloader
    dataloader_a = dataloaders_train[style_a]
    other_dataloaders = {style: dataloader for style, dataloader in dataloaders_train.items() if style != style_a}

    # Numero massimo di batch (assumiamo che dl_a determini il limite)
    max_batches = len(dataloader_a)

    # Creiamo un iteratore per ogni dataloader
    dataloader_iterators = {style: iter(dataloader) for style, dataloader in dataloaders_train.items()}

    # Ciclo principale
    for i in range(max_batches):
        try:
            # Estrai un batch da dl_a
            unsupervised_a = next(dataloader_iterators['style_a'])
        except StopIteration:
            # Se il dataloader principale termina, interrompi il ciclo
            break

        # Sorteggia un dataloader casuale tra gli altri
        chosen_style = random.choice(list(other_dataloaders.keys()))
        chosen_iterator = dataloader_iterators[chosen_style]

        try:
            # Estrai un batch dal dataloader sorteggiato
            unsupervised_b = next(chosen_iterator)
        except StopIteration:
            # Se il dataloader scelto termina, rigenera il suo iteratore
            dataloader_iterators[chosen_style] = iter(other_dataloaders[chosen_style])
            unsupervised_b = next(dataloader_iterators[chosen_style])

        # Assicura che i batch abbiano la stessa lunghezza
        len_a, len_b = len(unsupervised_a), len(unsupervised_b)
        if len_a > len_b:
            unsupervised_a = unsupervised_a[:len_b]
        elif len_b > len_a:
            unsupervised_b = unsupervised_b[:len_a]

        # Training step
        cycleGAN.training_cycle(
            sentences_a=unsupervised_a,
            sentences_b=unsupervised_b,
            sentences_a_style: style_a,
            sentences_b_style: chosen_style,
            lambdas=lambdas,
            comet_experiment=experiment,
            loss_logging=loss_logging,
            training_step=current_training_step
        )

        # Aggiorna i parametri
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        current_training_step += 1

        # Dummy metrics e valutazione
        if current_training_step == 5: #dopo 5 batch fa una prova di memoria
            evaluator.dummy_classif()
        if (args.eval_strategy == "steps" and current_training_step % args.eval_steps == 0) or (     #la prima condizione controllo la eval strategy, se epochs valida il modello solo una 
            epoch < args.additional_eval and current_training_step % (max_batches // 2 + 1) == 0 ):  #volta alla fine di ogni epoca, se è steps possiamo indicare noi ogni quanti batch
                                                                                                     #fare la validazione, la seconda condizione è se fare una validazione a metà delle prime additional_eval epoche
            
            # Esegui `run_eval_mono` intermedia secondo eval_strategy per tutte le coppie <mono_dl_a_test, ..>
            styles_to_evaluate = [style for style in style_list if style != style_a]
            total_metrics_eval = {}
            for style in styles_to_evaluate:
                mono_dl_other_test = dataloaders_eval[style]  # Recupera il dataloader dello stile corrente
                evaluator.run_eval_mono(epoch, current_training_step, 'validation', dataloaders_eval[style_a], mono_dl_other_test)
                total_metrics_eval[f'{style_a}-{style}'] = evaluator.run_eval_mono_multistyle(epoch, epoch, 'validation',
                                                                                         dataloaders_eval[style_a],
                                                                                         mono_dl_other_test, style_a,
                                                                                         style)
            metrics_df = pd.DataFrame(total_metrics_eval).T
            metrics_df['Mean'] = metrics_df.mean(axis=1)
            print(metrics_df)

            cycleGAN.train()
        

    
    if epoch%args.save_steps==0:
        cycleGAN.save_models(f"{args.save_base_folder}epoch_{epoch}/")
        checkpoint = {'epoch':epoch+1, 'training_step':current_training_step, 'optimizer':optimizer.state_dict(), 'lr_scheduler':lr_scheduler.state_dict()}
        torch.save(checkpoint, f"{args.save_base_folder}epoch_{epoch}/checkpoint.pth")
        if epoch > 0 and os.path.exists(f"{args.save_base_folder}epoch_{epoch-1}/checkpoint.pth"):
            os.remove(f"{args.save_base_folder}epoch_{epoch-1}/checkpoint.pth")
        if epoch > 0 and os.path.exists(f"{args.save_base_folder}loss.pickle"):
            os.remove(f"{args.save_base_folder}loss.pickle")
        pickle.dump(loss_logging, open(f"{args.save_base_folder}loss.pickle", 'wb'))
    if args.control_file is not None and os.path.exists(args.control_file):
        with open(args.control_file, 'r') as f:
            if f.read() == 'STOP':
                print(f'STOP command received - Stopped at epoch {epoch}')
                os.remove(args.control_file)
                break

    # Final `run_eval_mono` per tutte le coppie <mono_dl_a_test, ..>
    styles_to_evaluate = [style for style in style_list if style != style_a]
    total_metrics_eval = {}
    for style in styles_to_evaluate:
                mono_dl_other_test = dataloaders_eval[style]  # Recupera il dataloader dello stile corrente
                total_metrics_eval[f'{style_a}-{style}'] = evaluator.run_eval_mono_multistyle(epoch, epoch, 'validation',dataloaders_eval[style_a], mono_dl_other_test, style_a,style)

    metrics_df = pd.DataFrame(total_metrics_eval).T
    metrics_df['Mean'] = metrics_df.mean(axis=1)
    print(metrics_df)

    if args.comet_logging:
        experiment.log_table("final_metrics.csv", metrics_df)

    cycleGAN.train()
print('End training...')
