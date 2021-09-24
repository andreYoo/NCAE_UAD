import click
import torch
import logging
import random
import numpy as np

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid,plot_multiple_images_grid
from NCAE_UAD import NCAE_UAD
from datasets.main import load_dataset

from tensorboardX import SummaryWriter
import time
import os

polution_ratio_dict = {
    'arrhythmia':0.146,
    'cardio':0.096,
    'satellite':0.316,
    'satimage-2':0.012,
    'shuttle':0.071,
    'thyroid':0.0245,
    'mnist':0.5,
    'fmnist':0.5,
    'cifar10':0.5
}



################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'fmnist', 'cifar10', 'arrhythmia', 'cardio', 'satellite',
                                                   'satimage-2', 'shuttle', 'thyroid']))
@click.argument('net_name', type=click.Choice(['mnist_LeNet', 'fmnist_LeNet', 'cifar10_LeNet', 'arrhythmia_mlp',
                                               'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
                                               'thyroid_mlp']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--ratio_known_normal', type=float, default=0.0,
              help='Ratio of known (labeled) normal training examples.')
@click.option('--ratio_known_outlier', type=float, default=0.01,
              help='Ratio of known (labeled) anomalous training examples.')
@click.option('--ratio_pollution', type=float, default=0.0,
              help='Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--gan_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--lr_milestone', default=[20,50,70,90], multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--std', type=float, default=0.1,
              help='Standard deviation for GAN(0-1).')
@click.option('--lamdba', type=float, default=0.1,
              help='balancing weight for gan loss and recon loss.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--num_threads', type=int, default=0,
              help='Number of threads used for parallelizing CPU operations. 0 means that all resources are used.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--known_outlier_class', type=int, default=1,
              help='Specify the known outlier class of the dataset for semi-supervised anomaly detection.')
@click.option('--n_known_outlier_classes', type=int, default=1,
              help='Number of known outlier classes.'
                   'If 0, no anomalies are known.'
                   'If 1, outlier class as specified in --known_outlier_class option.'
                   'If > 1, the specified number of outlier classes will be sampled at random.')
@click.option('--monitor', type=bool, default=True,
              help='Moniting learnig process using Tensorboard')

def main(dataset_name, net_name, xp_path, data_path, load_config, load_model,
         ratio_known_normal, ratio_known_outlier, ratio_pollution, device, seed,std,lamdba,
         optimizer_name, lr,gan_lr, n_epochs, lr_milestone, batch_size, weight_decay,
         num_threads, n_jobs_dataloader, normal_class, known_outlier_class, n_known_outlier_classes,monitor):
    """
    Deep SAD, a method for deep semi-supervised anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if ratio_pollution==-1:
        ratio_pollution = polution_ratio_dict[dataset_name]
        logger.info('Pollution ratio of unlabeled train data: %.4f (Max)' % ratio_pollution)

    if monitor==True:
        _time_rightnow = time.strftime('/%y%m%d_%H:%M:%S')
        log_file = xp_path + _time_rightnow
        if not os.path.exists(log_file):
            os.makedirs(log_file)
            logger.info('NO Dir for learning monitoring - It is generated in %s' % log_file)
            writter = SummaryWriter(log_file)


    # Print experimental setup
    logger.info('[Experiment detils]-------------------------------------------------------------------------')
    logger.info('Log file is %s' % log_file)
    logger.info('Data path is %s' % data_path)
    logger.info('Export path is %s' % xp_path)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)


    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    # Set the number of threads used for parallelizing CPU operations
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    logger.info('Computation device: %s' % device)
    logger.info('Number of threads: %d' % num_threads)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, ratio_pollution,
                           random_state=np.random.RandomState(cfg.settings['seed']))
    # Log random sample of known anomaly classes if more than 1 class
    if n_known_outlier_classes > 1:
        logger.info('Known anomaly classes: %s' % (dataset.known_outlier_classes,))

    ncae_UAD = NCAE_UAD(writter,normal_class)
    ncae_UAD.set_network(net_name)


    # If specified, load Deep SAD model (center c, network weights, and possibly autoencoder weights)
    if load_model:
        ncae_UAD.load_model(model_path=load_model, load_ae=True, map_location=device)
        logger.info('Loading model from %s.' % load_model)


    # Log pretraining details
    logger.info('[Training Parameter setting]-----------------------------------------------------------------')
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    if ratio_pollution==-1:
        logger.info('Pollution ratio of unlabeled train data: %.4f (Max)' % ratio_pollution)
    else:
        logger.info('Pollution ratio of unlabeled train data: %.4f' % ratio_pollution)
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])
    logger.info('Training learning rate for GAN: %f' % cfg.settings['gan_lr'])
    logger.info('Std for GAN: %f' % cfg.settings['std'])
    logger.info('Balancing weight (lambda): %f' % cfg.settings['lamdba'])

    # Pretrain model on dataset (via autoencoder)
    ncae_UAD.train(dataset,
                     optimizer_name=cfg.settings['optimizer_name'],
                     lr=cfg.settings['lr'],
                     gan_lr = cfg.settings['gan_lr'],
                     n_epochs=cfg.settings['n_epochs'],
                     lr_milestones=cfg.settings['lr_milestone'],
                     batch_size=cfg.settings['batch_size'],
                     weight_decay=cfg.settings['weight_decay'],
                     std = cfg.settings['std'],
                     lamdba = cfg.settings['lamdba'],
                     device=device,
                     n_jobs_dataloader=n_jobs_dataloader)


    # Save results, model, and configuration
    ncae_UAD.save_results(export_json=xp_path + '/results.json')
    ncae_UAD.save_model(export_model=xp_path + '/model.tar')
    cfg.save_config(export_json=xp_path + '/config.json')

    # Plot most anomalous and most normal test samples
    indices, labels, scores = zip(*ncae_UAD.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    idx_all_sorted = indices[np.argsort(scores)]  # from lowest to highest score
    idx_normal_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # from lowest to highest score

    if dataset_name in ('mnist', 'fmnist', 'cifar10'):
        if dataset_name in ('mnist', 'fmnist'):
            X_all_low = dataset.test_set.data[idx_all_sorted[:32], ...].unsqueeze(1)
            X_all_high = dataset.test_set.data[idx_all_sorted[-32:], ...].unsqueeze(1)
            X_normal_low = dataset.test_set.data[idx_normal_sorted[:32], ...].unsqueeze(1)
            X_normal_high = dataset.test_set.data[idx_normal_sorted[-32:], ...].unsqueeze(1)

        if dataset_name == 'cifar10':
            X_all_low = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[:32], ...], (0,3,1,2)))
            X_all_high = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[-32:], ...], (0,3,1,2)))
            X_normal_low = torch.tensor(np.transpose(dataset.test_set.data[idx_normal_sorted[:32], ...], (0,3,1,2)))
            X_normal_high = torch.tensor(np.transpose(dataset.test_set.data[idx_normal_sorted[-32:], ...], (0,3,1,2)))

        # plot_images_grid('../images/all_low_%d_cls.png'%(normal_class), [X_all_low],title='All low', padding=2)
        # plot_images_grid('../images/all_high_%d_cls.png'%(normal_class), X_all_high ,title='All high', padding=2)
        # plot_images_grid('../images/normals_low_%d_cls.png'%(normal_class), X_normal_low,title='Normal low', padding=2)
        # plot_images_grid('../images/normals_high_%d_cls.png'%(normal_class),X_normal_high,title='Normal high', padding=2)
        plot_multiple_images_grid('../images/Analysis_%d_cls.png'%(normal_class),[X_all_low,X_all_high,X_normal_low,X_normal_high],title='%d cls error analysis'%(normal_class),subtitle=['All low','All high','Normal low','Normal high'],padding=2)


if __name__ == '__main__':
    main()
