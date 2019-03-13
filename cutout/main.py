import click

from torch import nn
from torchvision import transforms
from torch.utils.data import random_split

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Precision, Recall, RunningAverage, Loss
from ignite.handlers import ModelCheckpoint, EarlyStopping, TerminateOnNan
from ignite.contrib.handlers import ProgressBar

from cutout import model, dataset


@click.group()
def cli():
    pass

@cli.command()
@click.option('-n', '--name', default='model', help='prefix for checkpoint file names')
@click.option('-i', '--load', default=None, type=click.Path(exists=True, readable=True), help='pretrained weights to load')
@click.option('-l', '--lrate', default=0.0001, help='initial learning rate')
@click.option('--weight-decay', default=1e-5, help='weight decay')
@click.option('-d', '--device', default='cpu', help='pytorch device')
@click.option('-r', '--refine-features/--freeze-features', default=False, help='Freeze pretrained feature weights')
@click.option('--lag', show_default=True, default=20, help='Number of epochs to wait before stopping training without improvement')
@click.option('--min-delta', show_default=True, default=0.005, help='Minimum improvement between epochs to reset early stopping')
@click.option('--threads', default=min(len(os.sched_getaffinity(0)), 4))
@click.argument('ground_truth', nargs=1)
def train(name, load, lrate, weight_decay, device, validation, refine_features,
          lag, min_delta, threads, weigh_loss, augment, ground_truth):


    print('model output name: {}'.format(name))
    torch.set_num_threads(threads)

    data_set = dataset.CutoutDataset(ground_truth)

    train_split = len(data_set)*0.9
    train_set, val_set = random_split(data_set, [train_split, len(data_set)-train_split])
    train_data_loader = DataLoader(dataset=train_set, num_workers=workers, batch_size=1, shuffle=True, pin_memory=True)
    val_data_loader = DataLoader(dataset=val_set, num_workers=workers, batch_size=1, pin_memory=True)

    net = model.ClassificationNet(refine_features)

    if load:
        print('loading weights')
        net = torch.load(load, map_location='cpu')
        net.refine_features(refine_features)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lrate, weight_decay=weight_decay)

    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss

    trainer = create_supervised_trainer(net, optimizer, criterion, device=device, non_blocking=True)
    evaluator = create_supervised_evaluator(model, device=device, non_blocking=True, metrics={'accuracy': Accuracy(output_transform=output_preprocess),
                                                                                              'precision': Precision(output_transform=output_preprocess),
                                                                                              'recall': Recall(output_transform=output_preprocess),
                                                                                              'loss': Loss(criterion)})

    ckpt_handler = ModelCheckpoint('.', name, save_interval=1, n_saved=10, require_empty=False)
    est_handler = EarlyStopping(lag, score_function, trainer)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    progress_bar = ProgressBar(persist=True)
    progress_bar.attach(trainer, ['loss'])

    evaluator.add_event_handler(Events.COMPLETED, est_handler)
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=ckpt_handler, to_save={'net': model})
    trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED, handler=TerminateOnNan())

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_data_loader)
        metrics = evaluator.state.metrics
        progress_bar.log_message('eval results - epoch {} loss: {:.2f} accuracy: {:.2f} recall: {:.2f} precision {:.2f}'.format(engine.state.epoch,
                                                                                                                   metrics['loss'],
                                                                                                                   metrics['accuracy'],
                                                                                                                   metrics['recall'],
                                                                                                                   metrics['precision']))
    trainer.run(train_data_loader, max_epochs=1000)


@cli.command()
@click.option('-m', '--model', default=None, help='model file')
@click.option('-d', '--device', default='cpu', help='pytorch device')
@click.argument('images', nargs=-1)
def pred(model, device, images):

    device = torch.device(device)
    with open(model, 'rb') as fp:
        net = torch.load(fp, map_location=device)

    transform = transforms.Compose([transforms.Resize(1200), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    with torch.no_grad():
        for img in images:
            print('transforming image {}'.format(img))
            im = Image.open(img).convert('RGB')
            norm_im = transform(im)
            print('running forward pass')
            o = m.forward(norm_im.unsqueeze(0))
            o = torch.sigmoid(o)
            print('pred: {}'.format(o))
