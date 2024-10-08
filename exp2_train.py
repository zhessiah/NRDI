from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler

from utils import *
from modules import *

parser = argparse.ArgumentParser()
parser.add_argument('--sample-size', type=int, default=10000, help=' ')
parser.add_argument('--graph', type=str, default='rg', help='ba,ws,rg,grid')
parser.add_argument('--temp', type=float, default=0.5, help='Temperature for Gumbel softmax.')
parser.add_argument('--dims', type=int, default=8, help='The number of input dimensions (position + velocity).')
parser.add_argument('--num-atoms', type=int, default=10, help='Number of atoms in simulation.')
parser.add_argument('--hard', action='store_true', default=True,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--gpu', type=str, default='0', help='Disables CUDA training.')
parser.add_argument('--decoder', type=str, default='mlp', help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--prediction-steps', type=int, default=49, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--timesteps', type=int, default=49, help='The number of time steps per sample.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
# parser.add_argument('--epochs', type=int, default=500,help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128, help='Number of samples per batch.')
# parser.add_argument('--batch-size', type=int, default=128,help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--no-factor', action='store_true', default=False, help='Disables factor graph model.')
parser.add_argument('--encoder-dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--save-folder', type=str, default='newlogs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' + 'Leave empty to train from scratch')
parser.add_argument('--edge-types', type=int, default=2, help='The number of edge types to infer.')
parser.add_argument('--lr-decay', type=int, default=200, help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=False,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5, help='Output variance.')
parser.add_argument('--prior', action='store_true', default=False, help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=False,
                    help='Whether test with dynamically re-computed graph.')
start_time = datetime.datetime.now()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:" + args.gpu)
args.factor = not args.no_factor
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dynamic_graph:
    print("Testing with dynamically re-computed graph.")

# Save model and meta-data. Always saves in a new sub-folder.


# train_loader, valid_loader,  loc_max, loc_min, vel_max, vel_min = load_data(args.sample_size,args.batch_size, args.suffix)
# train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(args.batch_size, args.suffix)
train_loader = load_exp2_data(args.timesteps, args.num_atoms, args.graph, args.batch_size, args.sample_size, args.dims)
# valid_loader = load_exp2_data(args.timesteps, args.num_atoms, args.graph, args.batch_size, args.sample_size // 3,args.dims)

# Generate off-diagonal interaction graph
off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)
rel_rec = np.array(encode_onehot(np.where(off_diag)[0]),
                   dtype=np.float32)  # [20,5] np.where(off_diag)[0]:[0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4]
rel_send = np.array(encode_onehot(np.where(off_diag)[1]),
                    dtype=np.float32)  # [20,5] np.where(off_diag)[1]:[1 2 3 4 0 2 3 4 0 1 3 4 0 1 2 4 0 1 2 3]
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

encoder = MLPEncoder(args.timesteps * args.dims, args.encoder_hidden,
                     args.edge_types,
                     args.encoder_dropout, args.factor)

if args.decoder == 'mlp':
    decoder = MLPDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         msg_hid=args.decoder_hidden,
                         msg_out=args.decoder_hidden,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
elif args.decoder == 'ode':
    decoder = ODEDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         msg_hid=args.decoder_hidden,
                         msg_out=args.decoder_hidden,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
elif args.decoder == 'ode2':
    decoder = ODEDecoder2(n_in_node=args.dims,
                          edge_types=args.edge_types,
                          msg_hid=args.decoder_hidden,
                          msg_out=args.decoder_hidden,
                          n_hid=args.decoder_hidden,
                          do_prob=args.decoder_dropout,
                          skip_first=args.skip_first)

# if args.load_folder:
#     encoder_file = os.path.join(args.load_folder, 'encoder.pt')
#     encoder.load_state_dict(torch.load(encoder_file))
#     decoder_file = os.path.join(args.load_folder, 'decoder.pt')
#     decoder.load_state_dict(torch.load(decoder_file))
#     args.save_folder = False

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

# Linear indices of an upper triangular mx, used for acc calculation
# triu_indices = get_triu_offdiag_indices(args.num_atoms)
# tril_indices = get_tril_offdiag_indices(args.num_atoms)

# if args.prior:
#     prior = np.array([0.91, 0.03, 0.03, 0.03])  # TODO: hard coded for now
#     print("Using prior")
#     print(prior)
#     log_prior = torch.FloatTensor(np.log(prior))
#     log_prior = torch.unsqueeze(log_prior, 0)
#     log_prior = torch.unsqueeze(log_prior, 0)
#     log_prior = Variable(log_prior)
#
#     if args.cuda:
#         log_prior = log_prior.cuda()

if args.cuda:
    # encoder.cuda()
    # decoder.cuda()
    # rel_rec = rel_rec.cuda()
    # rel_send = rel_send.cuda()
    # triu_indices = triu_indices.cuda()
    # tril_indices = tril_indices.cuda()
    encoder.to(device)
    decoder.to(device)
    rel_rec = rel_rec.to(device)
    rel_send = rel_send.to(device)

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)

if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')
    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')
    pickle.dump({'args': args}, open(meta_file, "wb"))
    print(args, file=log)
    log.flush()
else:
    print("WARNING: No save_folder provided!" + "Testing (within this script) will throw an error.")


def train(epoch, best_val_loss):
    nll_train = []
    acc_train = []
    kl_train = []
    mse_train = []

    encoder.train()
    decoder.train()
    scheduler.step()

    for batch_idx, (data, relations) in enumerate(train_loader):
        if args.cuda:
            data, relations = data.to(device), relations.to(device)
        data, relations = Variable(data), Variable(relations)

        optimizer.zero_grad()

        # data  torch.Size([128, 5, 49, 4])
        # logit torch.Size([128, 20, 2])
        # edges torch.Size([128, 20, 2])

        logits = encoder(data, rel_rec, rel_send)
        edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)

        # output shape torch.Size([128, 5, 48, 4])

        if args.decoder == 'ode2':
            output = decoder(data, edges, rel_rec, rel_send, torch.linspace(0., 1., args.timesteps),args.timesteps, 1)
        else:
            output = decoder(data, edges, rel_rec, rel_send, args.timesteps)

        target = data[:, :, 1:, :]
        loss_nll = nll_gaussian(output, target, args.var)

        # if args.prior:
        #     loss_kl = kl_categorical(prob, log_prior, args.num_atoms)
        # else:
        #     loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)
        # loss_kl = F.cross_entropy(logits.view(-1, 2), relations.long().view(-1))
        prob = my_softmax(logits, -1)
        loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)
        loss = loss_nll + loss_kl

        acc = edge_accuracy(logits, relations)

        loss.backward()
        optimizer.step()

        acc_train.append(acc)
        mse_train.append(F.mse_loss(output, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())

    # nll_val = []
    # acc_val = []
    # kl_val = []
    # mse_val = []
    # encoder.eval()
    # decoder.eval()
    # for batch_idx, (data, relations) in enumerate(valid_loader):
    #     if args.cuda:
    #         data, relations = data.to(device), relations.to(device)
    #     with torch.no_grad():
    #         data, relations = Variable(data), Variable(relations)
    #
    #     logits = encoder(data, rel_rec, rel_send)
    #     edges = gumbel_softmax(logits, tau=args.temp, hard=True)
    #     prob = my_softmax(logits, -1)
    #
    #     if args.decoder == 'ode2':
    #         output = decoder(data, edges, rel_rec, rel_send, torch.linspace(0., 1., args.timesteps),
    #                          1,1)
    #     else:
    #         output = decoder(data, edges, rel_rec, rel_send, 1)
    #
    #     target = data[:, :, 1:, :]
    #     loss_nll = nll_gaussian(output, target, args.var)
    #     loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)
    #
    #     acc = edge_accuracy(logits, relations)
    #     acc_val.append(acc)
    #
    #     mse_val.append(F.mse_loss(output, target).item())
    #     nll_val.append(loss_nll.item())
    #     kl_val.append(loss_kl.item())

    if args.save_folder :
        if np.mean(mse_train) < best_val_loss:
            torch.save(encoder.state_dict(), encoder_file)
            torch.save(decoder.state_dict(), decoder_file)
            print('Best model so far, saving...', file=log)
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'kl_train: {:.10f}'.format(np.mean(kl_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              # 'nll_val: {:.10f}'.format(np.mean(nll_val)),
              # 'kl_val: {:.10f}'.format(np.mean(kl_val)),
              # 'mse_val: {:.10f}'.format(np.mean(mse_val)),
              # 'acc_val: {:.10f}'.format(np.mean(acc_val)),
              file=log)
        log.flush()
    return np.mean(mse_train)


# Train model
best_val_loss = np.inf
best_epoch = 0
for epoch in range(args.epochs):
    val_loss = train(epoch, best_val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_folder:
    print("Best Epoch is {:04d}".format(best_epoch), file=log)
    log.flush()

# test()  -------------------------
if log is not None:
    print(save_folder)
    log.close()

end_time = datetime.datetime.now()
print("运行时间：", end_time - start_time)
