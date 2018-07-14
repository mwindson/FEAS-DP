import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
import argparse
import os
from models.lstm import LSTM
from models.at_lstm import AT_LSTM
from models.atae_lstm import ATAE_LSTM
from utils import TextDataSet, WordEmbedding


# from models.ian import IAN
# from models.memnet import MemNet
# from models.ram import RAM
# from models.td_lstm import TD_LSTM
# from models.cabasc import Cabasc

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        print('> training arguments:')
        for arg in vars(opt):
            print('>>> {0}: {1}'.format(arg, getattr(opt, arg)))

        # absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim, max_seq_len=opt.max_seq_len)
        embed = WordEmbedding(os.path.dirname(__file__) + '/data/word2vec/sgns.financial.word')
        train_set = TextDataSet(os.path.dirname(__file__) + '/data/multi.csv', embed,
                                max_seq_len=opt.max_seq_len)
        test_set = TextDataSet(os.path.dirname(__file__) + '/data/multi.csv', embed, max_seq_len=opt.max_seq_len,
                               train=False, test=True)

        self.train_data_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(dataset=test_set, batch_size=opt.batch_size,
                                           shuffle=False)
        # self.writer = SummaryWriter(log_dir=opt.logdir)

        self.model = opt.model_class(embed.m, opt).to(opt.device)
        self.reset_parameters()

    def reset_parameters(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
                if len(p.shape) > 1:
                    self.opt.initializer(p)
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(params, lr=self.opt.learning_rate)
        print('-----------start train---------- ')
        max_test_acc = 0
        global_step = 0
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch models to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['label'].to(opt.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    # switch models to evaluation mode
                    self.model.eval()
                    n_test_correct, n_test_total = 0, 0
                    with torch.no_grad():
                        for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                            t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                            t_targets = t_sample_batched['label'].to(opt.device)
                            t_outputs = self.model(t_inputs)

                            n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                            n_test_total += len(t_outputs)
                        test_acc = n_test_correct / n_test_total
                        if test_acc > max_test_acc:
                            max_test_acc = test_acc

                        print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}'.format(loss.item(), train_acc, test_acc))

                        # log
                        # self.writer.add_scalar('loss', loss, global_step)
                        # self.writer.add_scalar('acc', train_acc, global_step)
                        # self.writer.add_scalar('test_acc', test_acc, global_step)

        # self.writer.close()

        print('max_test_acc: {0}'.format(max_test_acc))
        return max_test_acc


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='atae_lstm', type=str)
    parser.add_argument('--dataset', default='twitter', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--logdir', default='log', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str)
    opt = parser.parse_args()

    model_classes = {
        'lstm': LSTM,
        'at_lstm': AT_LSTM,
        'atae_lstm': ATAE_LSTM,
        # 'td_lstm': TD_LSTM,
        # 'ian': IAN,
        # 'memnet': MemNet,
        # 'ram': RAM,
        # 'cabasc': Cabasc
    }
    input_colses = {
        'lstm': ['text_raw_indices'],
        'at_lstm': ['text_raw_indices', 'entity_indices'],
        'atae_lstm': ['text_raw_indices', 'entity_indices'],
        # 'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        # 'ian': ['text_raw_indices', 'aspect_indices'],
        # 'memnet': ['text_raw_without_aspect_indices', 'aspect_indices', 'text_left_with_aspect_indices'],
        # 'ram': ['text_raw_indices', 'aspect_indices'],
        # 'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices',
        #            'text_right_with_aspect_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    ins = Instructor(opt)
    ins.run()
