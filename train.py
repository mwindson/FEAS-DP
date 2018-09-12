import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse
import os
import pandas as pd
from utils import TextDataSet, WordEmbedding

from models import LSTM, AT_LSTM, ATAE_LSTM, RAM, IAN, CNN, Cabasc, MemNet, LRG


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        print('> training arguments:')
        for arg in vars(opt):
            print('>>> {0}: {1}'.format(arg, getattr(opt, arg)))
        w2v_path = {
            'word': '/data/word2vec/sgns.financial.word',
            'word_bigram': '/data/word2vec/sgns.financial.bigram',
            'char': '/data/word2vec/sgns.financial.char',
            'char_bigram': '/data/word2vec/sgns.financial.bigram-char'
        }
        embed = WordEmbedding(os.path.dirname(__file__) + w2v_path[opt.vector_level], initializer='avg')
        train_set = TextDataSet(os.path.dirname(__file__) + '/data/single_test.csv', embed,
                                max_seq_len=opt.max_seq_len, vector_level=opt.vector_level, train=True)
        test_set = TextDataSet(os.path.dirname(__file__) + '/data/single_test.csv', embed,
                               max_seq_len=opt.max_seq_len, vector_level=opt.vector_level, test=True)
        self.train_data_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(dataset=test_set, batch_size=opt.batch_size,
                                           shuffle=True)
        self.writer = SummaryWriter(log_dir=opt.logdir)

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

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.opt.learning_rate * (0.25 ** (int(epoch / 10)))
        print("lr>>>>>> ", lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(params, lr=self.opt.learning_rate, weight_decay=1e-4)
        print('-----------start train---------- ')
        max_test_acc = 0
        global_step = 0
        try:
            for epoch in range(self.opt.num_epoch):
                print('>' * 100)
                print('epoch: ', epoch)
                self.adjust_learning_rate(optimizer, epoch)
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
                            errors = pd.DataFrame([], columns=['text', 'entity', 'predict', 'label'])
                            test_loss = 0.
                            n_test_loss = 0.
                            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                                t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                                t_targets = t_sample_batched['label'].to(opt.device)
                                t_outputs = self.model(t_inputs)
                                if max_test_acc > 0:
                                    t_res = torch.argmax(t_outputs, -1)
                                    t_diff = t_res == t_targets
                                    for i, d in enumerate(t_diff.cpu().numpy()):
                                        if d == 0:
                                            err = {'text': t_sample_batched['text_raw'][i],
                                                   'entity': t_sample_batched['entity'][i],
                                                   'predict': t_res.cpu().numpy()[i],
                                                   'label': t_targets.cpu().numpy()[i]}
                                            errors = errors.append(err, ignore_index=True)
                                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                                n_test_total += len(t_outputs)
                                n_test_loss += len(t_inputs) * criterion(t_outputs, t_targets)
                            test_acc = n_test_correct / n_test_total
                            test_loss = n_test_loss
                            if test_acc > max_test_acc:
                                max_test_acc = test_acc
                                torch.save({
                                    'epoch': epoch,
                                    'state_dict': self.model.state_dict(),
                                    'best_test_acc': max_test_acc,
                                }, './best/' + opt.model_name + '_best.pkl')
                                if max_test_acc > 0.80:
                                    errors.to_csv(opt.model_name + '_error.csv', encoding='utf8')
                            print(
                                'loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_loss: {:.4f}'.format(loss.item(),
                                                                                                        train_acc,
                                                                                                        test_acc,
                                                                                                        test_loss))

                            # log
                            self.writer.add_scalars('loss', {'train_loss': loss, 'test_loss': test_loss}, global_step)
                            self.writer.add_scalars('acc', {'train_acc': train_acc, 'test_acc': test_acc}, global_step)
        except KeyboardInterrupt:
            print('training has stopped early')
        self.writer.close()

        print('max_test_acc: {0}'.format(max_test_acc))
        return max_test_acc


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='ram', type=str)
    parser.add_argument('--vector_level', default='word', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--num_epoch', default=30, type=int)
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
        'cnn': CNN,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'lrg': LRG
    }
    input_colses = {
        'lstm': ['text_raw_indices'],
        'at_lstm': ['text_raw_indices', 'entity_indices'],
        'atae_lstm': ['text_raw_indices', 'entity_indices'],
        'cnn': ['text_raw_indices'],
        'ian': ['text_raw_indices', 'entity_indices'],
        'memnet': ['text_raw_without_entity_indices', 'entity_indices', 'text_left_with_entity_indices'],
        'ram': ['text_raw_indices', 'entity_indices'],
        'cabasc': ['text_raw_indices', 'entity_indices', 'text_left_with_entity_indices',
                   'text_right_with_entity_indices'],
        'lrg': ['text_raw_indices', 'entity_indices', 'text_left_with_entity_indices',
                'text_right_with_entity_indices'],
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
