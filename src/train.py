import pandas as pd
import numpy as np
import torch
import torchvision
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator
import time

gmf_config = {'alias': 'gmf_factor8neg4-implict',
              'num_epoch': 500,
              'batch_size': 1024,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': True,
              'device_id': 0,
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

mlp_config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
              'num_epoch': 500,
              'batch_size': 256,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 0,
              'pretrain': False,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': 'pretrain_neumf_factor8neg4',
                'num_epoch': 75,
                'batch_size': 1024, #8192,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 6040,
                'num_items': 3706,
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.0000001,
                'use_cuda': True,
                'device_id': 0,
                'pretrain': False,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model'),
                'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }

ml1m_dir = 'data/ml-1m/ratings.dat'
test_dir = 'data/ml-1m/ratingstest.dat'
# Load Data
def load_ratings(dirName):
    ml1m_rating = pd.read_csv(dirName, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    # Reindex
    user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    user_id.to_csv('userIds')
    ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
    item_id = ml1m_rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    item_id.to_csv('itemIds')
    ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
    ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
    #ml1m_rating = ml1m_rating.to_sparse()
    #print(type(ml1m_rating))
    print(ml1m_rating)
    print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))
    return ml1m_rating

ml1m_rating = load_ratings(ml1m_dir)

# DataLoader for training
sample_generator = SampleGenerator(ratings=ml1m_rating)
eval_data = sample_generator.evaluate_data
print(len(eval_data))





# Specify the exact model
#config = gmf_config
#engine = GMFEngine(config)
#config = mlp_config
#engine = MLPEngine(config)
config = neumf_config
engine = NeuMFEngine(config)
dummy_input_users = eval_data[0].cuda()
dummy_input_items = eval_data[1].cuda()
dummy_input = (dummy_input_users, dummy_input_items)



input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(5) ]
output_names = [ "output1" ]
for epoch in range(config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    start = time.time()

    hit_ratio, ndcg = engine.evaluate(eval_data, epoch_id=epoch)
    end = time.time()
    print(end - start)
    print("time")




    engine.save(config['alias'], epoch, hit_ratio, ndcg)
    torch.onnx.export(engine.model, dummy_input, "NCF.onnx", verbose=True, input_names=input_names, output_names=output_names)

