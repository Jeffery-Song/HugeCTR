"""
  Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
  
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  
      http://www.apache.org/licenses/LICENSE-2.0
  
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

import os
import datetime
from enum import Enum
import copy
import json

def percent_gen(lb, ub, gap=1):
  ret = []
  i = lb
  while i <= ub:
    ret.append(i/100)
    i += gap
  return ret

def reverse_percent_gen(lb, ub, gap=1):
  ret = percent_gen(lb, ub, gap)
  return list(reversed(ret))

datetime_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
LOG_DIR='run-logs/logs_samgraph_' + datetime_str
CONFIG_DIR='run-configs/config_hps_' + datetime_str

class System(Enum):
  hps = 0
  collcache = 1

  def __str__(self):
    if self is System.hps:
      return "HPS"
    else:
      return "COLL"

class Model(Enum):
  dlrm = 0

class Dataset(Enum):
  def __new__(cls, *args, **kwds):
    value = len(cls.__members__) + 1
    obj = object.__new__(cls)
    obj._value_ = value
    return obj
  def __init__(self, path, short_name, vocabulary, slot_num):
    self.path = path if path else self.name
    self.short_name = short_name if short_name else self.name
    self.vocabulary = vocabulary
    self.slot_num = slot_num
  def __str__(self):
    return self.path
  def short(self):
    return self.short_name
  criteo_like_uniform       = None,                      "CRU",        187767399, 26
  criteo_like_uniform_small = None,                      "CRU_S",      187767399, 26
  dlrm_datasets             = None,                      "DLRM",       None,      None
  simple_power02            = "simple_power0.2",         "SP_02",      100000000, 25
  simple_power02_slot100    = "simple_power0.2_slot100", "SP_02_S100", 100000000, 100
  simple_power1             = None,                      "SP_1",       100000000, 25
  simple_power1_slot100     = None,                      "SP_1_S100",  100000000, 100
  simple_uniform            = None,                      None,         100000000, 25
  criteo_tb                 = None,                      None,         882774559, 26
class CachePolicy(Enum):
  cache_by_degree=0
  cache_by_heuristic=1
  cache_by_pre_sample=2
  cache_by_degree_hop=3
  cache_by_presample_static=4
  cache_by_fake_optimal=5
  dynamic_cache=6
  cache_by_random=7
  coll_cache=8
  coll_cache_intuitive=9
  partition_cache=10
  part_rep_cache=11
  rep_cache=12
  coll_cache_asymm_link=13
  clique_part=14
  clique_part_by_degree=15

  def __str__(self):
    name_list = [
      'degree',
      'heuristic',
      'pre_sample',
      'degree_hop',
      'presample_static',
      'fake_optimal',
      'dynamic_cache',
      'random',
      'coll_cache',
      'coll_intuitive',
      'partition',
      'part_rep',
      'rep',
      'coll_asymm',
      'cliq_part',
      'cliq_part_degree'
    ]
    return name_list[self.value]
  
  def short(self):
    policy_str_short = [
      "Deg",
      "Heuristic",
      "PreS",
      "DegH",
      "PreSS",
      "FakeOpt",
      "DynCache",
      "Rand",
      "Coll",
      "CollIntui",
      "Part",
      "PartRep",
      "Rep",
      "CollAsymm",
      "CliqPart",
      "CliqPartDeg",
    ]
    return policy_str_short[self.value]

class RunConfig:
  def __init__(self, system:System, model:Model, dataset:Dataset, 
               gpu_num: int=8,
               global_batch_size: int=65536,
               coll_cache_policy:CachePolicy=CachePolicy.coll_cache_asymm_link, 
               cache_percent:float=0.1, 
               logdir:str=LOG_DIR,
               confdir:str=CONFIG_DIR):
    # arguments
    self.system         = system
    self.model          = model
    self.dataset        = dataset
    self.logdir         = logdir
    self.confdir        = confdir
    self.gpu_num        = gpu_num
    self.epoch          = 5
    # self.iter_num       = 6000
    self.slot_num       = None
    self.dense_dim      = 13
    self.embed_vec_size = 128
    self.combiner       = "mean"
    self.optimizer      = "plugin_adam"
    self.global_batch_size      = global_batch_size
    self.dataset_root_path      = "/nvme/songxiaoniu/hps-dataset/"
    self.model_root_path        = "/nvme/songxiaoniu/hps-model/dlrm_criteo/"
    # hps json
    self.cache_percent          = cache_percent
    self.coll_cache_policy      = coll_cache_policy
    self.mock_embedding         = False    # if true, mock embedding table by emb_vec_sz and max_voc_sz
    self.plain_dense_model      = False
    self.max_vocabulary_size    = None
    self.coll_cache_enable_iter = 1000
    self.iteration_per_epoch    = 1000
    # env variables
    self.coll_cache_no_group    = False
    self.coll_cache_concurrent_link   = False
    self.log_level              = "warn"
    self.profile_level          = 3

  def get_mock_sparse_name(self):
    if self.mock_embedding:
      return '_'.join(['mock', f'{self.max_vocabulary_size}', f'{self.embed_vec_size}'])
    else:
      return 'nomock'

  def get_output_fname_base(self):
    std_out_fname = '_'.join(
      [str(self.system), self.model.name, self.dataset.short()] + 
      [f'policy_{self.coll_cache_policy.short()}', f'cache_rate_{self.cache_percent}'] +
      [f'batch_size_{self.global_batch_size}'])
    if self.mock_embedding:
      std_out_fname += '_' + self.get_mock_sparse_name()
    return std_out_fname

  def get_conf_fname(self):
    std_out_conf = f'{self.confdir}/'
    std_out_conf += self.get_output_fname_base()
    std_out_conf += '.json'
    return std_out_conf

  def get_log_fname(self):
    std_out_log = f'{self.logdir}/'
    std_out_log += self.get_output_fname_base()
    return std_out_log

  def beauty(self):
    msg = ' '.join(
      ['Running', str(self.system), self.model.name, str(self.dataset)] +
      [str(self.coll_cache_policy), f'cache_rate {self.cache_percent}'])
    if self.mock_embedding:
      msg += f' mock({self.max_vocabulary_size} vocabs, {self.embed_vec_size} emb_vec_sz)'
    return datetime.datetime.now().strftime('[%H:%M:%S]') + msg + '.'

  def form_cmd(self, durable_log=True):
    assert((self.epoch * self.iteration_per_epoch + self.coll_cache_enable_iter) == self.iter_num)
    cmd_line = f'COLL_NUM_REPLICA={self.gpu_num} '
    if self.coll_cache_no_group != False:
      cmd_line += f'SAMGRAPH_COLL_CACHE_NO_GROUP=1 '
    if self.coll_cache_concurrent_link != False:
      cmd_line += f'SAMGRAPH_COLL_CACHE_CONCURRENT_LINK=1 '
    cmd_line += f'SAMGRAPH_LOG_LEVEL={self.log_level} '
    cmd_line += f'SAMGRAPH_PROFILE_LEVEL={self.profile_level} '

    cmd_line += f'python ../examples/inference.py'
    cmd_line += f' --gpu_num {self.gpu_num} '
    
    cmd_line += f' --iter_num {self.iter_num} '
    cmd_line += f' --slot_num {self.slot_num} '
    cmd_line += f' --dense_dim {self.dense_dim} '
    cmd_line += f' --embed_vec_size {self.embed_vec_size} '
    cmd_line += f' --global_batch_size {self.global_batch_size} '
    cmd_line += f' --combiner {self.combiner} '
    cmd_line += f' --optimizer {self.optimizer} '
    if self.plain_dense_model:
      cmd_line += f' --dense_model_path plain'
    else:
      cmd_line += f' --dense_model_path {self.model_root_path}dense.model'
    cmd_line += f' --dataset_path {self.dataset_root_path + str(self.dataset)}'
    cmd_line += f' --ps_config_file {self.get_conf_fname()}'

    if durable_log:
      std_out_log = self.get_log_fname() + '.log'
      std_err_log = self.get_log_fname() + '.err.log'
      cmd_line += f' > \"{std_out_log}\"'
      cmd_line += f' 2> \"{std_err_log}\"'
      cmd_line += ';'
    return cmd_line

  def generate_ps_config(self):
    self.iter_num = self.epoch * self.iteration_per_epoch + self.coll_cache_enable_iter
    self.max_vocabulary_size = self.dataset.vocabulary
    self.slot_num = self.dataset.slot_num
    assert((self.global_batch_size % self.gpu_num) == 0)
    conf = {
      "supportlonglong": True,
      "models": [{
          "num_of_worker_buffer_in_pool": 1,
          "num_of_refresher_buffer_in_pool": 0,
          "embedding_table_names":["sparse_embedding0"],
          "default_value_for_each_table": [1.0],
          "i64_input_key": False,
          "cache_refresh_percentage_per_iteration": 0,
          "hit_rate_threshold": 1.1,
          "gpucache": True,
          }
      ],
      "volatile_db": {
          "type": "direct_map",
          "num_partitions": 56
      },
      "use_multi_worker": True,
    }
    conf['models'][0]['model'] = self.model.name
    conf['models'][0]['sparse_files'] = [self.model_root_path + 'sparse_cont.model']
    if self.mock_embedding: conf['models'][0]['sparse_files'] = [self.get_mock_sparse_name()]
    conf['models'][0]['embedding_vecsize_per_table'] = [self.embed_vec_size]
    conf['models'][0]['maxnum_catfeature_query_per_table_per_sample'] = [self.slot_num]
    conf['models'][0]['deployed_device_list'] = list(range(self.gpu_num))
    conf['models'][0]['max_batch_size'] = self.global_batch_size // self.gpu_num
    conf['models'][0]['gpucacheper'] = self.cache_percent

    conf['models'][0]['max_vocabulary_size'] = [self.max_vocabulary_size]
    if self.system == System.hps: conf['use_coll_cache'] = False
    else: conf['use_coll_cache'] = True
    conf['coll_cache_enable_iter'] = self.coll_cache_enable_iter
    conf['iteration_per_epoch'] = self.iteration_per_epoch
    conf['epoch'] = self.epoch
    conf['coll_cache_policy'] = self.coll_cache_policy.value

    result = json.dumps(conf, indent=4)
    with open(self.get_conf_fname(), "w") as outfile:
      outfile.write(result)

  def run(self, mock=False, durable_log=True, callback = None):
    os.system('mkdir -p {}'.format(self.confdir))
    self.generate_ps_config()

    if mock:
      print(self.form_cmd(durable_log))
    else:
      print(self.beauty())

      if durable_log:
        os.system('mkdir -p {}'.format(self.logdir))
      status = os.system(self.form_cmd(durable_log))
      if os.WEXITSTATUS(status) != 0:
        print("FAILED!")
        return 1

      if callback != None:
        callback(self)
    return 0

def run_in_list(conf_list : list, mock=False, durable_log=True, callback = None):
  for conf in conf_list:
    conf : RunConfig
    conf.run(mock, durable_log, callback)

class ConfigList:
  def __init__(self):
    self.conf_list = [
      RunConfig(System.hps, Model.dlrm, Dataset.criteo_like_uniform)]

  def select(self, key, val_indicator):
    '''
    filter config list by key and list of value
    available key: model, dataset, cache_policy, pipeline
    '''
    newlist = []
    for cfg in self.conf_list:
      if getattr(cfg, key) in val_indicator:
        newlist.append(cfg)
    self.conf_list = newlist
    return self

  def override(self, key, val_list):
    '''
    override config list by key and value.
    if len(val_list)>1, then config list is extended, example:
       [cfg1(batch_size=4000)].override('batch_size',[1000,8000]) 
    => [cfg1(batch_size=1000),cfg1(batch_size=8000)]
    available key: arch, logdir, cache_percent, cache_policy, batch_size
    '''
    if len(val_list) == 0:
      return self
    orig_list = self.conf_list
    self.conf_list = []
    for val in val_list:
      new_list = copy.deepcopy(orig_list)
      for cfg in new_list:
        setattr(cfg, key, val)
      self.conf_list += new_list
    return self

  def override_T(self, key, val_list):
    if len(val_list) == 0:
      return self
    orig_list = self.conf_list
    self.conf_list = []
    for cfg in orig_list:
      for val in val_list:
        cfg = copy.deepcopy(cfg)
        setattr(cfg, key, val)
        self.conf_list.append(cfg)
    return self

  def part_override(self, filter_key, filter_val_list, override_key, override_val_list):
    newlist = []
    for cfg in self.conf_list:
      # print(cfg.cache_impl, cfg.logdir, filter_key, filter_val_list)
      if getattr(cfg, filter_key) in filter_val_list:
        # print(cfg.cache_impl, cfg.logdir)
        for val in override_val_list:
          # print(cfg.cache_impl, cfg.logdir)
          cfg = copy.deepcopy(cfg)
          setattr(cfg, override_key, val)
          newlist.append(cfg)
      else:
        newlist.append(cfg)
    self.conf_list = newlist
    return self

  def hyper_override(self, key_array, val_matrix):
    if len(key_array) == 0 or len(val_matrix) == 0:
      return self
    orig_list = self.conf_list
    self.conf_list = []
    for cfg in orig_list:
      for val_list in val_matrix:
        cfg = copy.deepcopy(cfg)
        for idx in range(len(key_array)):
          setattr(cfg, key_array[idx], val_list[idx])
        self.conf_list.append(cfg)
    return self

  def concat(self, another_list):
    self.conf_list += copy.deepcopy(another_list.conf_list)
    return self
  def copy(self):
    return copy.deepcopy(self)
  @staticmethod
  def Empty():
    ret = ConfigList()
    ret.conf_list = []
    return ret
  @staticmethod
  def MakeList(conf):
    ret = ConfigList()
    if isinstance(conf, list):
      ret.conf_list = conf
    elif isinstance(conf, RunConfig):
      ret.conf_list = [conf]
    else:
      raise Exception("Please construct fron runconfig or list of it")
    return ret

  def run(self, mock=False, durable_log=True, callback = None):
    for conf in self.conf_list:
      conf : RunConfig
      conf.run(mock, durable_log, callback)
