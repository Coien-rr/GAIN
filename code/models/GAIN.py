import dgl # Deep Graph Library 用于处理图形数据的库
import dgl.nn.pytorch as dglnn # DGL的 PyTorch 模块，用于构建和训练图神经网络模型
import numpy as np # numpy 科学计算库
import torch # PyTorch深度学习框架
import torch.nn as nn # PyTorch中的神经网络模块，提供预定义的神经网络层和模型
from transformers import * # Transformers提供预训练的语言模型和各种工具和函数

from utils import get_cuda # 导入了自定义模块utils中的函数get_cuda(), 用于进行GPU加速的辅助函数


class GAIN_GloVe(nn.Module):
    # 定义GAIN_Glove的PyTorch基类，该类继承nn.Module
    # nn.Module是PyTorch中所有神经网络模型的基类，定义了一些常用的方法和属性，例如forward()和parameters()方法。

    def __init__(self, config):
        # 定义GAIN_Glove的初始化函数

        super(GAIN_GloVe, self).__init__()
        self.config = config

        # 配置词向量维度、词汇表大小和编码器输入大小
        word_emb_size = config.word_emb_size # 词向量：用来表示词语语义的向量，通常是固定维度，即词向量维度
        vocabulary_size = config.vocabulary_size
        encoder_input_size = word_emb_size

        # 根据config的激活函数选择相应的激活函数
        ## nn.Tanh(): 双曲正切函数
        ## nn.ReLU(): 修正线性单元
        self.activation = nn.Tanh() if config.activation == 'tanh' else nn.ReLU()


        # 定义词嵌入层
        self.word_emb = nn.Embedding(vocabulary_size, word_emb_size, padding_idx=config.word_pad)
        # 定义了一个名为self.word_emb的成员变量，类型为nn.Embedding，用于将输入的单词序列转化为对应的词向量序列。
        # word_emb在模型的前向传播过程中会被调用，将输入的单词序列转化为对应的词向量序列，并输入到后续的网络结构中进行处理和预测。

        # 如果使用预训练词向量，则用预训练的词向量来初始化词嵌入层的权重
        ## 预训练词向量：预训练词向量是指在大规模的语料库上进行训练得到的词向量。
        ## 它通常使用词嵌入技术（如Word2Vec、GloVe等）来学习词汇在高维空间的分布式表示，使得语义上相似的词在向量空间中的距离也比较近。
        ## 预训练词向量通常能够提高模型的性能，特别是在训练数据有限的情况下。
        if config.pre_train_word:
            self.word_emb = nn.Embedding(config.data_word_vec.shape[0], word_emb_size, padding_idx=config.word_pad)
            self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec[:, :word_emb_size]))

        # 根据config确定是否微调词嵌入层self.word_emb的权重
        self.word_emb.weight.requires_grad = config.finetune_word

        # 如果使用实体类型信息，则将实体类型嵌入到编码器输入中
        if config.use_entity_type:
            # 将词嵌入层输出维度 word_emb_size 与实体类型嵌入层输出维度 config.entity_type_size 相加
            # 作为双向LSTM的输入维度encoder_input_size
            # encoder_input_size = config.word_emb_size + config.entity_type_size
            encoder_input_size += config.entity_type_size # Line24 encoder_input_size = word_emb_size
            # 增加实体类型嵌入层，
            self.entity_type_emb = nn.Embedding(config.entity_type_num, config.entity_type_size,
                                                padding_idx=config.entity_type_pad)

        # 如果使用实体ID信息，则将实体ID嵌入到编码器输入中
        if config.use_entity_id:
            # 如上
            encoder_input_size += config.entity_id_size
            #增加实体ID嵌入层
            self.entity_id_emb = nn.Embedding(config.max_entity_num + 1, config.entity_id_size,
                                              padding_idx=config.entity_id_pad)

        # 定义双向LSTM编码器
        self.encoder = BiLSTM(encoder_input_size, config)
        ## BiLSTM是一个双向的长短时记忆网络
        ## 由前向LSTM和后向LSTM组成，可以有效地捕捉序列中的时序信息，并生成一个固定维度的向量表示，用于后续的图卷积和关系预测。

        # 定义RelGraphConvLayer实例，构成多层的GCN层
        self.gcn_dim = config.gcn_dim # gcn_dim是GCN层的输入和输出维度
        assert self.gcn_dim == 2 * config.lstm_hidden_size, 'gcn dim should be the lstm hidden dim * 2'
        # lstm_hidden_size 是BiLSTM模型中的隐藏层维度。
        # 因为在图卷积网络层中，节点的特征是由它本身的特征以及邻居节点的特征组成，因此节点特征的维度应该是至少邻居节点特征的维度之和，

        rel_name_lists = ['intra', 'inter', 'global'] # rel_name_lists参数指定了每个图卷积层中考虑的不同关系类型的列表。

        # RelGraphConvLayer() 是一个基于关系图的图卷积层。
        # 在关系图卷积中，每个节点都表示一个实体，节点之间的边表示实体之间的关系，通过卷积运算在节点之间传递信息。
        self.GCN_layers = nn.ModuleList([RelGraphConvLayer(self.gcn_dim, self.gcn_dim, rel_name_lists,
                                                           num_bases=len(rel_name_lists), activation=self.activation,
                                                           self_loop=True, dropout=self.config.dropout)
                                         for i in range(config.gcn_layers)])

        # 计算Bank大小并定义dropout
        self.bank_size = self.config.gcn_dim * (self.config.gcn_layers + 1)

        # GCN encoder中使用的dropout层，它可以在训练过程中随机将一些神经元的输出值设为0，有助于防止过拟合。
        # self.config.dropout表示dropout的概率，即每个神经元被随机丢弃的概率。
        self.dropout = nn.Dropout(self.config.dropout)

        # 定义预测层
        self.predict = nn.Sequential(
            nn.Linear(self.bank_size * 5 + self.gcn_dim * 4, self.bank_size * 2),  #
            self.activation,
            self.dropout,
            nn.Linear(self.bank_size * 2, config.relation_nums),
        )

        # 定义边缘层
        self.edge_layer = RelEdgeLayer(node_feat=self.gcn_dim, edge_feat=self.gcn_dim,
                                       activation=self.activation, dropout=config.dropout)

        self.path_info_mapping = nn.Linear(self.gcn_dim * 4, self.gcn_dim * 4)
        self.attention = Attention(self.bank_size * 2, self.gcn_dim * 4)

    def forward(self, **params):
        """
            words: [batch_size, max_length]
            src_lengths: [batchs_size]
            mask: [batch_size, max_length]
            entity_type: [batch_size, max_length]
            entity_id: [batch_size, max_length]
            mention_id: [batch_size, max_length]
            distance: [batch_size, max_length]
            entity2mention_table: list of [local_entity_num, local_mention_num]
            graphs: list of DGLHeteroGraph
            h_t_pairs: [batch_size, h_t_limit, 2]
        """

        # words: 输入文本的词序列张量
        # src_lengths: words中每个样本的实际词数
        # mask: 对words进行padding的mask张量
        # entity_type: 输入文本中每个词的实体类型的标记张量
        # entity_id: 每个词所属实体的唯一标识符张量
        # mention_id: 每个词在文本中出现的唯一标识符的张量
        # distance: 每个词与所属实体之间的距离的张量
        # entity2mention_table: 实体与其所包含的提及之间的映射表
        # graphs: 输入文本中所有实体和提及之间的图
        # h_t_pairs:

        src = self.word_emb(params['words'])
        # 通过调用词嵌入层word_emb,将输入文本的词序列对应的词向量序列

        mask = params['mask']
        # 获得输入的遮盖掩码，以掩盖填充的位置。

        bsz, slen, _ = src.size()
        # bsz: 批量大小
        # slen: 序列长度
        # _: 词向量维度（不使用该变量

        # 如果使用实体类型
        if self.config.use_entity_type:
            src = torch.cat([src, self.entity_type_emb(params['entity_type'])], dim=-1)
            # 将实体类型的嵌入向量与 src 沿着最后一个维度（dim=-1）拼接在一起，最终得到一个新的张量作为模型的输入。

        # 如果使用实体ID
        if self.config.use_entity_id:
            src = torch.cat([src, self.entity_id_emb(params['entity_id'])], dim=-1)
            # 将实体ID的嵌入向量与 src 沿着最后一个维度（dim=-1）拼接在一起，最终得到一个新的张量作为模型的输入。

        # src: [batch_size, slen, encoder_input_size]
        # src_lengths: [batchs_size]

        encoder_outputs, (output_h_t, _) = self.encoder(src, params['src_lengths'])
        # 使用Encoder将输入的 src 序列编码成一个context向量序列 encoder_outputs，并返回最后一个时刻的hidden状态向量 output_h_t。
        encoder_outputs[mask == 0] = 0
        # encoder_outputs: [batch_size, slen, 2*encoder_hid_size]
        # output_h_t: [batch_size, 2*encoder_hid_size]

        graphs = params['graphs'] # 从参数中获取图结构

        mention_id = params['mention_id'] # 从参数中获取提及实体ID
        features = None

        # 遍历graphs
        for i in range(len(graphs)):
            encoder_output = encoder_outputs[i]  # 获取编码器输出，shape: [slen, 2*encoder_hid_size]
            mention_num = torch.max(mention_id[i]) # 获取当前图中提及实体的数量
            mention_index = get_cuda(
                (torch.arange(mention_num) + 1).unsqueeze(1).expand(-1, slen))  # [mention_num, slen]
            mentions = mention_id[i].unsqueeze(0).expand(mention_num, -1)  # [mention_num, slen]
            select_metrix = (mention_index == mentions).float()  # [mention_num, slen]
            # average word -> mention 将词向量转换为提及实体向量，即将每个提及实体的所有词的词向量进行平均
            word_total_numbers = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, slen)  # [mention_num, slen]
            select_metrix = torch.where(word_total_numbers > 0, select_metrix / word_total_numbers, select_metrix)
            x = torch.mm(select_metrix, encoder_output)  # [mention_num, 2*encoder_hid_size]

            x = torch.cat((output_h_t[i].unsqueeze(0), x), dim=0)

            if features is None:
                features = x
            else:
                features = torch.cat((features, x), dim=0)

        graph_big = dgl.batch_hetero(graphs)
        output_features = [features]

        for GCN_layer in self.GCN_layers:
            features = GCN_layer(graph_big, {"node": features})["node"]  # [total_mention_nums, gcn_dim]
            output_features.append(features)

        output_feature = torch.cat(output_features, dim=-1)

        graphs = dgl.unbatch_hetero(graph_big)

        # mention -> entity
        entity2mention_table = params['entity2mention_table']  # list of [entity_num, mention_num]
        entity_num = torch.max(params['entity_id'])
        entity_bank = get_cuda(torch.Tensor(bsz, entity_num, self.bank_size))
        global_info = get_cuda(torch.Tensor(bsz, self.bank_size))

        cur_idx = 0
        entity_graph_feature = None
        for i in range(len(graphs)):
            # average mention -> entity
            select_metrix = entity2mention_table[i].float()  # [local_entity_num, mention_num]
            select_metrix[0][0] = 1
            mention_nums = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, select_metrix.size(1))
            select_metrix = torch.where(mention_nums > 0, select_metrix / mention_nums, select_metrix)
            node_num = graphs[i].number_of_nodes('node')
            entity_representation = torch.mm(select_metrix, output_feature[cur_idx:cur_idx + node_num])
            entity_bank[i, :select_metrix.size(0) - 1] = entity_representation[1:]
            global_info[i] = output_feature[cur_idx]
            cur_idx += node_num

            if entity_graph_feature is None:
                entity_graph_feature = entity_representation[1:, -self.config.gcn_dim:]
            else:
                entity_graph_feature = torch.cat(
                    (entity_graph_feature, entity_representation[1:, -self.config.gcn_dim:]), dim=0)

        h_t_pairs = params['h_t_pairs']
        h_t_pairs = h_t_pairs + (h_t_pairs == 0).long() - 1  # [batch_size, h_t_limit, 2]
        h_t_limit = h_t_pairs.size(1)

        # [batch_size, h_t_limit, bank_size]
        h_entity_index = h_t_pairs[:, :, 0].unsqueeze(-1).expand(-1, -1, self.bank_size)
        t_entity_index = h_t_pairs[:, :, 1].unsqueeze(-1).expand(-1, -1, self.bank_size)

        # [batch_size, h_t_limit, bank_size]
        h_entity = torch.gather(input=entity_bank, dim=1, index=h_entity_index)
        t_entity = torch.gather(input=entity_bank, dim=1, index=t_entity_index)

        global_info = global_info.unsqueeze(1).expand(-1, h_t_limit, -1)

        entity_graphs = params['entity_graphs']
        entity_graph_big = dgl.batch(entity_graphs)
        self.edge_layer(entity_graph_big, entity_graph_feature)
        entity_graphs = dgl.unbatch(entity_graph_big)
        path_info = get_cuda(torch.zeros((bsz, h_t_limit, self.gcn_dim * 4)))
        relation_mask = params['relation_mask']
        path_table = params['path_table']
        for i in range(len(entity_graphs)):
            path_t = path_table[i]
            for j in range(h_t_limit):
                if relation_mask is not None and relation_mask[i, j].item() == 0:
                    break

                h = h_t_pairs[i, j, 0].item()
                t = h_t_pairs[i, j, 1].item()
                # for evaluate
                if relation_mask is None and h == 0 and t == 0:
                    continue

                if (h + 1, t + 1) in path_t:
                    v = [val - 1 for val in path_t[(h + 1, t + 1)]]
                elif (t + 1, h + 1) in path_t:
                    v = [val - 1 for val in path_t[(t + 1, h + 1)]]
                else:
                    print(h, t, v)
                    print(entity_graphs[i].all_edges())
                    print(h_t_pairs)
                    print(relation_mask)
                    assert 1 == 2

                middle_node_num = len(v)

                if middle_node_num == 0:
                    continue

                # forward
                edge_ids = get_cuda(entity_graphs[i].edge_ids([h for _ in range(middle_node_num)], v))
                forward_first = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)
                edge_ids = get_cuda(entity_graphs[i].edge_ids(v, [t for _ in range(middle_node_num)]))
                forward_second = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)

                # backward
                edge_ids = get_cuda(entity_graphs[i].edge_ids([t for _ in range(middle_node_num)], v))
                backward_first = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)
                edge_ids = get_cuda(entity_graphs[i].edge_ids(v, [h for _ in range(middle_node_num)]))
                backward_second = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)

                tmp_path_info = torch.cat((forward_first, forward_second, backward_first, backward_second), dim=-1)
                _, attn_value = self.attention(torch.cat((h_entity[i, j], t_entity[i, j]), dim=-1), tmp_path_info)
                path_info[i, j] = attn_value

            entity_graphs[i].edata.pop('h')

        path_info = self.dropout(
            self.activation(
                self.path_info_mapping(path_info)
            )
        )

        predictions = self.predict(torch.cat(
            (h_entity, t_entity, torch.abs(h_entity - t_entity), torch.mul(h_entity, t_entity), global_info, path_info),
            dim=-1))
        return predictions


class GAIN_BERT(nn.Module):
    def __init__(self, config):
        super(GAIN_BERT, self).__init__()
        self.config = config
        if config.activation == 'tanh':
            self.activation = nn.Tanh()
        elif config.activation == 'relu':
            self.activation = nn.ReLU()
        else:
            assert 1 == 2, "you should provide activation function."

        if config.use_entity_type:
            self.entity_type_emb = nn.Embedding(config.entity_type_num, config.entity_type_size,
                                                padding_idx=config.entity_type_pad)
        if config.use_entity_id:
            self.entity_id_emb = nn.Embedding(config.max_entity_num + 1, config.entity_id_size,
                                              padding_idx=config.entity_id_pad)

        self.bert = BertModel.from_pretrained(config.bert_path)
        if config.bert_fix:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.gcn_dim = config.gcn_dim
        assert self.gcn_dim == config.bert_hid_size + config.entity_id_size + config.entity_type_size

        rel_name_lists = ['intra', 'inter', 'global']
        self.GCN_layers = nn.ModuleList([RelGraphConvLayer(self.gcn_dim, self.gcn_dim, rel_name_lists,
                                                           num_bases=len(rel_name_lists), activation=self.activation,
                                                           self_loop=True, dropout=self.config.dropout)
                                         for i in range(config.gcn_layers)])

        self.bank_size = self.gcn_dim * (self.config.gcn_layers + 1)
        self.dropout = nn.Dropout(self.config.dropout)
        self.predict = nn.Sequential(
            nn.Linear(self.bank_size * 5 + self.gcn_dim * 4, self.bank_size * 2),
            self.activation,
            self.dropout,
            nn.Linear(self.bank_size * 2, config.relation_nums),
        )

        self.edge_layer = RelEdgeLayer(node_feat=self.gcn_dim, edge_feat=self.gcn_dim,
                                       activation=self.activation, dropout=config.dropout)

        self.path_info_mapping = nn.Linear(self.gcn_dim * 4, self.gcn_dim * 4)

        self.attention = Attention(self.bank_size * 2, self.gcn_dim * 4)

    def forward(self, **params):
        '''
        words: [batch_size, max_length]
        src_lengths: [batchs_size]
        mask: [batch_size, max_length]
        entity_type: [batch_size, max_length]
        entity_id: [batch_size, max_length]
        mention_id: [batch_size, max_length]
        distance: [batch_size, max_length]
        entity2mention_table: list of [local_entity_num, local_mention_num]
        graphs: list of DGLHeteroGraph
        h_t_pairs: [batch_size, h_t_limit, 2]
        ht_pair_distance: [batch_size, h_t_limit]
        '''
        words = params['words']
        mask = params['mask']
        bsz, slen = words.size()

        encoder_outputs, sentence_cls = self.bert(input_ids=words, attention_mask=mask)
        # encoder_outputs[mask == 0] = 0

        if self.config.use_entity_type:
            encoder_outputs = torch.cat([encoder_outputs, self.entity_type_emb(params['entity_type'])], dim=-1)

        if self.config.use_entity_id:
            encoder_outputs = torch.cat([encoder_outputs, self.entity_id_emb(params['entity_id'])], dim=-1)

        sentence_cls = torch.cat(
            (sentence_cls, get_cuda(torch.zeros((bsz, self.config.entity_type_size + self.config.entity_id_size)))),
            dim=-1)
        # encoder_outputs: [batch_size, slen, bert_hid+type_size+id_size]
        # sentence_cls: [batch_size, bert_hid+type_size+id_size]

        graphs = params['graphs']

        mention_id = params['mention_id']
        features = None

        for i in range(len(graphs)):
            encoder_output = encoder_outputs[i]  # [slen, bert_hid]
            mention_num = torch.max(mention_id[i])
            mention_index = get_cuda(
                (torch.arange(mention_num) + 1).unsqueeze(1).expand(-1, slen))  # [mention_num, slen]
            mentions = mention_id[i].unsqueeze(0).expand(mention_num, -1)  # [mention_num, slen]
            select_metrix = (mention_index == mentions).float()  # [mention_num, slen]
            # average word -> mention
            word_total_numbers = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, slen)  # [mention_num, slen]
            select_metrix = torch.where(word_total_numbers > 0, select_metrix / word_total_numbers, select_metrix)

            x = torch.mm(select_metrix, encoder_output)  # [mention_num, bert_hid]
            x = torch.cat((sentence_cls[i].unsqueeze(0), x), dim=0)

            if features is None:
                features = x
            else:
                features = torch.cat((features, x), dim=0)

        graph_big = dgl.batch_hetero(graphs)
        output_features = [features]

        for GCN_layer in self.GCN_layers:
            features = GCN_layer(graph_big, {"node": features})["node"]  # [total_mention_nums, gcn_dim]
            output_features.append(features)

        output_feature = torch.cat(output_features, dim=-1)

        graphs = dgl.unbatch_hetero(graph_big)

        # mention -> entity
        entity2mention_table = params['entity2mention_table']  # list of [entity_num, mention_num]
        entity_num = torch.max(params['entity_id'])
        entity_bank = get_cuda(torch.Tensor(bsz, entity_num, self.bank_size))
        global_info = get_cuda(torch.Tensor(bsz, self.bank_size))

        cur_idx = 0
        entity_graph_feature = None
        for i in range(len(graphs)):
            # average mention -> entity
            select_metrix = entity2mention_table[i].float()  # [local_entity_num, mention_num]
            select_metrix[0][0] = 1
            mention_nums = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, select_metrix.size(1))
            select_metrix = torch.where(mention_nums > 0, select_metrix / mention_nums, select_metrix)
            node_num = graphs[i].number_of_nodes('node')
            entity_representation = torch.mm(select_metrix, output_feature[cur_idx:cur_idx + node_num])
            entity_bank[i, :select_metrix.size(0) - 1] = entity_representation[1:]
            global_info[i] = output_feature[cur_idx]
            cur_idx += node_num

            if entity_graph_feature is None:
                entity_graph_feature = entity_representation[1:, -self.gcn_dim:]
            else:
                entity_graph_feature = torch.cat((entity_graph_feature, entity_representation[1:, -self.gcn_dim:]),
                                                 dim=0)

        h_t_pairs = params['h_t_pairs']
        h_t_pairs = h_t_pairs + (h_t_pairs == 0).long() - 1  # [batch_size, h_t_limit, 2]
        h_t_limit = h_t_pairs.size(1)

        # [batch_size, h_t_limit, bank_size]
        h_entity_index = h_t_pairs[:, :, 0].unsqueeze(-1).expand(-1, -1, self.bank_size)
        t_entity_index = h_t_pairs[:, :, 1].unsqueeze(-1).expand(-1, -1, self.bank_size)

        # [batch_size, h_t_limit, bank_size]
        h_entity = torch.gather(input=entity_bank, dim=1, index=h_entity_index)
        t_entity = torch.gather(input=entity_bank, dim=1, index=t_entity_index)

        global_info = global_info.unsqueeze(1).expand(-1, h_t_limit, -1)

        entity_graphs = params['entity_graphs']
        entity_graph_big = dgl.batch(entity_graphs)
        self.edge_layer(entity_graph_big, entity_graph_feature)

        entity_graphs = dgl.unbatch(entity_graph_big)
        path_info = get_cuda(torch.zeros((bsz, h_t_limit, self.gcn_dim * 4)))
        relation_mask = params['relation_mask']
        path_table = params['path_table']
        for i in range(len(entity_graphs)):
            path_t = path_table[i]
            for j in range(h_t_limit):
                if relation_mask is not None and relation_mask[i, j].item() == 0:
                    break

                h = h_t_pairs[i, j, 0].item()
                t = h_t_pairs[i, j, 1].item()
                # for evaluate
                if relation_mask is None and h == 0 and t == 0:
                    continue

                if (h + 1, t + 1) in path_t:
                    v = [val - 1 for val in path_t[(h + 1, t + 1)]]
                elif (t + 1, h + 1) in path_t:
                    v = [val - 1 for val in path_t[(t + 1, h + 1)]]
                else:
                    print(h, t, v)
                    print(entity_graphs[i].number_of_nodes())
                    print(entity_graphs[i].all_edges())
                    print(path_table)
                    print(h_t_pairs)
                    print(relation_mask)
                    assert 1 == 2

                middle_node_num = len(v)

                if middle_node_num == 0:
                    continue

                # forward
                edge_ids = get_cuda(entity_graphs[i].edge_ids([h for _ in range(middle_node_num)], v))
                forward_first = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)
                edge_ids = get_cuda(entity_graphs[i].edge_ids(v, [t for _ in range(middle_node_num)]))
                forward_second = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)

                # backward
                edge_ids = get_cuda(entity_graphs[i].edge_ids([t for _ in range(middle_node_num)], v))
                backward_first = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)
                edge_ids = get_cuda(entity_graphs[i].edge_ids(v, [h for _ in range(middle_node_num)]))
                backward_second = torch.index_select(entity_graphs[i].edata['h'], dim=0, index=edge_ids)

                tmp_path_info = torch.cat((forward_first, forward_second, backward_first, backward_second), dim=-1)
                _, attn_value = self.attention(torch.cat((h_entity[i, j], t_entity[i, j]), dim=-1), tmp_path_info)
                path_info[i, j] = attn_value

            entity_graphs[i].edata.pop('h')

        path_info = self.dropout(
            self.activation(
                self.path_info_mapping(path_info)
            )
        )

        predictions = self.predict(torch.cat(
            (h_entity, t_entity, torch.abs(h_entity - t_entity), torch.mul(h_entity, t_entity), global_info, path_info),
            dim=-1))
        # predictions = self.predict(torch.cat((h_entity, t_entity, torch.abs(h_entity-t_entity), torch.mul(h_entity, t_entity), global_info), dim=-1))
        return predictions


class Attention(nn.Module):
    def __init__(self, src_size, trg_size):
        super().__init__()
        self.W = nn.Bilinear(src_size, trg_size, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, trg, attention_mask=None):
        '''
        src: [src_size]
        trg: [middle_node, trg_size]
        '''

        score = self.W(src.unsqueeze(0).expand(trg.size(0), -1), trg)
        score = self.softmax(score)
        value = torch.mm(score.permute(1, 0), trg)

        return score.squeeze(0), value.squeeze(0)


class BiLSTM(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=config.lstm_hidden_size,
                            num_layers=config.nlayers, batch_first=True,
                            bidirectional=True)
        self.in_dropout = nn.Dropout(config.dropout)
        self.out_dropout = nn.Dropout(config.dropout)

    def forward(self, src, src_lengths):
        '''
        src: [batch_size, slen, input_size]
        src_lengths: [batch_size]
        '''

        self.lstm.flatten_parameters()
        bsz, slen, input_size = src.size()

        src = self.in_dropout(src)

        new_src_lengths, sort_index = torch.sort(src_lengths, dim=-1, descending=True)
        new_src = torch.index_select(src, dim=0, index=sort_index)

        packed_src = nn.utils.rnn.pack_padded_sequence(new_src, new_src_lengths, batch_first=True, enforce_sorted=True)
        packed_outputs, (src_h_t, src_c_t) = self.lstm(packed_src)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True,
                                                      padding_value=self.config.word_pad)

        unsort_index = torch.argsort(sort_index)
        outputs = torch.index_select(outputs, dim=0, index=unsort_index)

        src_h_t = src_h_t.view(self.config.nlayers, 2, bsz, self.config.lstm_hidden_size)
        src_c_t = src_c_t.view(self.config.nlayers, 2, bsz, self.config.lstm_hidden_size)
        output_h_t = torch.cat((src_h_t[-1, 0], src_h_t[-1, 1]), dim=-1)
        output_c_t = torch.cat((src_c_t[-1, 0], src_c_t[-1, 1]), dim=-1)
        output_h_t = torch.index_select(output_h_t, dim=0, index=unsort_index)
        output_c_t = torch.index_select(output_c_t, dim=0, index=unsort_index)

        outputs = self.out_dropout(outputs)
        output_h_t = self.out_dropout(output_h_t)
        output_c_t = self.out_dropout(output_c_t)

        return outputs, (output_h_t, output_c_t)


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class RelEdgeLayer(nn.Module):
    def __init__(self,
                 node_feat,
                 edge_feat,
                 activation,
                 dropout=0.0):
        super(RelEdgeLayer, self).__init__()
        self.node_feat = node_feat
        self.edge_feat = edge_feat
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.mapping = nn.Linear(node_feat * 2, edge_feat)

    def forward(self, g, inputs):
        # g = g.local_var()

        g.ndata['h'] = inputs  # [total_mention_num, node_feat]
        g.apply_edges(lambda edges: {
            'h': self.dropout(self.activation(self.mapping(torch.cat((edges.src['h'], edges.dst['h']), dim=-1))))})
        g.ndata.pop('h')


class Bert():
    MASK = '[MASK]'
    CLS = "[CLS]"
    SEP = "[SEP]"

    def __init__(self, model_class, model_name, model_path=None):
        super().__init__()
        self.model_name = model_name
        print(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.max_len = 512

    def tokenize(self, text, masked_idxs=None):
        tokenized_text = self.tokenizer.tokenize(text)
        if masked_idxs is not None:
            for idx in masked_idxs:
                tokenized_text[idx] = self.MASK
        # prepend [CLS] and append [SEP]
        # see https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py#L195  # NOQA
        tokenized = [self.CLS] + tokenized_text + [self.SEP]
        return tokenized

    def tokenize_to_ids(self, text, masked_idxs=None, pad=True):
        tokens = self.tokenize(text, masked_idxs)
        return tokens, self.convert_tokens_to_ids(tokens, pad=pad)

    def convert_tokens_to_ids(self, tokens, pad=True):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = torch.tensor([token_ids])
        # assert ids.size(1) < self.max_len
        ids = ids[:, :self.max_len]  # https://github.com/DreamInvoker/GAIN/issues/4
        if pad:
            padded_ids = torch.zeros(1, self.max_len).to(ids)
            padded_ids[0, :ids.size(1)] = ids
            mask = torch.zeros(1, self.max_len).to(ids)
            mask[0, :ids.size(1)] = 1
            return padded_ids, mask
        else:
            return ids

    def flatten(self, list_of_lists):
        for list in list_of_lists:
            for item in list:
                yield item

    def subword_tokenize(self, tokens):
        """Segment each token into subwords while keeping track of
        token boundaries.
        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subwords, flanked by the special symbols required
                by Bert (CLS and SEP).
            - An array of indices into the list of subwords, indicating
                that the corresponding subword is the start of a new
                token. For example, [1, 3, 4, 7] means that the subwords
                1, 3, 4, 7 are token starts, while all other subwords
                (0, 2, 5, 6, 8...) are in or at the end of tokens.
                This list allows selecting Bert hidden states that
                represent tokens, which is necessary in sequence
                labeling.
        """
        subwords = list(map(self.tokenizer.tokenize, tokens))
        subword_lengths = list(map(len, subwords))
        subwords = [self.CLS] + list(self.flatten(subwords))[:509] + [self.SEP]
        token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
        token_start_idxs[token_start_idxs > 509] = 512
        return subwords, token_start_idxs

    def subword_tokenize_to_ids(self, tokens):
        """Segment each token into subwords while keeping track of
        token boundaries and convert subwords into IDs.
        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subword IDs, including IDs of the special
                symbols (CLS and SEP) required by Bert.
            - A mask indicating padding tokens.
            - An array of indices into the list of subwords. See
                doc of subword_tokenize.
        """
        subwords, token_start_idxs = self.subword_tokenize(tokens)
        subword_ids, mask = self.convert_tokens_to_ids(subwords)
        return subword_ids.numpy(), token_start_idxs, subwords

    def segment_ids(self, segment1_len, segment2_len):
        ids = [0] * segment1_len + [1] * segment2_len
        return torch.tensor([ids])
