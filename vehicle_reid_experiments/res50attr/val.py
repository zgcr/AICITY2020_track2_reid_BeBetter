import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import torch
import pickle
from sklearn.metrics import precision_recall_curve
import random
from itertools import groupby
import pickle


def get_l2_square_dist(qry: np.array, glr: np.array):
    qry_square = (qry**2).sum(axis=1).reshape(-1, 1)
    glr_square = (glr**2).sum(axis=1).reshape(1, -1)
    cross = np.matmul(qry, np.transpose(glr))
    distance = qry_square + glr_square - 2 * cross
    return distance


class Rerank(object):
    def __init__(self, features, distances, bmk):
        # benchmark信息统计
        self.bmk = bmk
        self.q_info, self.g_info = self.bmk['reid_query'], self.bmk[
            'reid_gallery']

        self.q_features = features[:len(self.q_info)]
        self.g_features = features[len(self.q_info):]
        self.qg_distances = distances

        assert len(self.q_info) == len(self.q_features) and len(
            self.g_info) == len(self.g_features)

        # 获取query、gallery的group信息
        self.get_group_info()

        # 获取query-gallery的precision-recall信息
        self.qg_pr = self.get_query_gallery_pr()

    def _get_gallery_track_info(self):
        # 获取gallery的track信息
        track2indexs = {}
        for index, item in enumerate(self.g_info):
            track_id = item['track_id']
            if track_id not in track2indexs:
                track2indexs[track_id] = []
            track2indexs[track_id].append(index)
        return track2indexs

    def _get_distance(self, vec1, vec2):
        # 计算两个向量的距离
        return np.sum(np.square(vec1 - vec2))

    def _dist2prob(self, dists):
        # 距离正则化
        dists = np.array(dists)
        max_dist = dists.max()
        min_dist = dists.min()
        return 1 - (dists - min_dist) / (max_dist -
                                         min_dist), max_dist, min_dist

    def _get_pr(self, all_distances, all_labels):
        # 计算不同precision下对应的recall，用于卡阈值
        all_probs, max_prob, min_prob = self._dist2prob(all_distances)
        all_labels = np.array(all_labels)

        indexes = (-all_probs).argsort()
        precision, recall, threshold = precision_recall_curve(
            all_labels, all_probs)
        recall_at_precision = {}
        for i in [
                0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                0.99, 0.998, 0.9998, 0.99998
        ]:
            index = np.where(precision >= i)[0][0]
            if index >= len(threshold) - 1:
                recall_at_precision[i] = (-1.0, -1.0)
            else:
                recall_at_precision[i] = (recall[index],
                                          (1 - threshold[index]) *
                                          (max_prob - min_prob) + min_prob,
                                          index)
        return recall_at_precision

    def get_query_dismat(self):
        # 获取query之间的距离信息，用于query成团
        distances, labels = [], []
        # 下三角矩阵
        for i, (f_i, info_i) in enumerate(zip(self.q_features, self.q_info)):
            for j, (f_j, info_j) in enumerate(zip(self.q_features,
                                                  self.q_info)):
                if i <= j:
                    continue
                dis = self._get_distance(f_i, f_j)
                label = 1 if info_i['vehicle_id'] == info_j['vehicle_id'] else 0

                distances.append(dis)
                labels.append(label)
        return distances, labels

    def get_gallery_dismat(self):
        # 获取gallery track之间的距离信息，用于track成团
        # 成团后，fp较多，所以实际没有使用

        # combine track feature by mean
        combine_gallery_info = []
        for f, info in zip(self.g_features, self.g_info):
            info['feature'] = f
            combine_gallery_info.append(info)
        combine_gallery_info.sort(key=lambda x: x['track_id'])

        track_groups = []
        for k, g in groupby(combine_gallery_info, key=lambda x: x['track_id']):
            features, vid = [], ''
            group = list(g)

            random.shuffle(group)
            for info in group:
                features.append(info['feature'])
                assert vid == info['vehicle_id'] or vid == ''
                vid = info['vehicle_id']

            features = np.asarray(features)
            item = {
                'track_id': k,
                'vehicle_id': vid,
                'feature': np.mean(features, axis=0)
            }
            # assert len(item['feature']) == 2048
            track_groups.append(item)

        # get distance
        distances, labels, track = [], [], []
        for i, info_i in enumerate(track_groups):
            for j, info_j in enumerate(track_groups):
                # 下三角矩阵
                if i <= j:
                    continue
                dis = self._get_distance(info_i['feature'], info_j['feature'])
                label = 1 if info_i['vehicle_id'] == info_j['vehicle_id'] else 0
                track.append([info_i['track_id'], info_j['track_id']])

                distances.append(dis)
                labels.append(label)
        return distances, labels, track_groups

    def get_query_pr(self):
        # 获取query-query的precision、recall
        all_distances, all_labels = self.get_query_dismat()
        return self._get_pr(all_distances, all_labels)

    def get_gallery_pr(self):
        # 获取gallery-gallery的precision、recall
        all_distances, all_labels, _ = self.get_gallery_dismat()
        return self._get_pr(all_distances, all_labels)

    def get_query_gallery_pr(self):
        # 获取query-gallery的precision、recall
        all_distances = self.qg_distances.reshape(-1)
        all_labels = []
        for q in self.bmk['reid_query']:
            for g in self.bmk['reid_gallery']:
                if q['vehicle_id'] == g['vehicle_id']:
                    all_labels.append(1)
                else:
                    all_labels.append(0)
        all_labels = np.asarray(all_labels)
        assert all_labels.shape == all_distances.shape

        return self._get_pr(all_distances, all_labels)

    def get_query_group(self, qq_thre):
        # query成group
        q_dists = get_l2_square_dist(self.q_features, self.q_features)
        mask = q_dists <= qq_thre

        result = []
        for i, line in enumerate(mask):
            r_item = [i]
            for j, x in enumerate(line):
                if i != j and x == True:
                    r_item.append(j)
            r_item.sort()
            if len(r_item) != 1 and r_item not in result:
                result.append(r_item)
        killed = []
        for i in range(len(result)):
            for j in range(len(result)):
                if set(result[i]) > set(result[j]):
                    killed.append(j)
        final_result, labels = [], []
        for i in range(len(result)):
            if i not in killed:
                final_result.append(result[i])
                vids = []
                for x in result[i]:
                    vids.append(self.q_info[x]['vehicle_id'])
                label = 1 if len(set(vids)) == 1 else 0
                labels.append(label)
        print('query group total {} right {}.'.format(len(final_result),
                                                      sum(labels)))
        return final_result

    def get_gallery_group(self, gg_thre):
        # gallery成group
        _, _, tracks_info = self.get_gallery_dismat()
        track_features = list(track['feature'] for track in tracks_info)
        track_features = np.asarray(track_features)
        g_dists = get_l2_square_dist(track_features, track_features)
        mask = g_dists <= gg_thre

        result = []
        for i, line in enumerate(mask):
            r_item = [i]
            for j, x in enumerate(line):
                if i != j and x == True:
                    r_item.append(j)
            r_item.sort()
            if len(r_item) != 1 and r_item not in result:
                result.append(r_item)
        killed = []
        for i in range(len(result)):
            for j in range(len(result)):
                if set(result[i]) > set(result[j]):
                    killed.append(j)
        final_result, labels = [], []
        for i in range(len(result)):
            if i not in killed:
                track = []
                for x in result[i]:
                    track.append(tracks_info[x]['track_id'])

                final_result.append(track)
                vids = []
                for x in result[i]:
                    vids.append(tracks_info[x]['vehicle_id'])
                label = 1 if len(set(vids)) == 1 else 0
                labels.append(label)
        print('query group total {} right {}.'.format(len(final_result),
                                                      sum(labels)))
        return final_result

    def get_group_info(self, qq_precision=0.99, gg_precision=0.75):
        # 获取query和gallery的group信息
        qq_thre = self.get_query_pr()[qq_precision][1]
        gg_thre = self.get_gallery_pr()[gg_precision][1]

        self.qq_group = self.get_query_group(qq_thre)
        self.gg_group = self.get_gallery_group(gg_thre)
        return


# 一些写在外面的辅助函数，没有整理


def _contain(i, group):
    for x in group:
        if i in x:
            return x
    return None


def get_sorted_index(distance, track2indexs, g_info, thre, max_contine=500):
    # 重排序，距离排序结果结合track信息，将同一track的所有图片前移
    redata = []

    # 因为有可能image重复，即gallery的某张图与group内的多个query距离都很小，
    # 所以选择 max_contine 会大一些
    distance_sort = distance[:max_contine]

    # 找到满足precision阈值要求的所有图片索引
    max_j = -1
    thre_index = []
    for j, info in enumerate(distance_sort):
        if info[0] <= thre:
            thre_index.append(info[1])
            if j > max_j:
                max_j = j

    # 如果没有找到符合阈值的distance，说明gallery的所有图与query的距离都很大，
    # 这种情况下，把某条track的图片整体前移是危险的，所以保持原顺序返回
    if max_j == -1:
        return [x[1] for x in distance_sort[:100]]
    else:
        # 顺序插入同track的数据
        item = []
        for idx in thre_index:
            item.extend(track2indexs[g_info[idx]['track_id']])
        item.extend(list(x[1] for x in distance[max_j + 1:1000]))

        # 去重
        redata = [distance_sort[0][1]]
        for j, x in enumerate(item):
            if x not in item[:j]:
                redata.append(x)
        assert len(redata) >= 100
    return redata[:100]


def get_rerank_result(rerank,
                      bmk,
                      qg_precision=0.1,
                      color_bias=400,
                      type_bias=400):
    """
    qg_precision : 卡一个precision的阈值；
    color_bias : color不同时，增加的距离惩罚项；
    type_bias : type不同时，增加的距离惩罚项；
    """

    # 获取gallery的track信息
    track2indexs = rerank._get_gallery_track_info()
    qg_thre = rerank.qg_pr[qg_precision][1]
    print('####### qg_thre:', qg_thre)
    # query的group信息
    query_group = rerank.qq_group
    # 初始化最终返回结果
    rerank_result = np.zeros(shape=(rerank.qg_distances.shape[0], 100)) - 1

    pbar = tqdm(total=len(rerank.qg_distances), desc='sample')

    for i, _ in enumerate(rerank.qg_distances):
        # 如果已经包含在之前的group里，直接跳过
        if -1 not in rerank_result[i]:
            continue

        # 如果这条query可以形成group
        if _contain(i, query_group):
            group_indexs = _contain(i, query_group)
            # 把各个group的distance整合在一起
            combine_distances = []
            for gidx in group_indexs:
                item_list = []
                for u, dis in enumerate(rerank.qg_distances[gidx]):
                    # 如果属性不同，增加距离惩罚项
                    if bmk['reid_query'][gidx]['pred_color'] != bmk[
                            'reid_gallery'][u]['pred_color']:
                        dis += color_bias
                    if bmk['reid_query'][gidx]['pred_type'] != bmk[
                            'reid_gallery'][u]['pred_type']:
                        dis += type_bias
                    item_list.append((dis, u, gidx))

                combine_distances.extend(item_list)
            # 按照距离排序
            combine_distances.sort(key=lambda x: x[0])
            # 返回group所有距离的重排序结果，并写入每个group元素的对应行
            sort_index = get_sorted_index(combine_distances, track2indexs,
                                          rerank.g_info, qg_thre)
            for gidx in group_indexs:
                rerank_result[gidx, :] = sort_index[:]

        else:
            # 没有形成group的query，整理成相同的格式
            combine_distances = []
            item_list = []
            for u, dis in enumerate(rerank.qg_distances[i]):
                if bmk['reid_query'][i]['pred_color'] != bmk['reid_gallery'][
                        u]['pred_color']:
                    dis += color_bias
                if bmk['reid_query'][i]['pred_type'] != bmk['reid_gallery'][u][
                        'pred_type']:
                    dis += type_bias
                item_list.append((dis, u, i))
            combine_distances.extend(item_list)

            combine_distances.sort(key=lambda x: x[0])
            # 返回结果，写入对应行
            sort_index = get_sorted_index(combine_distances, track2indexs,
                                          rerank.g_info, qg_thre)
            rerank_result[i, :] = sort_index[:]
        pbar.update(1)
    pbar.close()
    assert -1 not in rerank_result
    return rerank_result


def standardization(feature):
    mu = np.mean(feature, axis=1, keepdims=True)
    sigma = np.std(feature, axis=1, keepdims=True)

    return (feature - mu) / (sigma + 1e-8)


def get_dist_feature(val_loader, model, epoch, norm_feature=True):
    # switch to evaluate mode
    model.eval()

    total_features = []
    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.cuda()
            batch_feat_bn, _, _, _ = model(inputs)
            total_features.append(batch_feat_bn)

    total_features = torch.cat(total_features, axis=0)
    total_features = total_features.cpu().numpy()

    if norm_feature:
        total_features = standardization(total_features)

    query_num = 1052
    query_features = total_features[:query_num]
    gallery_features = total_features[query_num:]
    gallery_num = gallery_features.shape[0]

    bias_a = (query_features**2).sum(axis=1)
    bias_a = bias_a[:, np.newaxis]
    bias_a = np.repeat(bias_a, gallery_num, axis=1)
    bias_b = (gallery_features**2).sum(axis=1)
    bias_b = bias_b[np.newaxis, :]
    bias_b = np.repeat(bias_b, query_num, axis=0)
    cross = np.matmul(query_features, gallery_features.transpose((1, 0)))

    dists = bias_a + bias_b - 2 * cross

    return dists, query_features, gallery_features


def generate_txt_from_dists(dists, gallery_info, topn=100, be_sorted=False):
    if not be_sorted:
        sorted_index = np.argsort(dists, axis=1)
    else:
        sorted_index = dists
    sorted_index = sorted_index[:, :topn]

    line_str = []
    for line in sorted_index:
        names = []
        for idx in line:
            names.append(gallery_info[int(idx)]['image_name'].replace(
                '.jpg', ''))
        line_str.append(' '.join(names))
    txt = '\n'.join(line_str)

    return txt


def generate_result_txt(dists,
                        query_features,
                        gallery_features,
                        val_dataset_pkl,
                        color_bias=400,
                        type_bias=400,
                        group_threhold=0.05,
                        group_rerank=True):
    with open(val_dataset_pkl, 'rb') as f:
        bmk = pickle.load(f, encoding='bytes')
    txt = generate_txt_from_dists(dists, bmk['reid_gallery'], topn=100)
    if group_rerank:
        features = np.concatenate([query_features, gallery_features], axis=0)
        rerank = Rerank(features, dists, bmk)
        sort_list = get_rerank_result(rerank,
                                      bmk,
                                      color_bias=color_bias,
                                      type_bias=type_bias,
                                      qg_precision=group_threhold)
        txt = generate_txt_from_dists(sort_list,
                                      bmk['reid_gallery'],
                                      topn=100,
                                      be_sorted=True)
    else:
        txt = generate_txt_from_dists(sort_list, bmk['reid_gallery'], topn=100)

    return txt


class Evaluator():
    def __init__(self, val_dataset_pkl):
        with open(val_dataset_pkl, 'rb') as f:
            bmk = pickle.load(f, encoding='bytes')
        query_imgs = bmk['reid_query']
        gallery_imgs = bmk['reid_gallery']
        self.img_to_id_query = self._make_dict(query_imgs, 'image_name',
                                               'vehicle_id')
        self.img_to_id_gallery = self._make_dict(gallery_imgs, 'image_name',
                                                 'vehicle_id')
        self.id_to_img_query = self._make_list_dict(query_imgs, 'vehicle_id',
                                                    'image_name')
        self.id_to_img_gallery = self._make_list_dict(gallery_imgs,
                                                      'vehicle_id',
                                                      'image_name')
        self.query_img_names = [data['image_name'] for data in query_imgs]

    def _make_dict(self, origin, key, value):
        ret = {}
        for item in origin:
            ret[item[key]] = item[value]
        return ret

    def _make_list_dict(self, origin, key, value):
        ret = defaultdict(list)
        for item in origin:
            ret[item[key]].append(item[value])
        return ret

    def _compute_mAP_CMC_single_line(self, ranked_idx, pos_idx):
        ap = 0
        cmc = np.zeros((max(ranked_idx.size, 10)))
        right_mask = np.in1d(ranked_idx, pos_idx)
        arg_pos = np.argwhere(right_mask == True).flatten()
        # total recall
        n_pos = len(pos_idx)
        if n_pos == 0 or len(arg_pos) == 0:
            return ap, cmc
        cmc[arg_pos[0]:] = 1
        for i, pos in enumerate(arg_pos):
            ap_at_r = (i + 1) * 1.0 / (pos + 1)
            ap += ap_at_r * 1.0 / n_pos
        return ap, cmc

    def _compute_mAP_CMC(self, ranked_, pos_):
        n_line = len(ranked_)
        metric = np.zeros((n_line, 101))
        for i in range(n_line):
            np_ranked, np_pos = np.array(ranked_[i]), np.array(pos_[i])
            ap, cmc = self._compute_mAP_CMC_single_line(np_ranked, np_pos)
            metric[i] = np.array([ap] + cmc.tolist())
        effi = metric.mean(axis=0)
        mAP, CMC = effi[0], effi[1:].tolist()
        return mAP, CMC, metric[:, 0].tolist()

    def eval_from_txt(self,
                      txt,
                      topn=100,
                      CMCat=[1, 5, 10, 20, 30, 40, 50],
                      details=False):
        """
        从txt结果计算mAP，top1，top5。
        params:
            txt(string): txt文本。
            topn(int): 取前topn进行mAP与CMC计算。
            CMCat(list): 需要统计的CMC@n的所有n参数。eg.需要统计CMC1和CMC5则输入[1, 5]
        """
        lines = txt.split('\n')
        ranked_ = []
        pos_ = []
        for i, (line,
                line_img_name) in enumerate(zip(lines, self.query_img_names)):
            line_imgs = line.split()[:topn]
            line_imgs = [img + '.jpg' for img in line_imgs]
            if len(line_imgs) == 0:
                print('input missing in line {}'.format(i))
                continue
            ranked_.append(line_imgs)
            pos_.append(
                self.id_to_img_gallery[self.img_to_id_query[line_img_name]])
        mAP, cmc, total_ap = self._compute_mAP_CMC(ranked_, pos_)
        effi = {'mAP': mAP}
        effi.update({'CMC_{}'.format(i): cmc[i - 1] for i in CMCat})
        if details:
            effi['query_AP'] = total_ap

        return effi

    def _get_topn_from_txt(self, txt, topn=1):
        lines = txt.split('\n')
        ret = []
        for _, line in enumerate(lines):
            line_imgs = line.split()[:topn]
            ret.append([img + '.jpg' for img in line_imgs])
        return ret

    def get_topn_neg_confusion(self, txt, attr, nr_class, topn=1,
                               name_list=[]):
        name_item = self.gallery_name_item
        confusion = np.zeros((nr_class, nr_class))
        topn_metrix = self._get_topn_from_txt(txt, topn)
        print(np.array(topn_metrix).shape)
        for query, matches in zip(self.query_img_names, topn_metrix):
            qid = self.img_to_id_query[query]
            q_attr = name_item[query][attr]
            for mname in matches:
                mid = name_item[mname]['vehicle_id']
                m_attr = name_item[mname][attr]
                if qid != mid:
                    confusion[int(q_attr)][int(m_attr)] += 1
        if name_list != []:
            confusion = pd.DataFrame(confusion, index=name_list[:nr_class])
            confusion.columns = name_list[:nr_class]
        return confusion

    def eval_from_txt_path(self,
                           txt_path,
                           topn=100,
                           CMCat=[1, 5, 10, 20, 30, 40, 50]):
        with open(txt_path, 'r') as f:
            txt = f.read()
        assert isinstance(txt, str)
        return self.eval_from_txt(txt, topn, CMCat)


def val_ensemble_model_from_dist_feature(result_name,
                                         val_dataset_pkl,
                                         dists_pkls,
                                         query_features_pkls,
                                         gallery_features_pkls,
                                         model_weights=[1, 1, 1, 1, 1]):
    assert len(dists_pkls) >= 2
    dist_sum = pickle.load(open(dists_pkls[0], 'rb'))
    qfeats = pickle.load(open(query_features_pkls[0], 'rb'))
    gfeats = pickle.load(open(gallery_features_pkls[0], 'rb'))

    dist_sum = model_weights[0]**2 * dist_sum
    qfeats, gfeats = [model_weights[0] * qfeats], [model_weights[0] * gfeats]

    for dist_pkl, query_pkl, gallery_pkl, model_weight in zip(
            dists_pkls[1:], query_features_pkls[1:], gallery_features_pkls[1:],
            model_weights[1:]):
        dist = pickle.load(open(dist_pkl, 'rb'))
        qfeat = pickle.load(open(query_pkl, 'rb'))
        gfeat = pickle.load(open(gallery_pkl, 'rb'))

        dist_sum += dist * (model_weight**2)
        qfeats.append(qfeat * model_weight)
        gfeats.append(gfeat * model_weight)
    query_f = np.concatenate(qfeats, axis=1)
    gallery_f = np.concatenate(gfeats, axis=1)
    txt = generate_result_txt(dist_sum,
                              query_f,
                              gallery_f,
                              val_dataset_pkl,
                              color_bias=400,
                              type_bias=400,
                              group_threhold=0.05,
                              group_rerank=True)
    with open('{}_result.txt'.format(result_name), 'w') as f:
        f.write(txt)
    evl = Evaluator(val_dataset_pkl)
    effi = evl.eval_from_txt(txt)
    with open('{}_map_and_cmc.txt'.format(result_name), 'w') as f:
        f.write(effi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dists', type=str, nargs='+', default=[])
    parser.add_argument('-querys', type=str, nargs='+', default=[])
    parser.add_argument('-gallerys', type=str, nargs='+', default=[])
    args = parser.parse_args()
    val_ensemble_model_from_dist_feature(
        "ensemble",
        '/data/aicity_pkl/benchmark_pytorch.pkl',
        args.dists,
        args.querys,
        args.gallerys,
        model_weights=[1, 1, 1, 1, 1])
