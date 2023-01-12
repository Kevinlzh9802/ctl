

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.info_detail = {}

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def update_detail(self, detail_information_dict):
        for i in detail_information_dict:
            if i not in self.info_detail.keys():
                self.info_detail[i] = detail_information_dict[i]
            else:
                self.info_detail[i] = self.merge_info(self.info_detail[i], detail_information_dict[i])

    def merge_info(self, ori_dict, new_dict):
        res_dict = {}
        res_dict['sum'] = ori_dict['sum'] + new_dict['sum']
        res_dict['count'] = ori_dict['count'] + new_dict['count']
        res_dict['multi_num'] = ori_dict['multi_num'] + new_dict['multi_num']
        res_dict['avg'] = round(res_dict['sum']/res_dict['count'], 3)
        res_dict['multi_rate'] = round(res_dict['multi_num']/res_dict['count'], 3)
        return res_dict

    def get_avg_detail(self):
        total_name_list = list(self.info_detail.keys())

        finest_name_list = []
        coarse_name_list = []
        for i in total_name_list:
            if i < 0:
                coarse_name_list.append(i)
            else:
                finest_name_list.append(i)
        if coarse_name_list:
            coarse_avg = round(sum([self.info_detail[i]['sum'] for i in coarse_name_list]) / sum(
                [self.info_detail[i]['count'] for i in coarse_name_list]), 3)
        else:
            coarse_avg = None

        if finest_name_list:

            finest_avg = round(sum([self.info_detail[i]['sum'] for i in finest_name_list])/sum([self.info_detail[i]['count'] for i in finest_name_list]), 3)
        else:
            finest_avg = None

        return {'total_avg': round(self.avg, 3), 'finest_avg': finest_avg, 'coarse_avg': coarse_avg, 'finest_count': len(finest_name_list), 'coarse_count': len(coarse_name_list)}
