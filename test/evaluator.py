import torch

class Evaluator:
    def eval(self, input_dict):
        y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']
        y_pred = torch.cat([y_pred_pos.view(-1,1), y_pred_neg], dim = 1)
        argsort = torch.argsort(y_pred, dim = 1, descending = True)
        ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
        ranking_list = ranking_list[:, 1] + 1
        hits1_list = (ranking_list <= 1).to(torch.float)
        hits3_list = (ranking_list <= 3).to(torch.float)
        hits10_list = (ranking_list <= 10).to(torch.float)
        mrr_list = 1./ranking_list.to(torch.float)

        return mrr_list, hits1_list, hits3_list, hits10_list