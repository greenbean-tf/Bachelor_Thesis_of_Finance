import torch
import torch.nn as nn
from tqdm import tqdm as tqdm
import utils

class GumIndLoss(nn.Module):
    def forward(self, y_pred, y_true):
        Revert = torch.unsqueeze(y_true[:, 0], 1).long()
        Rtop = torch.unsqueeze(y_true[:, 1],1)
        Top = torch.unsqueeze(y_true[:, 2], 1)
        Close = torch.unsqueeze(y_true[:, 3], 1)

        r = y_pred[0]
        rtop = torch.unsqueeze(torch.exp(y_pred[1][:, 0]),1)
        sigma_rtop = torch.unsqueeze(torch.exp(y_pred[1][:, 1]),1)
        top = torch.unsqueeze(torch.exp(y_pred[1][:, 2]), 1)
        sigma_top = torch.unsqueeze(torch.exp(y_pred[1][:, 3]), 1)
        close = torch.unsqueeze(torch.exp(y_pred[1][:, 4]), 1)
        sigma_close = torch.unsqueeze(torch.exp(y_pred[1][:, 5]), 1)

        eps = torch.tensor(1e-4)
        rtop_z = torch.clamp((rtop-Rtop)/torch.max(sigma_rtop,eps),-10,10)
        Rtop_Gumbel_loss = ((torch.exp(rtop_z)
                                - rtop_z
                                + torch.log(torch.max(sigma_rtop,eps))) )
        top_z = (top - Top) / torch.max(sigma_top, eps)
        Top_Gumbel_loss = ((torch.exp(top_z)
                                       - top_z
                                       + torch.log(torch.max(sigma_top, eps))) )
        Close_Normal_loss = 1/2 * (torch.square((close - Close) / torch.max(sigma_close, eps))
                                     + torch.log(torch.square(sigma_close)))

        prob_loss = Revert * (Rtop_Gumbel_loss + Top_Gumbel_loss + Close_Normal_loss)

        return (nn.CrossEntropyLoss()(r, torch.squeeze(Revert))
                + torch.mean(prob_loss))

class NormIndLoss(nn.Module):
    def forward(self, y_pred, y_true):
        Revert = torch.unsqueeze(y_true[:, 0], 1).long()
        Rtop = torch.unsqueeze(y_true[:, 1],1)
        Top = torch.unsqueeze(y_true[:, 2], 1)
        Close = torch.unsqueeze(y_true[:, 3], 1)

        r = y_pred[0]
        rtop = torch.unsqueeze(torch.exp(y_pred[1][:, 0]),1)
        sigma_rtop = torch.unsqueeze(torch.exp(y_pred[1][:, 1]),1)
        top = torch.unsqueeze(torch.exp(y_pred[1][:, 2]), 1)
        sigma_top = torch.unsqueeze(torch.exp(y_pred[1][:, 3]), 1)
        close = torch.unsqueeze(torch.exp(y_pred[1][:, 4]), 1)
        sigma_close = torch.unsqueeze(torch.exp(y_pred[1][:, 5]), 1)

        eps = torch.tensor(1e-4)

        Rtop_Normal_loss = 1/2 * (torch.square((rtop - Rtop) / torch.max(sigma_rtop, eps))
                                       + torch.log(torch.square(sigma_rtop)))
        Top_Normal_loss = 1/2 * (torch.square((top - Top) / torch.max(sigma_top, eps))
                                       + torch.log(torch.square(sigma_top)))
        Close_Normal_loss = 1/2 * (torch.square((close - Close) / torch.max(sigma_close, eps))
                                     + torch.log(torch.square(sigma_close)))

        prob_loss = Revert*(Rtop_Normal_loss + Top_Normal_loss + Close_Normal_loss)

        return (nn.CrossEntropyLoss()(r, torch.squeeze(Revert))
                + torch.mean(prob_loss))

class GaussCopGumLoss(nn.Module):
    def forward(self, y_pred, y_true):
        Revert = torch.unsqueeze(y_true[:, 0], 1).long()
        Rtop = torch.unsqueeze(y_true[:, 1],1)
        Top = torch.unsqueeze(y_true[:, 2], 1)
        Close = torch.unsqueeze(y_true[:, 3], 1)

        r = y_pred[0]
        rtop = torch.unsqueeze(torch.exp(y_pred[1][:, 0]),1)
        sigma_rtop = torch.unsqueeze(torch.exp(y_pred[1][:, 1]),1)
        top = torch.unsqueeze(torch.exp(y_pred[1][:, 2]), 1)
        sigma_top = torch.unsqueeze(torch.exp(y_pred[1][:, 3]), 1)
        close = torch.unsqueeze(torch.exp(y_pred[1][:, 4]), 1)
        sigma_close = torch.unsqueeze(torch.exp(y_pred[1][:, 5]), 1)

        r12 = y_pred[2][:, 0]
        r13 = y_pred[2][:, 1]
        r23 = y_pred[2][:, 2]

        R = torch.eye(3).unsqueeze(0).expand(len(y_true), -1, -1).to('cuda')

        R[:, 0, 1] = r12
        R[:, 1, 0] = r12
        R[:, 0, 2] = r13
        R[:, 2, 0] = r13
        R[:, 1, 2] = r23
        R[:, 2, 1] = r23

        eps = torch.tensor(1e-4).to('cuda')

        detR = torch.unsqueeze(torch.det(R),1)

        q1 = utils.GumbelCDF(Rtop, u=rtop, s=sigma_rtop, eps=eps, do_torch=True)
        q2 = utils.GumbelCDF(Top, u=top, s=sigma_top, eps=eps, do_torch=True)
        q3 = utils.NormalCDF(Close, u=close, s=sigma_close, eps=eps, do_torch=True)

        q = torch.concatenate((q1, q2, q3), dim=1)

        quantile_vector = utils.Normal_distribution_q(0, 1, q, True).unsqueeze(1)
        quantile_vector = torch.clamp(quantile_vector, -3, 3)
        quantile_vector_T = torch.transpose(quantile_vector, 1, 2)
        exp_term = 1/2 * (
            torch.matmul(torch.matmul(quantile_vector, ((torch.linalg.inv(R) - torch.eye(3).to('cuda')))),
                         quantile_vector_T))
        exp_term = torch.squeeze(exp_term,dim=1)

        rtop_z = (rtop - Rtop) / torch.max(sigma_rtop, eps)
        Rtop_Gumbel_loss = (torch.exp(rtop_z)
                                      - rtop_z
                                      + torch.log(torch.max(sigma_rtop, eps)))
        top_z = (top - Top) / torch.max(sigma_top, eps)
        Top_Gumbel_loss = (torch.exp(top_z)
                                     - top_z
                                     + torch.log(torch.max(sigma_top, eps)))
        Close_Normal_loss = 1/2 * (torch.square((close - Close) / torch.max(sigma_close, eps))
                                       + torch.log(torch.square(sigma_close)))

        cGaussian = 1/2*torch.log(torch.max(detR,eps)) + exp_term

        Copula_loss = Revert*(cGaussian + Rtop_Gumbel_loss + Top_Gumbel_loss + Close_Normal_loss)
        
        if Copula_loss < 0 :
            print("Revert:", Revert)
            print("Labels: Rtop=", Rtop, " Top=", Top, " Close=", Close)

            print("Pred μ: rtop=", rtop, " top=", top, " close=", close)
            print("Pred σ: sigma_rtop=", sigma_rtop, " sigma_top=", sigma_top, " sigma_close=", sigma_close)

            print("Correlations: r12=", r12, " r13=", r13, " r23=", r23)
            print("R matrix:", R)
            print("detR:", detR)

            print("q1=", q1, " q2=", q2, " q3=", q3)
            print("quantile_vector:", quantile_vector)

            print("Rtop_Gumbel_loss=", Rtop_Gumbel_loss)
            print("Top_Gumbel_loss=", Top_Gumbel_loss)
            print("Close_Normal_loss=", Close_Normal_loss)
            print("cGaussian=", cGaussian)

            print("Copula_loss (per sample)=", Copula_loss)

        return nn.CrossEntropyLoss()(r, torch.squeeze(Revert, dim=1)) + torch.mean(Copula_loss)

class GaussCopNormLoss(nn.Module):
    def forward(self, y_pred, y_true):
        Revert = torch.unsqueeze(y_true[:, 0], 1).long()
        Rtop = torch.unsqueeze(y_true[:, 1],1)
        Top = torch.unsqueeze(y_true[:, 2], 1)
        Close = torch.unsqueeze(y_true[:, 3], 1)

        r = y_pred[0]
        rtop = torch.unsqueeze(torch.exp(y_pred[1][:, 0]),1)
        sigma_rtop = torch.unsqueeze(torch.exp(y_pred[1][:, 1]),1)
        top = torch.unsqueeze(torch.exp(y_pred[1][:, 2]), 1)
        sigma_top = torch.unsqueeze(torch.exp(y_pred[1][:, 3]), 1)
        close = torch.unsqueeze(torch.exp(y_pred[1][:, 4]), 1)
        sigma_close = torch.unsqueeze(torch.exp(y_pred[1][:, 5]), 1)

        r12 = y_pred[2][:, 0]
        r13 = y_pred[2][:, 1]
        r23 = y_pred[2][:, 2]

        R = torch.eye(3).unsqueeze(0).expand(len(y_true), -1, -1).to('cuda')

        R[:, 0, 1] = r12
        R[:, 1, 0] = r12
        R[:, 0, 2] = r13
        R[:, 2, 0] = r13
        R[:, 1, 2] = r23
        R[:, 2, 1] = r23

        eps = torch.tensor(1e-4).to('cuda')

        detR = torch.unsqueeze(torch.det(R), 1)

        q1 = utils.NormalCDF(Rtop, u=rtop, s=sigma_rtop, eps=eps, do_torch=True)
        q2 = utils.NormalCDF(Top, u=top, s=sigma_top, eps=eps, do_torch=True)
        q3 = utils.NormalCDF(Close, u=close, s=sigma_close, eps=eps, do_torch=True)

        q = torch.concatenate((q1, q2, q3), dim=1)

        quantile_vector = utils.Normal_distribution_q(0, 1, q, True).unsqueeze(1)
        quantile_vector = torch.clamp(quantile_vector, -3, 3)
        quantile_vector_T = torch.transpose(quantile_vector, 1, 2)
        exp_term = 1 / 2 * (
            torch.matmul(torch.matmul(quantile_vector, ((torch.linalg.inv(R) - torch.eye(3).to('cuda')))),
                         quantile_vector_T))
        exp_term = torch.squeeze(exp_term, dim=1)

        Rtop_Normal_loss = 1/2 * (torch.square((rtop - Rtop) / torch.max(sigma_rtop, eps))
                                             + torch.log(torch.square(sigma_rtop)))
        Top_Normal_loss = 1/2 *  (torch.square((top - Top) / torch.max(sigma_top, eps))
                                            + torch.log(torch.square(sigma_top)))
        Close_Normal_loss = 1/2 * (torch.square((close - Close) / torch.max(sigma_close, eps))
                                     + torch.log(torch.square(sigma_close)))

        cGaussian = 1/2 * torch.log(torch.max(detR, eps)) + exp_term

        Copula_loss = Revert * (cGaussian + Rtop_Normal_loss + Top_Normal_loss + Close_Normal_loss)

        return nn.CrossEntropyLoss()(r, torch.squeeze(Revert)) + torch.mean(Copula_loss)