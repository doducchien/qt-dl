import torch
import torch.nn as nn
import torch.nn.functional as F

#Hàm loss hỗn hợp của BCE và Dice
class BceDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()


    def forward(self, input:torch.Tensor, target:torch.Tensor):
        #Tính BCE loss trước
        bce_loss = F.binary_cross_entropy_with_logits(
            input=input, 
            target=target, 
            reduction='mean'
        )
        
        #Tính Dice loss

        ## Đầu tiên tính dice score trước
        ### Duỗi thẳng về vector B*C*H*W
        input_flat = input.reshape(-1)
        target_flat = target.reshape(-1)

        ### tính phần giao nhau mềm
        interection = (input * target).sum()
        ## dice_loss = 1 - 2*intersection / (tổng x + y)
        dice_loss = 1 -( 2 * interection) / (input_flat.sum() + target_flat.sum() + 1e-5)

        total_loss = bce_loss + dice_loss
        return total_loss
