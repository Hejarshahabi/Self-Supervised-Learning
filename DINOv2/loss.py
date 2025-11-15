import torch.nn.functional as F

def dino_loss(student_out, teacher_out, student_temp=0.1, teacher_temp=0.04):
    student_log_probs = F.log_softmax(student_out / student_temp, dim=-1)
    teacher_probs = F.softmax(teacher_out / teacher_temp, dim=-1).detach()
    return -(teacher_probs * student_log_probs).sum(dim=-1).mean()
