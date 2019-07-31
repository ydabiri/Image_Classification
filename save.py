import torch
def save(model,filepath):
    checkpoint = {'epochs': model.epochs,
                  'state_dict': model.state_dict(),
                 'class_to_idx': model.class_to_idx}


    torch.save(checkpoint, filepath)