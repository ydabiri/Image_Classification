import torch
import json

def predict(model, image, top_k,device_name,categy_names):
     #This part is used to transfer the classes to names.
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    device = torch.device("cuda:0" if device_name=="gpu" else "cpu")
    model.to(device)
    
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.view(1,image.shape[0], image.shape[1],image.shape[2])
    image.to(device)
    
    probabilities = torch.exp(model.forward(image))
    top_probabilities, top_indexs = probabilities.topk(top_k,dim=1) 
    print(model.class_to_idx)
    idx_to_class = {idx: flower for flower, idx in model.class_to_idx.items()}
    top_flowers = [idx_to_class[x.item()] for x in top_indexs[0]]  
    flower_names = [cat_to_name[x] for x in top_flowers]  
    print(top_indexs)
    #print(flower_names)
    print(top_probabilities)     
    return [top_flowers,top_probabilities]      