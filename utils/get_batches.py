import torch
import random

def get_my_batch(split,loaded_dicts, batch_size,stoi, device, define_method, question_max_length, ans_max_length, pattern_list):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    # print(device)
    question_data=loaded_dicts['question']
    ans_data=loaded_dicts['ans']
    
    ix = torch.randint(len(ans_data), (batch_size,))
    
    x = torch.from_numpy(question_data[ix,:])
    y = torch.from_numpy(ans_data[ix,:])
    # print('x',x.shape)
    question_end=torch.nonzero(x == stoi['<END_Q>'], as_tuple=False)

    x_list=[]
    y_list=[]
    if split=='train':
        for idx in range(batch_size):
            selected_batch=0 
            end_p=0
            if 'cot' in define_method:
                if define_method == 'cot_hex' or define_method == 'cot_pet':
                    selected_batch = random.randint(1, 2)
                elif define_method == 'cot_hex2':
                    selected_batch = random.randint(1, 4)
                else:
                    selected_batch = random.randint(1, 3)
            if 'cot' in define_method:
                if define_method == 'cot_hex' or define_method == 'cot_pet':
                    end_p=1
                elif define_method == 'cot_hex2':
                    end_p=3
                else:
                    end_p=2
                if selected_batch < end_p:
                    for num_i in range(1,end_p+1):
                        # num_i=selected_batch
                        ans_begin=torch.nonzero(y[idx,:] == stoi[f'<P{num_i}>'], as_tuple=False)
                        ans_end=torch.nonzero(y[idx,:] == stoi[f'<P{num_i}_END>'], as_tuple=False)
                        selected_vis=torch.randint(ans_begin,ans_end+2,(1,))
                        ans=y[idx,:selected_vis]
                        question_nodes=x[idx,-4:]
                        if idx >= question_end.shape[0]:
                            selected_idx=question_end.shape[0] - 1
                        else:selected_idx=idx
                        question=x[selected_idx,:question_end[selected_idx,1]]
                        provided_ans=y[selected_idx,:selected_vis-1]
                        pads_tensor_x=torch.ones(question_max_length-question.shape[0]-question_nodes.shape[0])*stoi['<PAD>']
                        provide_ans_pads=torch.ones(ans_max_length-provided_ans.shape[0])*stoi['<PAD>']
                        
                        x_list.append(torch.cat((question,pads_tensor_x.to(torch.int64),question_nodes.to(torch.int64),provided_ans,provide_ans_pads.to(torch.int64)),0).unsqueeze(0))

                        pads_graph=torch.ones(question_max_length)*stoi['<PAD>']
                        pads_tensor_y=torch.ones(ans_max_length-ans.shape[0])*stoi['<PAD>']
                        y_list.append(torch.cat((pads_graph.to(torch.int64),ans,pads_tensor_y.to(torch.int64)),0).unsqueeze(0))
            else:
                ans_begin=1
            if 'cot' not in define_method or selected_batch>=end_p:
                if 'cot' in define_method:
                    ans_begin=torch.nonzero(y[idx,:] == stoi[f'<ANS>'], as_tuple=False)
                ans_end=torch.nonzero(y[idx,:] == stoi[f'<END>'], as_tuple=False)
                if ans_end.numel() == 0:
                    ans_end = torch.tensor([[ans_max_length-1]],dtype=torch.int64)
                # print('ans_end',ans_end)
                selected_vis=torch.randint(ans_begin,ans_end+2,(1,))
                ans=y[idx,:selected_vis]
                # print(y)
                question_nodes=x[idx,-4:]
                if idx >= question_end.shape[0]:
                    selected_idx=question_end.shape[0] - 1
                else:selected_idx=idx
                question=x[selected_idx,:question_end[selected_idx,1]]
                provided_ans=y[selected_idx,:selected_vis-1]
                pads_tensor_x=torch.ones(question_max_length-question.shape[0]-question_nodes.shape[0])*stoi['<PAD>']
                provide_ans_pads=torch.ones(ans_max_length-provided_ans.shape[0])*stoi['<PAD>']
                
                x_list.append(torch.cat((question,pads_tensor_x.to(torch.int64),question_nodes.to(torch.int64),provided_ans,provide_ans_pads.to(torch.int64)),0).unsqueeze(0))

                pads_graph=torch.ones(question_max_length)*stoi['<PAD>']
                pads_tensor_y=torch.ones(ans_max_length-ans.shape[0])*stoi['<PAD>']
                y_list.append(torch.cat((pads_graph.to(torch.int64),ans,pads_tensor_y.to(torch.int64)),0).unsqueeze(0))
                # else:
                #     continue
                
    else:
        for idx in range(batch_size):
            if 'cot' in define_method:
                ans_begin=torch.nonzero(y[idx,:] == stoi[f'<ANS>'], as_tuple=False)
                ans_end=torch.nonzero(y[idx,:] == stoi[f'<END>'], as_tuple=False)
                if ans_end.numel() == 0:
                    ans_end = torch.tensor([[ans_max_length-1]],dtype=torch.int64)
                selected_vis=torch.randint(ans_begin,ans_end+2,(1,))
                ans=y[idx,:selected_vis]
                question_nodes=x[idx,-4:]
                if idx >= question_end.shape[0]:
                    selected_idx=question_end.shape[0] - 1
                else:selected_idx=idx
                question=x[selected_idx,:question_end[selected_idx,1]]
                provided_ans=y[selected_idx,:selected_vis-1]
                pads_tensor_x=torch.ones(question_max_length-question.shape[0]-question_nodes.shape[0])*stoi['<PAD>']
                provide_ans_pads=torch.ones(ans_max_length-provided_ans.shape[0])*stoi['<PAD>']
                
                x_list.append(torch.cat((question,pads_tensor_x.to(torch.int64),question_nodes.to(torch.int64),provided_ans,provide_ans_pads.to(torch.int64)),0).unsqueeze(0))

                pads_graph=torch.ones(question_max_length)*stoi['<PAD>']
                pads_tensor_y=torch.ones(ans_max_length-ans.shape[0])*stoi['<PAD>']
                y_list.append(torch.cat((pads_graph.to(torch.int64),ans,pads_tensor_y.to(torch.int64)),0).unsqueeze(0))
            else:
                ans_end = torch.nonzero(y[idx, :] == stoi['<END>'], as_tuple=False)# [0, 0]
                if ans_end.numel() == 0:
                    ans_end = torch.tensor([[ans_max_length-1]],dtype=torch.int64)
                if len(pattern_list)>1:
                    ans_end = ans_end[0, 0]
                    candidates = torch.arange(1, ans_end + 2)
                    weights = torch.linspace(1.0, ans_max_length/2, steps=candidates.shape[0])
                    selected_vis_samples = candidates[torch.multinomial(weights, 3)]
                else:
                    selected_vis_samples = torch.randint(1, ans_end + 2, (2,))
                # [0,1]
                for selected_vis in selected_vis_samples:
                    ans=y[idx,:selected_vis]
                    question_nodes=x[idx,-4:]
                    if idx >= question_end.shape[0]:
                        selected_idx=question_end.shape[0] - 1
                    else:selected_idx=idx
                    question=x[selected_idx,:question_end[selected_idx,1]]
                    provided_ans=y[selected_idx,:selected_vis-1]
                    pads_tensor_x=torch.ones(question_max_length-question.shape[0]-question_nodes.shape[0])*stoi['<PAD>']
                    provide_ans_pads=torch.ones(ans_max_length-provided_ans.shape[0])*stoi['<PAD>']
                    
                    x_list.append(torch.cat((question,pads_tensor_x.to(torch.int64),question_nodes.to(torch.int64),provided_ans,provide_ans_pads.to(torch.int64)),0).unsqueeze(0))

                    pads_graph=torch.ones(question_max_length)*stoi['<PAD>']
                    pads_tensor_y=torch.ones(ans_max_length-ans.shape[0])*stoi['<PAD>']
                    y_list.append(torch.cat((pads_graph.to(torch.int64),ans,pads_tensor_y.to(torch.int64)),0).unsqueeze(0))

    x=torch.cat(x_list,0)
    y=torch.cat(y_list,0)
    y_mask=stoi['<PAD>']
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    
    if 'cuda' in device_type :
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y,y_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), y_mask# .pin_memory().to(device, non_blocking=True)
    else:
        x, y,y_mask = x.to(device), y.to(device),y_mask# .to(device)
    return x, y,y_mask



