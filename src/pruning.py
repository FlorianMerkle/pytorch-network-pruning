import torch
def _prune_random_local_unstruct(model, ratio):
    def prune(model, ratio, weights_list, masks_list):
        params = model.state_dict()
        for i, _ in enumerate(weights_list):
            weights = params[weights_list[i]]
            mask = params[masks_list[i]]
            shape = weights.shape
            flat_weights = weights.flatten()
            flat_mask = mask.flatten()

            no_of_weights_to_prune = int(round(ratio * len(flat_weights)))
            non_zero_weights = torch.nonzero(flat_weights)
            no_of_weights_to_prune_left = int(no_of_weights_to_prune - (len(flat_weights) - len(non_zero_weights)) )
            non_zero_weights = non_zero_weights[torch.randperm(non_zero_weights.nelement())] #shuffle tensor
            indices_to_delete = non_zero_weights[:no_of_weights_to_prune_left]

            for idx_to_delete in indices_to_delete:
                flat_mask[idx_to_delete] = 0
                flat_weights[idx_to_delete] = 0

            params[weights_list[i]] = weights.view(shape)
            params[masks_list[i]] = mask.view(shape)
        model.load_state_dict(params)
        return params

    weights = prune(model, ratio, model.conv_weights, model.conv_masks)
    weights = prune(model, ratio, model.fully_connected_weights, model.fully_connected_masks)
    return weights

def _prune_magnitude_global_unstruct(model, ratio):
    params = model.state_dict()
    flat_weights = torch.empty(0)
    flat_masks = torch.empty(0)
    all_masks = model.conv_masks + model.fully_connected_masks
    all_weights = model.conv_weights+model.fully_connected_weights
    for i, _ in enumerate(all_weights):
        flat_weights = torch.cat((flat_weights, params[all_weights[i]].view(-1)))
        flat_masks = torch.cat((flat_masks, params[all_masks[i]].view(-1)))
    
    no_of_weights_to_prune = int(round(len(flat_weights)*ratio))
    indices_to_delete = torch.abs(flat_weights).argsort(0)[:no_of_weights_to_prune]
    for idx_to_delete in indices_to_delete:
        flat_masks[idx_to_delete] = 0
        flat_weights[idx_to_delete] = 0
    
    z = 0
    for i, _ in enumerate(all_weights):
        params[all_weights[i]] = flat_weights[z:z + params[all_weights[i]].nelement()].reshape(params[all_weights[i]].shape)
        params[all_masks[i]] = flat_masks[z:z + params[all_weights[i]].nelement()].reshape(params[all_weights[i]].shape)
        z = z + params[all_weights[i]].nelement()
    model.load_state_dict(params)
    return params


# rework kernel/filter wise
def _prune_random_local_struct(model, ratio, prune_dense_layers=False, structure='kernel'):
    def prune_filters(model, ratio, params):
        for i, _ in enumerate(model.conv_weights):
            # shape = (3,3,64,128)
            weight_layer = params[model.conv_weights[i]]
            mask_layer = params[model.conv_masks[i]]
            shape = weight_layer.shape
            no_of_filters = shape[0]
            no_of_filters_to_prune = int(round(ratio * no_of_filters))
            non_zero_filters = torch.nonzero(torch.tensor([torch.sum(filt) for filt in weight_layer]))
            no_of_filters_to_prune_left = no_of_filters_to_prune - (len(weight_layer) - len(non_zero_filters))
            
            non_zero_filters = non_zero_filters[torch.randperm(non_zero_filters.nelement())] #shuffle tensor
            filters_to_prune = non_zero_filters[:no_of_filters_to_prune_left]
            
            for filter_to_prune in filters_to_prune:
                weight_layer[filter_to_prune] = torch.zeros_like(weight_layer[filter_to_prune])
                mask_layer[filter_to_prune] = torch.zeros_like(mask_layer[filter_to_prune])        
        return params
    
    def prune_kernels(model, ratio, weights):
        for i, _ in enumerate(model.conv_weights):
            # shape = (3,3,64,128)
            weight_layer = params[model.conv_weights[i]]
            mask_layer = params[model.conv_masks[i]]
            shape = weight_layer.shape
            kernels = weight_layer.view(shape[0]*shape[1], shape[2], shape[3])
            masks = mask_layer.view(shape[0]*shape[1], shape[2], shape[3])
            
            no_of_kernels = kernels.shape[0]
            no_of_kernels_to_prune = int(round(ratio * no_of_kernels))
            non_zero_kernels = torch.nonzero(torch.tensor([torch.sum(kernel) for kernel in kernels]))
            no_of_kernels_to_prune_left = no_of_kernels_to_prune - (len(kernels) - len(non_zero_kernels))
            non_zero_kernels_randomized = non_zero_kernels[torch.randperm(non_zero_kernels.nelement())] #shuffle tensor
            kernels_to_prune = non_zero_kernels_randomized[:no_of_kernels_to_prune_left]
            
            for kernel_to_prune in kernels_to_prune:
                kernels[kernel_to_prune] = torch.zeros_like(kernels[kernel_to_prune])
                masks[kernel_to_prune] = torch.zeros_like(masks[kernel_to_prune])
            params[model.conv_weights[i]] = kernels.view(shape)
            params[model.conv_masks[i]] = masks.view(shape)
        return params
    
    def prune_dense(model, ratio, weights):
        raise Exception('not implemented')
        return
    params = model.state_dict()
    if structure == 'filter':
        params = prune_filters(model, ratio, params)
    if structure == 'kernel':
        params = prune_kernels(model, ratio, params)
    if prune_dense_layers==True:
        params = prune_dense(model, ratio, params)
    model.load_state_dict(params)
    return params

def _prune_random_global_struct(model, ratio, prune_dense_layers=False):
    raise Warning('Not yet implemented')
    return False

def _prune_magnitude_local_struct(model, ratio, prune_dense_layers=False, structure='kernel'):
    def prune_filters(model, ratio, params):
        for i, _ in enumerate(model.conv_weights):
            weight_layer = params[model.conv_weights[i]]
            mask_layer = params[model.conv_masks[i]]
            vals = []
            # shape = (128,64,3,3)
            no_of_filters = weight_layer.shape[0]
            no_of_filters_to_prune = int(round(ratio * no_of_filters))
            for single_filter in weight_layer:
                #shape of single_filter = (64,3,3)
                vals.append(torch.sum(torch.abs(single_filter)))
            filters_to_prune = torch.argsort(torch.tensor(vals))[:no_of_filters_to_prune]
            for filter_to_prune in filters_to_prune:
                weight_layer[filter_to_prune] = torch.zeros_like(weight_layer[filter_to_prune])
                mask_layer[filter_to_prune] = torch.zeros_like(mask_layer[filter_to_prune])

                # shape = (128,64,3,3)
            params[model.conv_weights[i]] = weight_layer
            params[model.conv_masks[i]] = mask_layer
        return params
    
    def prune_kernels(model, ratio, params):
        for i, _ in enumerate(model.conv_weights):
            weight_layer = params[model.conv_weights[i]]
            mask_layer = params[model.conv_masks[i]]
            shape = weight_layer.shape
            kernels = weight_layer.view(shape[0]*shape[1], shape[2], shape[3])
            masks = mask_layer.view(shape[0]*shape[1], shape[2], shape[3])
            no_of_kernels = kernels.shape[0]
            no_of_kernels_to_prune = int(round(ratio * no_of_kernels))

            vals = []
            for kernel in kernels:
                vals.append(torch.sum(torch.abs(kernel)))
            kernels_to_prune = torch.argsort(torch.tensor(vals))[:no_of_kernels_to_prune]
            for kernel_to_prune in kernels_to_prune:
                kernels[kernel_to_prune] = torch.zeros_like(kernels[kernel_to_prune])
                masks[kernel_to_prune] = torch.zeros_like(masks[kernel_to_prune])
            
            params[model.conv_weights[i]] = kernels.view(shape)
            params[model.conv_masks[i]] = masks.view(shape)
        return params
    
    def prune_dense_layers(model, ratio, weights):
        raise Exception('not implemented')
        return
    
    params = model.state_dict()
    if structure == 'kernel':
        params = prune_kernels(model,ratio, params)
    if structure == 'filter':
        params = prune_filters(model,ratio, params)
    
    if prune_dense_layers==True:
        params = prune_dense(model, ratio, params)
        
    model.load_state_dict(params)
    return params
    
def _prune_magnitude_global_struct(model, ratio, prune_dense_layers=False,structure='kernel'):
    def prune_filters(model, ratio, params):
        all_filters = torch.empty(0)
        all_masks = torch.empty(0)
        vals = []
        for i, _ in enumerate(model.conv_weights):
            weight_layer = params[model.conv_weights[i]]
            mask_layer = params[model.conv_masks[i]]
            vals = vals + [torch.sum(torch.abs(single_filter)) / single_filter.nelement() for single_filter in weight_layer]
            #all_filters = torch.cat((all_filters,weight_layer))
            all_filters = list(all_filters) +  list(weight_layer)
            all_masks = list(all_masks) + list(mask_layer)
            
        no_of_filters_to_prune = int(round(ratio * len(vals)))
        filters_to_prune = torch.argsort(torch.tensor(vals))[:no_of_filters_to_prune]
        for filter_to_prune in filters_to_prune:
            all_filters[filter_to_prune] = torch.zeros_like(all_filters[filter_to_prune]) 
            all_masks[filter_to_prune] = torch.zeros_like(all_filters[filter_to_prune])

        z = 0
        for i, _ in enumerate(model.conv_weights):
            shape = params[model.conv_weights[i]].shape
            params[model.conv_weights[i]] = torch.cat(all_filters[z:z + shape[0]]).view(shape)
            params[model.conv_masks[i]] = torch.cat(all_masks[z:z + shape[0]]).view(shape)
            z = z + shape[0]
        return params
    
    def prune_kernels(model, ratio, params):
        all_kernels = torch.empty(0)
        all_masks = torch.empty(0)
        vals = []
        for i, _ in enumerate(model.conv_weights):
            weight_layer = params[model.conv_weights[i]]
            #print(model.conv_masks)
            mask_layer = params[model.conv_masks[i]]
            shape = weight_layer.shape
            kernels = weight_layer.view(shape[0]*shape[1], shape[2], shape[3])
            masks = mask_layer.view(shape[0]*shape[1], shape[2], shape[3])
            
            vals = vals + [torch.sum(torch.abs(kernel)) / kernel.nelement() for kernel in kernels]
            #all_filters = torch.cat((all_filters,weight_layer))
            all_kernels = list(all_kernels) +  list(kernels)
            all_masks = list(all_masks) + list(masks)
        
        no_of_kernels_to_prune = int(round(ratio * len(vals)))
        kernels_to_prune = torch.argsort(torch.tensor(vals))[:no_of_kernels_to_prune]
        for kernel_to_prune in kernels_to_prune:
            all_kernels[kernel_to_prune] = torch.zeros_like(all_kernels[kernel_to_prune]) 
            all_masks[kernel_to_prune] = torch.zeros_like(all_masks[kernel_to_prune])
        
        z = 0
        for i, _ in enumerate(model.conv_weights):
            shape = params[model.conv_weights[i]].shape
            params[model.conv_weights[i]] = torch.cat(all_kernels[z:z + shape[0]*shape[1]]).view(shape)
            params[model.conv_masks[i]] = torch.cat(all_masks[z:z + shape[0]*shape[1]]).view(shape)
            z = z + shape[0]*shape[1]
        return params
        
    def prune_dense_layers(model, ratio):
        raise Exception('not implemented')
        return
    
    params = model.state_dict()
    if structure == 'filter':
        params = prune_filters(model, ratio, params)
    if structure == 'kernel':
        params = prune_kernels(model, ratio, params)
    if prune_dense_layers==True:
        params = prune_dense_layers(model, ratio, params)
    model.load_state_dict(params)
    return params

def _prune_magnitude_local_unstruct(model, ratio, scope='layer'):
    def prune_conv_layers_layerwise(model, ratio, params):
        for i, _ in enumerate(model.conv_weights):
            weight_layer = params[model.conv_weights[i]]
            mask_layer = params[model.conv_masks[i]]
            shape = weight_layer.shape
            flat_weights = weight_layer.view(-1)
            flat_masks = mask_layer.view(-1)
            no_of_weights_to_prune = int(round(ratio * len(flat_weights)))
            indices_to_delete = torch.argsort(torch.abs(flat_weights))[:no_of_weights_to_prune]
            for idx_to_delete in indices_to_delete:
                flat_masks[idx_to_delete] = 0
                flat_weights[idx_to_delete] = 0
            params[model.conv_weights[i]] = flat_weights.view(shape)
            params[model.conv_masks[i]] = flat_masks.view(shape)
        return params
    
    def prune_conv_layers_filterwise(model, ratio, params):
        for i, _ in enumerate(model.conv_weights):
            weight_layer = params[model.conv_weights[i]]
            mask_layer = params[model.conv_masks[i]]
            for j,_ in enumerate(weight_layer):
                filt = weight_layer[j]
                filter_mask = mask_layer[j]
                shape = weight_layer[j].shape
                flat_weights = filt.view(-1)
                flat_mask = filter_mask.view(-1)
                no_of_weights_to_prune = int(round(ratio * len(filt)))
                indices_to_delete = torch.argsort(torch.abs(filt))[:no_of_weights_to_prune]
                for idx_to_delete in indices_to_delete:
                    flat_mask[idx_to_delete] = 0
                    flat_weights[idx_to_delete] = 0
                
                weight_layer[j] = flat_weights.view(shape)
                mask_layer[j] = flat_mask.view(shape)
                
            params[model.conv_weights[i]] = weight_layer.view(shape)
            params[model.conv_masks[i]] = mask_layer.view(shape)
        return params
        
    def prune_dense_layers(model, ratio, params):
        for i, _ in enumerate(model.fully_connected_weights):
            weight_layer = params[model.fully_connected_weights[i]]
            mask_layer = params[model.fully_connected_masks[i]]
            flat_weights = weight_layer.view(-1)
            flat_mask = mask_layer.view(-1)
            shape = weight_layer.shape

            no_of_weights_to_prune = int(round(len(flat_weights)*ratio))
            indices_to_delete = torch.argsort(torch.abs(flat_weights))[:no_of_weights_to_prune]
            for idx_to_delete in indices_to_delete:
                flat_mask[idx_to_delete] = 0
                flat_weights[idx_to_delete] = 0
            params[model.fully_connected_weights[i]] = flat_weights.view(shape)
            params[model.fully_connected_masks[i]] = flat_mask.view(shape)
        return params
    
    params = model.state_dict()
    if scope == 'layer':
        params = prune_conv_layers_layerwise(model,ratio, params)
    if scope == 'filter':
        params = prune_conv_layers_filterwise(model,ratio, params)
    if scope != 'filter' and scope != 'layer':
        raise Exception('scope should be one of "layer" and "filter"')
    params = prune_dense_layers(model,ratio, params)
    model.load_state_dict(params)
    return params