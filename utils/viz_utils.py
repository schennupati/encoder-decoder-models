import os
import numpy as np
import matplotlib


def add_images_to_writer(inputs,outputs,predictions,targets,writer,task,epoch,train=False):
    
    state = 'train' if train else 'validation'
    img = inputs[0,:,:,:]
    writer.add_image('Images/{}/Input_image'.format(state),img,epoch,dataformats='CHW')
    if not train:
        matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/Input_image_{}.png'.format(epoch)), (img.permute(1,2,0)).numpy())
    
    if task == 'semantic':
        img = decode_segmap(predictions[task][0,:,:].detach().cpu())
        target = decode_segmap(targets[task][0,:,:].detach().cpu())
        writer.add_image('Images/{}/gt/{}'.format(state,task), target,epoch,dataformats='HWC')
        writer.add_image('Images/{}/det/{}'.format(state,task), img,epoch,dataformats='HWC')
        if not train:
            matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/gt_{}_{}.png'.format(task,epoch)), target)
            matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/det_{}_{}.png'.format(task,epoch)), img)
        
    
    elif task == 'instance_contour':
        img = decode_segmap(predictions[task][0,:,:].detach().cpu(),nc=11,labels=inst_labels)
        contours = img[:,:,0]
        contours[contours>0] = 1

        mask = (predictions['semantic'][0,:,:].cpu().numpy()>=11).astype(np.uint8)#predictions['instance_probs'][0,:,:].cpu().numpy()
        seg_img = decode_segmap(predictions['semantic'][0,:,:].cpu())
        instance_img = get_inst_img_from_contours(mask, seg_img, contours)
        panoptic_img = get_pan_img(mask, seg_img, instance_img)
    
        target = decode_segmap(targets[task][0,:,:].cpu(),nc=11,labels=inst_labels)
        gt_contours = target[:,:,0]
        gt_contours[gt_contours>0] = 1
        gt_mask = targets['instance_probs'][0,:,:].cpu().numpy()
        gt_seg_img = decode_segmap(targets['semantic'][0,:,:].cpu())
        gt_instance_img = get_inst_img_from_contours(gt_mask, gt_seg_img, gt_contours)
        gt_panoptic_img = get_pan_img(gt_mask, gt_seg_img, gt_instance_img)

        writer.add_image('Images/{}/gt/{}'.format(state,task), target,epoch,dataformats='HWC')
        writer.add_image('Images/{}/gt/instance_seg_contour'.format(state,task), gt_instance_img,epoch,dataformats='HWC')
        writer.add_image('Images/{}/gt/panoptic_seg_contour'.format(state,task), gt_panoptic_img,epoch,dataformats='HWC')
        writer.add_image('Images/{}/det/{}'.format(state,task), img,epoch,dataformats='HWC')
        writer.add_image('Images/{}/det/instance_seg_contour'.format(state,task), instance_img,epoch,dataformats='HWC')
        writer.add_image('Images/{}/det/panoptic_seg_contour'.format(state,task), panoptic_img,epoch,dataformats='HWC')
    
        if not train:
            matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/gt_{}_{}.png'.format(task,epoch)), gt_contours.squeeze())
            matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/gt_instance_seg_contour_{}.png'.format(epoch)), gt_instance_img.astype(np.uint8))
            matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/gt_panoptic_seg_contour_{}.png'.format(epoch)), gt_panoptic_img.astype(np.uint8))
            matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/det_{}_{}.png'.format(task,epoch)), contours.squeeze())
            matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/det_instance_seg_contour_{}.png'.format(epoch)), instance_img.astype(np.uint8))
            matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/det_panoptic_seg_contour_{}.png'.format(epoch)), panoptic_img.astype(np.uint8))
        
    elif task == 'instance_regression':
        if train:
            vecs = outputs[task][0,:,:,:].detach().cpu().numpy()
            mask = (predictions['semantic'][0,:,:].cpu().numpy() >= 11).astype(np.uint8) #predictions['instance_probs'][0,:,:].detach().cpu().numpy()
            heatmap = outputs['instance_heatmap'][0,:,:].detach().cpu().numpy()
            img = get_color_inst(vecs.transpose(1,2,0))
        else:
            vecs = outputs[task][0,:,:,:].cpu().numpy()
            mask = (predictions['semantic'][0,:,:].cpu().numpy() >= 11).astype(np.uint8) #predictions['instance_probs'][0,:,:].cpu().numpy()
            heatmap = outputs['instance_heatmap'][0,:,:].cpu().numpy()
            img = get_color_inst(vecs.transpose(1,2,0))
        
        seg_img = decode_segmap(predictions['semantic'][0,:,:].cpu())
        gt_seg_img = decode_segmap(targets['semantic'][0,:,:].cpu())
        gt_vecs = targets[task][0,:,:,:].cpu().numpy()
        gt_mask = targets['instance_probs'][0,:,:].cpu().numpy()
        gt_heatmap = targets['instance_heatmap'][0,:,:].cpu().unsqueeze(0).numpy()
        gt_img = get_color_inst(gt_vecs.transpose(1,2,0))
        gt_inst_img = get_clusters(gt_vecs, gt_mask, gt_heatmap[0])
        inst_img = get_clusters(vecs, mask, np.clip(heatmap[0],0,np.max(heatmap)))
        panoptic_img = get_pan_img(mask, seg_img, inst_img)

        gt_inst_img = get_clusters(gt_vecs, gt_mask, np.clip(gt_heatmap[0],0,np.max(gt_heatmap)))
        gt_panoptic_img = get_pan_img(gt_mask, gt_seg_img, gt_inst_img)



        writer.add_image('Images/{}/gt/{}_vecs'.format(state,task),gt_img,epoch,dataformats='HWC')
        #writer.add_image('Images/{}/gt_{}_dy'.format(state,task),gt_dy,epoch,dataformats='HWC')
        writer.add_image('Images/{}/det/{}_vecs'.format(state,task),img,epoch,dataformats='HWC')
        #writer.add_image('Images/{}/det_t_{}_dy'.format(state,task),det_dy,epoch,dataformats='HWC')
        writer.add_image('Images/{}/det_{}_instance'.format(state,task),inst_img,epoch,dataformats='HWC')
        writer.add_image('Images/{}/det_{}_panoptic'.format(state,task),panoptic_img,epoch,dataformats='HWC')
        if not train:

            matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/gt_{}_vecs_{}.png'.format(task,epoch)), gt_img)
            matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/det_{}_vecs_{}.png'.format(task,epoch)), img)
            matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/det_{}_instance_{}.png'.format(task,epoch)), inst_img.astype(np.uint8))
            matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/det_{}_panoptic_{}.png'.format(task,epoch)), panoptic_img.astype(np.uint8))
            matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/gt_{}_instance_{}.png'.format(task,epoch)), gt_inst_img.astype(np.uint8))
            matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/gt_{}_instance_{}.png'.format(task,epoch)), gt_panoptic_img.astype(np.uint8))
        
    elif task == 'instance_heatmap':
        if train:
            img = outputs[task][0,:,:].detach().cpu().numpy()
        else:
            img = outputs[task][0,:,:].cpu().numpy()
        gt_img = targets[task][0,:,:].cpu().unsqueeze(0).numpy()
        writer.add_image('Images/{}/gt/{}'.format(state,task),gt_img,epoch)
        writer.add_image('Images/{}/det/{}'.format(state,task),np.clip(img,0,np.max(img)),epoch)
        #if not train:
        #    matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/{}/gt/{}.png'.format(state,task)), gt_img)
        #    matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/{}/det/{}.png'.format(state,task)), img)
            
    elif task == 'instance_probs':
        if train:
            img = decode_segmap(predictions[task][0,:,:].detach().cpu(),nc=2,labels = prob_labels)
        else:
            img = decode_segmap(predictions[task][0,:,:].cpu(),nc=2,labels = prob_labels)
        target = decode_segmap(targets[task][0,:,:].cpu(),nc=2,labels = prob_labels)
        writer.add_image('Images/{}/gt/{}'.format(state,task), target,epoch,dataformats='HWC')
        writer.add_image('Images/{}/det/{}'.format(state,task), img,epoch,dataformats='HWC')

        #if not train:
        #    matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/{}/gt/{}.png'.format(state,task)), target)
        #    matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/{}/det/{}.png'.format(state,task)), img)
        
    
    elif task == 'disparity':
        if train:
            img = predictions[task][0,0,:,:].detach().cpu().unsqueeze(0).numpy().astype(np.uint8)
        else:
            img = predictions[task][0,0,:,:].cpu().unsqueeze(0).numpy().astype(np.uint8)
        gt_img = targets[task][0,0,:,:].cpu().unsqueeze(0).numpy().astype(np.uint8)
        writer.add_image('Images/{}/gt/{}'.format(state,task),gt_img,epoch)
        writer.add_image('Images/{}/det/{}'.format(state,task),img,epoch)
        if not train:
            matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/{}/gt/{}.png'.format(state,task)), gt_img)
            matplotlib.image.imsave(os.path.join(RESULTS_DIR,'Images/{}/det/{}.png'.format(state,task)), img)
