import numpy as np
import cv2

def heatmap(bev_embed,bev_h,bev_w,name):
    """
    bev_feat = [C,H,W]
    """
    bev_feat = bev_embed.squeeze(1).permute(1,0).view(256,bev_h,bev_w)

    indx = bev_feat.detach().cpu().numpy()
    heatmap = np.linalg.norm(indx,ord=2,axis=0)
    heatmap= (heatmap-np.min(heatmap)) / (np.max(heatmap)-np.min(heatmap))

    heatmap = np.uint8(256 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    path = '/home/yuqi_wang/code/Occupancy/work_dirs/visualize/heatmap_'+name+'.png'
    cv2.imwrite(path, heatmap)