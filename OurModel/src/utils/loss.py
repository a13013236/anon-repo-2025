def get_loss(prediction, ground_truth, base_price, mask, batch_size, reg_loss_weight, rank_loss_weight, ic_weight):
   """
   Compute composite loss combining:
   1. Regression loss using Huber loss
   2. Ranking loss for preserving relative order
   3. IC (Information Coefficient) loss for correlation
   """
   device = prediction.device
   all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
   
   return_ratio = torch.div(torch.sub(prediction, base_price), base_price)
   
   # Regression loss using Huber loss for robustness
   huber_loss_fn = nn.HuberLoss(delta=1.0).to(device)
   reg_loss = huber_loss_fn(return_ratio * mask, ground_truth * mask)
   
   # Pairwise ranking loss to maintain correct relative ordering
   pre_pw_dif = torch.sub(return_ratio @ all_one.t(), all_one @ return_ratio.t())
   gt_pw_dif = torch.sub(all_one @ ground_truth.t(), ground_truth @ all_one.t())
   mask_pw = mask @ mask.t()
   rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw))
   
   # IC loss to maximize correlation between predictions and ground truth
   def calculate_ic_loss(pred, gt, mask):
       """Calculate Information Coefficient loss"""
       valid_mask = mask.bool().squeeze()
       pred_valid = pred[valid_mask]
       gt_valid = gt[valid_mask]
       
       if pred_valid.shape[0] < 2:
           return torch.tensor(0.0).to(device)
       
       # Normalize predictions and ground truth
       pred_std = torch.std(pred_valid) + 1e-8
       gt_std = torch.std(gt_valid) + 1e-8
       pred_normalized = (pred_valid - torch.mean(pred_valid)) / pred_std
       gt_normalized = (gt_valid - torch.mean(gt_valid)) / gt_std
       
       # Calculate IC and convert to loss
       ic = torch.mean(pred_normalized * gt_normalized)
       return 1 - ic 
   
   ic_loss = calculate_ic_loss(return_ratio, ground_truth, mask)
   
   # Combine losses using learnable weights
   total_loss = reg_loss_weight * reg_loss + rank_loss_weight * rank_loss + ic_weight * ic_loss
   
   return total_loss, reg_loss_weight * reg_loss, rank_loss_weight * rank_loss, ic_weight * ic_loss, return_ratio


def weighted_kl_divergence(prediction, c_ground_truth, mask, bins_class, bins, kl_weight):
   """
   Compute KL divergence loss between predicted and actual return distributions
   Ensures predicted returns follow realistic market distribution
   """
   assert (
       prediction.shape == c_ground_truth.shape
       and c_ground_truth.shape == mask.shape
   ), "Dimension size does not match"
   
   mask = mask.bool()

   # Setup bins for return distribution
   bins_length = torch.arange(bins_class)
   bins_min, bins_max = bins_length.min().item(), bins_length.max().item()
   
   device = prediction.device
   bins = torch.tensor(bins).to(device)
   
   # Discretize predictions into bins
   c_prediction = torch.bucketize(prediction, bins, right=True)  
   c_prediction = torch.clamp(prediction, min=bins_min, max=bins_max) 

   # Get valid predictions and ground truth values
   valid_predictions = c_prediction[mask].long() 
   valid_ground_truth = c_ground_truth[mask].long() 
   valid_ground_truth = torch.clamp(valid_ground_truth, min=bins_min, max=bins_max)
   
   # Create histograms of predictions and ground truth
   num_classes = bins_class
   pred_histogram = torch.zeros(num_classes, device=device) 
   gt_histogram = torch.zeros(num_classes, device=device)

   pred_histogram.scatter_add_(0, valid_predictions, torch.ones_like(valid_predictions, dtype=torch.float))
   gt_histogram.scatter_add_(0, valid_ground_truth, torch.ones_like(valid_ground_truth, dtype=torch.float))
   
   # Convert histograms to probabilities and compute KL divergence
   eps = 1e-8
   pred_prob = (pred_histogram / (pred_histogram.sum() + eps)).clamp(min=eps)
   gt_prob = (gt_histogram / (gt_histogram.sum() + eps)).clamp(min=eps)

   kl_loss = F.kl_div(pred_prob.log(), gt_prob, reduction='batchmean')
   weighted_kl_loss = kl_loss * kl_weight
   
   return weighted_kl_loss
