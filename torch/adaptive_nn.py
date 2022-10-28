import torch

class AdaptiveNN(torch.nn.Module):
    def __init__(self, base_model, sim_to_pv_transformer):
        super().__init__()
        self.base_model = base_model
        self.base_model.requires_grad_(False)
        self.base_model.eval()
        
        self.sim_to_pv_transformer = sim_to_pv_transformer
        sim_key_list = self.sim_to_pv_transformer.sim_key_list
        self.input_list = [
            self.sim_to_pv_transformer.key_mapping[ele] for ele in sim_key_list
        ]
        self.output_list = self.base_model.model_output_list + ["geometric_size"]
        
        #self.model_cal = torch.nn.Sequential(
         #       torch.nn.Linear(16, 16)
         #   )
        self.register_parameter("cal_w",torch.nn.Parameter(torch.ones(11)))
        self.register_parameter("cal_b",torch.nn.Parameter(torch.zeros(11)))
        self.register_parameter("cal_w2",torch.nn.Parameter(torch.zeros(11)))

        
    def forward(self, X):
        """forward method that maps PV inputs to real model outputs"""
        X = self.sim_to_pv_transformer.untransform(X)
        

        X_new = X.clone()
        X_new[...,3:6] = (X[...,3:6]**2)*self.cal_w2[0:3] + X[...,3:6]*self.cal_w[0:3] + self.cal_b[0:3]
        X_new[...,7:8] = (X[...,7:8]**2)*self.cal_w2[3:4] + X[...,7:8]*self.cal_w[3:4]+ self.cal_b[3:4]
        X_new[...,9:]  = (X[...,9:]**2)*self.cal_w2[4:] + X[...,9:]*self.cal_w[4:] + self.cal_b[4:]
        
        preds = self.base_model(X_new, return_log=True)
        
        # add softplus transform to outputs
        #preds = torch.nn.functional.softplus(preds)
        preds = torch.clip(preds, 1e-8)
        
        geometric_size = torch.sqrt(preds[..., 0] * preds[...,1]).unsqueeze(-1)
        #print(preds.shape)
        #print(geometric_size)
        new_preds = torch.cat([preds, geometric_size], dim=-1)

        return new_preds