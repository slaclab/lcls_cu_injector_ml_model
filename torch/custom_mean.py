import torch
from botorch import fit_gpytorch_model
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.means.mean import Mean


class CustomMean(Mean):
    def __init__(
        self,
        model,
        model_input_names,
        model_output_names,
        input_names,
        output_name,
        gp_input_transform,
        gp_outcome_transform,
        static_values,
        static_value_names,
        multiplier=1
    ):
        """
        Custom prior mean for a GP based on an arbitrary model

        :param model: torch.nn.Module representation of the model
        :param model_input_names: list of feature names for model input
        :param model_output_names: list of feature names for model output
        :param input_names: list of feature names for input to prior mean
        :param output_name: feature name for output of prior mean
        :param gp_input_transform: module used to transform inputs in the GP
        :param gp_outcome_transform: module used to transform outcomes in the GP
        """

        super(CustomMean, self).__init__()
        self.input_names = input_names
        self.model_input_names = model_input_names
        self.NN_model = model
        self.NN_model.eval()
        self.NN_model.requires_grad_(False)
        self.static_values = static_values
        self.multiplier = multiplier

        self.gp_input_transform = gp_input_transform
        self.gp_outcome_transform = gp_outcome_transform

        # get ordering of column names to reshape the input x
        self.input_indicies = []
        all_names = input_names + static_value_names
        for ele in model_input_names:
            self.input_indicies.append(all_names.index(ele))
            
        # print([self.input_indicies])
        # get model output index
        self.output_index = model_output_names.index(output_name)

    def forward(self, x, verbose=False):
        """
        takes in input_transform(x) from GP, returns outcome_transform(y)
        x columns are specified by input_names
        """
        self.gp_outcome_transform.eval()
        self.gp_input_transform.eval()

        # untransform inputs
        x = self.gp_input_transform.untransform(x)

        # hard coded static values
        x_static = torch.ones(*x.shape[:-1],len(self.static_values))
        for i in range(len(self.static_values)):
            x_static[...,i] = x_static[...,i] * self.static_values[i]
            
        x = torch.cat((x, x_static),dim=-1)
        
        # need to reorder columns
        x = x[..., self.input_indicies]
        
        if verbose:
            print(x[0])
        self.NN_model.eval()
        m = self.NN_model(x)  # real x |-> real y
        
        if verbose:
            print(m[0])
            print("output_index",self.output_index)
        # grab the correct output column from the model
        m = m[..., self.output_index].unsqueeze(-1) * self.multiplier
        
        m = self.gp_outcome_transform(m)[0]  # real y -> standardized y

        self.gp_outcome_transform.train()
        self.gp_input_transform.train()
        return m.squeeze(dim=-1)


def create_nn_prior_model(
    data, 
    vocs, 
    model,
    static_values,
    static_value_names,
    multiplier = 1.0
):
    tkwargs = {"dtype": torch.double, "device": "cpu"}
    input_data, objective_data, constraint_data = vocs.extract_data(data)
    train_X = torch.tensor(input_data.to_numpy(), **tkwargs)
    
    
    assert vocs.n_objectives == 1
    objective_name = vocs.objective_names[0]
    input_names = vocs.variable_names

    gp_input_transform = Normalize(
        vocs.n_variables, bounds=torch.tensor(vocs.bounds, **tkwargs)
    )
    gp_outcome_transform = Standardize(1)

    # construct prior mean
    model_input_names = model.input_list
    model_output_names = model.output_list
    input_names = input_names
    output_name = objective_name

    prior_mean = CustomMean(
        model,
        model_input_names,
        model_output_names,
        input_names,
        output_name,
        gp_input_transform,
        gp_outcome_transform,
        static_values,
        static_value_names,
        multiplier
    )
    
    #prior_mean.requires_grad_(False)

    
    #print(prior_mean.input_names)
    #print(prior_mean.model_input_names)
    objective_models = []

    train_Y = torch.tensor(
        objective_data[objective_name].to_numpy(), **tkwargs
    ).unsqueeze(-1)

    #print(train_X.shape)
    #print(train_Y.shape)
    
    objective_models.append(
        SingleTaskGP(
            train_X,
            train_Y,
            input_transform=gp_input_transform,
            outcome_transform=gp_outcome_transform,
            mean_module=prior_mean
        )
    )
    mll = ExactMarginalLogLikelihood(
        objective_models[-1].likelihood, objective_models[-1]
    )
    fit_gpytorch_model(mll)

    return ModelListGP(*objective_models)
