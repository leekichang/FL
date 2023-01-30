import torch
from torch.nn.functional import nll_loss
from torch.optim import SGD
from torch.utils.data import DataLoader
from models.simpleNet import Net

class Client():
    def __init__(self, client_id:str, dataset):
        self.id = client_id
        self.model = Net()
        self.optimizer = SGD(self.model.parameters(), lr=0.1)
        self.loss = nll_loss
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    def local_train(self):
        pass
    
    def upload_model(self):
        pass
    
    def local_test(self):
        pass
    
    
# # Perform Federated Learning on the client
# for round_num in range(NUM_ROUND):
#     # Perform local training on the client
#     for batch_num, (batch_data, batch_target) in enumerate(dataloader):
#         optimizer.zero_grad()
#         output = model(batch_data)
#         loss = loss_fn(output, batch_target)
#         loss.backward()
#         optimizer.step()

#     # Send the model to the server
#     model_params = model.state_dict()
#     send_model_to_server(model_params)

#     # Receive the updated model from the server
#     updated_model_params = receive_updated_model_from_server()
#     model.load_state_dict(updated_model_params)

# # Perform inference on the client
# inference_data = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
# inference_output = model(inference_data)
# print(inference_output)