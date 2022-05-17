import torch
from torch import Tensor
from transformers import AlbertConfig, AlbertModel, BertModel, BertConfig


class Albert4Rec(torch.nn.Module):

    def __init__(self, vocab_size: int, seq_length: int, inter_dim: int = 3072, hidden_size: int = 64 * 12,
                 num_head: int = 12, device=torch.device('cpu')):
        super().__init__()
        self.vocab_size = vocab_size
        # AlBERT
        self.albert_configuration = AlbertConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_head,
            intermediate_size=inter_dim,
            vocab_size=vocab_size,
            max_position_embeddings=seq_length
        )

        self.albert = AlbertModel(self.albert_configuration, add_pooling_layer=False)
        self.gelu = torch.nn.GELU()
        self.linear = torch.nn.Linear(hidden_size, vocab_size, bias=True)
        
        self.device = device
        self.to(self.device)
        parameter_size(self)

    def forward(self, seq: Tensor, attention: Tensor) -> Tensor:
        out = self.albert(input_ids=seq, attention_mask=attention)
        out = self.gelu(out.last_hidden_state)
        return self.linear(out)
    
    def predict(self, seq: Tensor, attention: Tensor) -> Tensor:
        out = self.forward(seq, attention)

        dummie = torch.zeros((out.size(0), 1), device=self.device)
        attention = torch.concat((attention, dummie), dim=1)
        indices = torch.argmin(attention, dim=1, keepdim=False) - 1
        
        return out[torch.arange(out.size(0), device=self.device), indices]

    
class bert4Rec(torch.nn.Module):

    def __init__(self, vocab_size: int, seq_length: int, inter_dim: int = 3072, hidden_size: int = 64 * 12,
                 num_head: int = 12, num_layer: int = 3, device=torch.device('cpu')):
        super().__init__()
        self.vocab_size = vocab_size
        # BERT
        self.configuration = BertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=seq_length,
            hidden_size=hidden_size,
            num_hidden_layers=num_layer,
            num_attention_heads=num_head,
            intermediate_size=inter_dim,
            hidden_act='gelu',
            attention_probs_dropout_prob=0.2,
            hidden_dropout_prob=0.2,
            initializer_range=0.02
            
        )

        self.bert = BertModel(self.configuration, add_pooling_layer=False)
        self.gelu = torch.nn.GELU()
        self.linear = torch.nn.Linear(hidden_size, vocab_size, bias=True)
        
        self.device = device
        self.to(self.device)
        parameter_size(self)

    def forward(self, seq: Tensor, attention: Tensor) -> Tensor:
        out = self.bert(input_ids=seq, attention_mask=attention)
        out = self.gelu(out.last_hidden_state)
        return self.linear(out)
    
    def predict(self, seq: Tensor, attention: Tensor) -> Tensor:
        out = self.forward(seq, attention)

        dummie = torch.zeros((out.size(0), 1), device=self.device)
        attention = torch.concat((attention, dummie), dim=1)
        indices = torch.argmin(attention, dim=1, keepdim=False) - 1
        
        return out[torch.arange(out.size(0), device=self.device), indices]
        

def parameter_size(torch_model):
    param_size = 0
    for param in torch_model.parameters():
        param_size += param.nelement() * param.element_size()
    print(f'model size : {param_size / 1024 / 1024:1.5f} mb')
