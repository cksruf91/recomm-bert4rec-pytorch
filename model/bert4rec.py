import torch
from torch import Tensor
from transformers import AlbertConfig, AlbertModel


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

        # albert_configuration = AlbertConfig() # xxlage
        self.albert = AlbertModel(self.albert_configuration, add_pooling_layer=False)
        self.linear = torch.nn.Linear(hidden_size, vocab_size, bias=True)
        
        self.to(device)
        parameter_size(self)

    def forward(self, seq: Tensor, attention: Tensor) -> Tensor:
        out = self.albert(input_ids=seq, attention_mask=attention)
        return self.linear(out.last_hidden_state).reshape(-1, self.vocab_size)
    
    def predict(self, seq: Tensor, attention: Tensor) -> Tensor:
        out = self.albert(input_ids=seq, attention_mask=attention)
        return self.linear(out.last_hidden_state)[:, -1, :]


def parameter_size(torch_model):
    param_size = 0
    for param in torch_model.parameters():
        param_size += param.nelement() * param.element_size()
    print(f'model size : {param_size / 1024 / 1024:1.5f} mb')
